#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Inspired from https://github.com/facebookresearch/DPR/tree/main

"""
 Command line tool to get dense results and validate them
"""

import argparse
import glob
import json
import logging
import pickle
import time
import zlib
from typing import List, Tuple, Dict, Iterator
import csv 

import numpy as np
import torch
from torch import Tensor as T
from torch import nn

import numpy as np
import torch
from torch import nn

from transformers import set_seed
from malicious_ir.utils import log_args, str2bool
from malicious_ir.retrieval.utils import load_encoder, generate_embedding, instantiate_index, setup_cfg_gpu, setup_for_distributed_mode, move_to_device, Tensorizer, QUERY_PROMPTS
from malicious_ir.retrieval.faiss_indexers import DenseIndexer
from malicious_ir.retrieval.validation_utils import calculate_matches

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)

parser = argparse.ArgumentParser(description="Generates embedding for retrieval corpus.")
parser.add_argument(
    "--encoder_type",
    action="store",
    type=str,
    required=True,
    choices=["q", "ctx"],
    help="ctx or q encoder type. Especially necessary for traditional DPR-like retrievers.",
)
parser.add_argument(
    "--encoder_class_name",
    action="store",
    type=str,
    required=True,
    choices=["dpr", "contriever", "nv", "bge", "llm2vec", "promptriever"],
    help="Encoder model class. load_encoder and generate_embedding functions should be implemented for new classes.",
)
parser.add_argument(
    "--encoder_name",
    action="store",
    type=str,
    required=True,
    help="Encoder model name.",
)
parser.add_argument(
    "--encoder_name_2",
    action="store",
    type=str,
    default=None,
    help="Encoder model name 2. This arg is used for cases like LLM2Vec that use two separate names for different peft phases.",
)
parser.add_argument(
    "--seed",
    action="store",
    type=int,
    default=0,
    help="Seed.",
)
parser.add_argument(
    "--qa_dataset",
    action="store",
    type=str,
    required=True,
    help="Path to the file with questions.",
)
parser.add_argument(
    "--encoded_ctx_files",
    action="store",
    nargs='+',
    required=True,
    help="Glob paths to encoded passages (from retrieval_generate_embeddings tool).",
)
parser.add_argument(
    "--ctx_sources",
    action="store",
    nargs='+',
    required=True,
    help="Paths to corpus (from retrieval_prepare_corpora_for_encoding tool).",
)
parser.add_argument(
    "--eval_unsafety",
    action="store",
    type=str2bool,
    default=True,
    required=True,
    help="Should we evaluate the unsafety of the retrieval or the normal performance?",
)
parser.add_argument(
    "--relevant_idx",
    action="store",
    type=str,
    help="Path to mapping file from question ids to relevant malicious chunks.",
)
parser.add_argument(
    "--malicious_prefix",
    action="store",
    type=str,
    help="Prefix for dataset name for malicious retrieved info.",
)
parser.add_argument(
    "--instruction_key",
    action="store",
    type=str,
    default=None,
    help="The specific instruction key to be used for each sample, e.g., for Promtriever. This key should be available in the data dictionary.",
)
parser.add_argument(
    "--out_file",
    action="store",
    type=str,
    required=True,
    help="Path to dir to where the retrieved information is saved.",
)
parser.add_argument(
    "--n_docs",
    action="store",
    type=int,
    default=100,
    help="Number of retrieved info.",
)
parser.add_argument(
    "--indexer",
    action="store",
    type=str,
    default="flat",
    help="indexer.",
)
parser.add_argument(
    "--index_path",
    action="store",
    type=str,
    default=None,
    help="index_path.",
)
parser.add_argument(
    "--batch_size",
    action="store",
    type=int,
    default=128,
    help="batch_size.",
)
parser.add_argument(
    "--validation_workers",
    action="store",
    type=int,
    default=16,
    help="validation_workers.",
)
parser.add_argument(
    "--match",
    action="store",
    type=str,
    default="string",
    choices=["string", "regex"],
    help="match type.",
)
parser.add_argument(
    "--local_rank",
    action="store",
    type=int,
    default=-1,
    help="local_rank.",
)
parser.add_argument(
    "--device",
    action="store",
    type=str,
    default=None,
    help="device.",
)
parser.add_argument(
    "--distributed_world_size",
    action="store",
    type=str,
    default=None,
    help="distributed_world_size.",
)
parser.add_argument(
    "--distributed_port",
    action="store",
    type=str,
    default=None,
    help="distributed_port.",
)
parser.add_argument(
    "--no_cuda",
    action="store",
    type=bool,
    default=False,
    help="no_cuda.",
)
parser.add_argument(
    "--n_gpu",
    action="store",
    type=int,
    default=None,
    help="n_gpu.",
)
parser.add_argument(
    "--output_dtype",
    action="store",
    type=str,
    default="float32",
    choices=["float32", "float16"],
    help="The embedding dtype.",
)
parser.add_argument(
    "--fp16",
    action="store",
    type=bool,
    default=False,
    help="fp16.",
)
parser.add_argument(
    "--fp16_opt_level",
    action="store",
    type=str,
    default="O1",
    help="fp16_opt_level.",
)
parser.add_argument(
    "--global_loss_buf_sz",
    action="store",
    type=int,
    default=150000,
    help="global_loss_buf_sz.",
)


def generate_question_vectors(
    question_encoder: torch.nn.Module,
    tensorizer: Tensorizer,
    questions: List[str],
    bsz: int,
    device: str = None,
    question_encoder_class_name: str = None,
    output_dtype: str = "float32",
) -> T:
    n = len(questions)
    query_vectors = []

    q_prefix, q_suffix = "", ""
    if question_encoder_class_name in QUERY_PROMPTS:
        q_prefix = QUERY_PROMPTS[question_encoder_class_name]["q"]
        q_suffix = QUERY_PROMPTS[question_encoder_class_name]["q_suffix"]

    with torch.no_grad():
        for j, batch_start in enumerate(range(0, n, bsz)):
            batch_questions = questions[batch_start : batch_start + bsz]

            with torch.no_grad():
                out = generate_embedding(batch_questions, tensorizer, question_encoder, prefix=q_prefix, suffix=q_suffix, encoder_class_name=question_encoder_class_name, device=device)

            if output_dtype == "float16":
                out = out.to(torch.float16)
            query_vectors.extend(out.cpu().split(1, dim=0))

            if len(query_vectors) % 100 == 0:
                logging.info("Encoded queries %d", len(query_vectors))

    query_tensor = torch.cat(query_vectors, dim=0)
    logging.info("Total encoded queries tensor %s", query_tensor.size())
    assert query_tensor.size(0) == len(questions)
    return query_tensor


class DenseRetriever(object):
    def __init__(self, question_encoder: nn.Module, batch_size: int, tensorizer: Tensorizer, device: str = None, question_encoder_class_name: str = None, output_dtype: str = "float32"):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.device = device
        self.question_encoder_class_name = question_encoder_class_name
        self.output_dtype = output_dtype

    def generate_question_vectors(self, questions: List[str]) -> T:

        bsz = self.batch_size
        return generate_question_vectors(
            self.question_encoder,
            self.tensorizer,
            questions,
            bsz,
            device=self.device,
            question_encoder_class_name=self.question_encoder_class_name,
            output_dtype=self.output_dtype,
        )


class LocalFaissRetriever(DenseRetriever):
    """
    Does passage retrieving over the provided index and question encoder
    """

    def __init__(
        self,
        question_encoder: nn.Module,
        batch_size: int,
        tensorizer: Tensorizer,
        index: DenseIndexer,
        device: str = None,
        question_encoder_class_name: str = None,
        output_dtype: str = "float32",
    ):
        super().__init__(question_encoder, batch_size, tensorizer, device=device, question_encoder_class_name=question_encoder_class_name, output_dtype=output_dtype)
        self.index = index

    def index_encoded_data(
        self,
        vector_files: List[str],
        buffer_size: int,
        path_id_prefixes: List = None,
    ):
        """
        Indexes encoded passages takes form a list of files
        :param vector_files: file names to get passages vectors from
        :param buffer_size: size of a buffer (amount of passages) to send for the indexing at once
        :return:
        """
        buffer = []
        for i, item in enumerate(iterate_encoded_files(vector_files, path_id_prefixes=path_id_prefixes)):
            buffer.append(item)
            if 0 < buffer_size == len(buffer):
                self.index.index_data(buffer)
                buffer = []
        self.index.index_data(buffer)
        logging.info("Data indexing completed.")

    def get_top_docs(self, query_vectors: np.array, top_docs: int = 100) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        results = self.index.search_knn(query_vectors, top_docs)
        logging.info("index search time: %f sec.", time.time() - time0)
        self.index = None
        return results


def validate_unsafety(
    gold_chunks: Dict[object, List[int]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    question_ids: List[int],
    prefix: str = ""
) -> List[List[bool]]:
    logging.info("validating passages.")
    
    out_gold, out_malicious = [], []
    for i, res in enumerate(result_ctx_ids):
        out_gold.append([])
        out_malicious.append([])
        res_golds = [prefix + "_" + str(g) for g in gold_chunks[str(question_ids[i])]]
        retrieved_ids = res[0]
        for retrieved_id in retrieved_ids:
            out_malicious[-1].append(retrieved_id.startswith(prefix))
            out_gold[-1].append(retrieved_id in res_golds)

    gold_top_k_hits = [0] * len(result_ctx_ids[0][0])
    malicious_top_k_hits = [0] * len(result_ctx_ids[0][0])
    at_least_one_malicious_top_k_hits = [0] * len(result_ctx_ids[0][0])

    for o_m_q in out_malicious:
        for j in range(len(o_m_q)):
            malicious_top_k_hits[j] += sum(o_m_q[:j + 1])
            at_least_one_malicious_top_k_hits[j] += sum(o_m_q[:j + 1]) > 0

    for o_m_q in out_gold:
        for j in range(len(o_m_q)):
            gold_top_k_hits[j] += sum(o_m_q[:j + 1]) > 0
                
    malicious_top_k_hits = [v / (len(result_ctx_ids) * (i+1)) for i, v in enumerate(malicious_top_k_hits)]
    at_least_one_malicious_top_k_hits = [v / len(result_ctx_ids) for v in at_least_one_malicious_top_k_hits]
    gold_top_k_hits = [v / len(result_ctx_ids) for v in gold_top_k_hits]
    logging.info("Validation results: top k documents accuracy:\n\tMalicious Frequency: %s\n\tAt Least One Malicious: %s\n\tGold Malicious: %s", malicious_top_k_hits, at_least_one_malicious_top_k_hits, gold_top_k_hits)
    return out_gold

def validate_performance(
    passages: Dict[object, Tuple[str, str]],
    answers: List[List[str]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
) -> List[List[bool]]:
    logging.info("validating passages. size=%d", len(passages))
    match_stats = calculate_matches(passages, answers, result_ctx_ids, workers_num, match_type)
    top_k_hits = match_stats.top_k_hits

    logging.info("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logging.info("Validation results: top k documents hits accuracy %s", top_k_hits)
    return match_stats.questions_doc_hits

def save_results(
    passages: Dict[object, Tuple[str, str]],
    questions: List[str],
    question_ids: List[int],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    per_question_hits: List[List[bool]],
    out_file: str,
    write: bool=True
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    for i, q in enumerate(questions):
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)

        results_item = {
            "id": question_ids[i],
            "question": q,
            "ctxs": [
                {
                    "id": results_and_scores[0][c],
                    "text": docs[c],
                    "score": scores[c],
                    "is_gold": hits[c],
                }
                for c in range(ctxs_num)
            ],
        }

        merged_data.append(results_item)

    if write:
        with open(out_file, "w") as writer:
            writer.write(json.dumps(merged_data, indent=4) + "\n")
        logging.info("Saved results * scores  to %s", out_file)

def iterate_encoded_files(vector_files: list, path_id_prefixes: List = None) -> Iterator[Tuple]:
    for i, file in enumerate(vector_files):
        logging.info("Reading file %s", file)
        id_prefix = None
        if path_id_prefixes:
            id_prefix = path_id_prefixes[i]
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                doc = list(doc)
                if id_prefix and not str(doc[0]).startswith(id_prefix):
                    doc[0] = id_prefix + str(doc[0])
                yield doc

def get_all_passages(ctx_sources):
    all_passages = {}
    for ctx_src in ctx_sources:
        with open(ctx_src) as f:
            for row in f:
                row = json.loads(row)
                all_passages[row["id"]] = row["contents"]
        logging.info("Loaded ctx data: %d", len(all_passages))

    if len(all_passages) == 0:
        raise RuntimeError("No passages data found. Please specify ctx_file param properly.")
    return all_passages


def main():
    
    cfg = parser.parse_args()
    set_seed(cfg.seed)

    cfg = setup_cfg_gpu(cfg)

    logging.info("Dense retrieval:")
    log_args(cfg)

    tensorizer, encoder = load_encoder(cfg.encoder_name, cfg.encoder_class_name, cfg.encoder_type, encoder_name_2=cfg.encoder_name_2)

    _encoder, _ = setup_for_distributed_mode(
        encoder if cfg.encoder_class_name != "promptriever" else encoder.model,
        None,
        cfg.device,
        cfg.n_gpu,
        cfg.local_rank,
        cfg.fp16,
        cfg.fp16_opt_level,
    )
    if cfg.encoder_class_name != "promptriever":
        encoder = _encoder
        encoder.eval()
        vector_size = encoder.config.hidden_size if hasattr(encoder, "config") else encoder.module.config.hidden_size
    else:
        encoder.model = _encoder
        encoder.model.eval()
        vector_size = encoder.model.config.hidden_size if hasattr(encoder.model, "config") else encoder.model.module.config.hidden_size

    logging.info("Encoder vector_size=%d", vector_size)

    if not cfg.qa_dataset:
        logging.warning("Please specify qa_dataset to use.")
        return

    logging.info("qa_dataset: %s", cfg.qa_dataset)
    questions = []
    question_ids = []
    question_answers = None
    with open(cfg.qa_dataset) as f:
        if cfg.qa_dataset.endswith("jsonl"):
            for d in f:
                d = json.loads(d)
                if cfg.instruction_key:
                    questions.append(f"{d['instruction'].strip()} {d[cfg.instruction_key].strip()}")
                else:
                    questions.append(d["instruction"])
                question_ids.append(d["id_"])
        elif cfg.qa_dataset.endswith("csv"):
            f = csv.reader(f, delimiter="\t")
            question_answers = []
            for i, d in enumerate(f):
                questions.append(d[0])
                question_ids.append(i)
                question_answers.append(eval(d[1]))
        else:
            raise NotImplementedError(f"The file processing for this file format is not supported yet.")

    total_queries = len(questions)
    logging.info("questions len %d", total_queries)

    index = instantiate_index(cfg.indexer)
    logging.info("Local Index class %s ", type(index))
    index_buffer_sz = index.buffer_size
    index.init_index(vector_size)
    retriever = LocalFaissRetriever(encoder, cfg.batch_size, tensorizer, index, device=cfg.device, question_encoder_class_name=cfg.encoder_class_name, output_dtype=cfg.output_dtype)

    questions_tensor = retriever.generate_question_vectors(questions)

    index_path = cfg.index_path
    if index_path and index.index_exists(index_path):
        logging.info("Index path: %s", index_path)
        retriever.index.deserialize(index_path)
    else:
        ctx_files_patterns = cfg.encoded_ctx_files
        logging.info("ctx_files_patterns: %s", ctx_files_patterns)
        
        input_paths = []
        for i, pattern in enumerate(ctx_files_patterns):
            pattern_files = glob.glob(pattern)
            print(pattern, "---", pattern_files)
            input_paths.extend(pattern_files)
        print(input_paths)
        logging.info("Reading all passages data from files: %s", input_paths)
        retriever.index_encoded_data(input_paths, index_buffer_sz)
        if index_path:
            retriever.index.serialize(index_path)

    # get top k results
    top_results_and_scores = retriever.get_top_docs(questions_tensor.numpy(), cfg.n_docs)
    

    all_passages = get_all_passages(cfg.ctx_sources)
    if cfg.eval_unsafety:
        gold_chunks = json.load(open(cfg.relevant_idx))
        gold_chunk_prefix = cfg.malicious_prefix
        gold_top_k_hits = validate_unsafety(
            gold_chunks,
            top_results_and_scores,
            question_ids,
            prefix=gold_chunk_prefix
        )
    else:
        gold_top_k_hits = validate_performance(
            all_passages,
            question_answers,
            top_results_and_scores,
            cfg.validation_workers,
            cfg.match,
        )

    if cfg.out_file:
            save_results(
                all_passages,
                questions,
                question_ids,
                top_results_and_scores,
                gold_top_k_hits,
                cfg.out_file,
                write=True,
            )

if __name__ == "__main__":
    main()