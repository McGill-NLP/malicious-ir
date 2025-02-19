#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Inspired from https://github.com/facebookresearch/DPR/tree/main

"""
 Command line tool that produces embeddings for a large documents base based on the pretrained ctx & question encoders
 Supposed to be used in a 'sharded' way to speed up the process.
"""
import argparse
import logging
import math
import os
import pathlib
import pickle
import json
from tqdm import tqdm
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from transformers import set_seed
from malicious_ir.utils import log_args
from malicious_ir.retrieval.utils import load_encoder, generate_embedding, setup_cfg_gpu, setup_for_distributed_mode, move_to_device, Tensorizer, QUERY_PROMPTS


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
    help="ctx or q encoder type",
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
    "--out_file",
    action="store",
    type=str,
    required=True,
    help="Path to dir to where the embeddings are saved.",
)
parser.add_argument(
    "--ctx_src",
    action="store",
    type=str,
    required=True,
    help="path to the corpus.",
)
parser.add_argument(
    "--shard_id",
    action="store",
    type=int,
    default=0,
    help="Shard id of the processed shard. 0-based.",
)
parser.add_argument(
    "--num_shards",
    action="store",
    type=int,
    default=1,
    help="Number of all shards of the processed corpus.",
)
parser.add_argument(
    "--batch_size",
    action="store",
    type=int,
    default=64,
    help="batch_size.",
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

def gen_ctx_vectors(
    cfg: argparse.ArgumentParser,
    ctx_rows: List[Tuple[object]],
    encoder: nn.Module,
    tensorizer: Tensorizer,
) -> List[Tuple[object, np.array]]:
    n = len(ctx_rows)
    bsz = cfg.batch_size
    total = 0
    results = []

    ctx_prefix = ""
    if cfg.encoder_class_name in QUERY_PROMPTS:
        ctx_prefix = QUERY_PROMPTS[cfg.encoder_class_name]["ctx"]
    
    for j, batch_start in tqdm(enumerate(range(0, n, bsz))):
        batch = ctx_rows[batch_start : batch_start + bsz]
        batch_texts = [ctx[1] for ctx in ctx_rows[batch_start : batch_start + bsz]]
        batch_ids = [ctx[0] for ctx in batch]
        
        out = generate_embedding(batch_texts, tensorizer, encoder, prefix=ctx_prefix, encoder_class_name=cfg.encoder_class_name, device=cfg.device)
        out = out.cpu()

        if cfg.output_dtype == "float16":
            out = out.to(torch.float16)

        assert len(batch_ids) == out.size(0)
        total += len(batch_ids)

        results.extend([(batch_ids[i], out[i].view(-1).numpy()) for i in range(out.size(0))])

        if total % 10 == 0:
            logging.info("Encoded passages %d", total)
    
    logging.info("Encoded total passages %d", total)
    return results

def load_data(file):
    ctxs = {}
    with open(file) as ifile:
        for row in tqdm(ifile):
            row = json.loads(row)
            ctxs[row["id"]] = row["contents"]
    return ctxs
            

def main():

    cfg = parser.parse_args()
    set_seed(cfg.seed)

    cfg = setup_cfg_gpu(cfg)

    logging.info("Corpus encoding:")
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
    else:
        encoder.model = _encoder
        encoder.model.eval()

    # load weights from the model file
    logging.info("reading data source: %s", cfg.ctx_src)
    all_passages_dict = load_data(cfg.ctx_src)
    all_passages = [(k, v) for k, v in all_passages_dict.items()]

    shard_size = math.ceil(len(all_passages) / cfg.num_shards)
    start_idx = cfg.shard_id * shard_size
    end_idx = start_idx + shard_size

    logging.info(
        "Producing encodings for passages range: %d to %d (out of total %d)",
        start_idx,
        end_idx,
        len(all_passages),
    )
    shard_passages = all_passages[start_idx:end_idx]

    data = gen_ctx_vectors(cfg, shard_passages, encoder, tensorizer)

    file = cfg.out_file + "_" + str(cfg.shard_id) + "_vec"
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    logging.info("Writing results to %s" % file)
    with open(file, mode="wb") as f:
        pickle.dump(data, f)

    logging.info("Total passages processed %d. Written to %s", len(data), file)


if __name__ == "__main__":
    main()