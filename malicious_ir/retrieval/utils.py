import logging
import socket
import subprocess
import os
from typing import List, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer, AutoTokenizer, AutoModel, AutoConfig
from llm2vec import LLM2Vec
from malicious_ir.retrieval import faiss_indexers
from malicious_ir.retrieval.tensorizer import Tensorizer
from malicious_ir.retrieval.models import Promptriever


INDEX_MAP = {
    "flat": faiss_indexers.DenseFlatIndexer,
    "hnsw": faiss_indexers.DenseHNSWFlatIndexer,
    "hsnw_sq": faiss_indexers.DenseHNSWSQIndexer
}

MODEL_MAP = {
    "dpr": {
        "q": {
            "tokenizer": DPRQuestionEncoderTokenizer,
            "encoder": DPRQuestionEncoder
        },
        "ctx": {
            "tokenizer": DPRContextEncoderTokenizer,
            "encoder": DPRContextEncoder
        }
    },
    "contriever": {
        "q": {
            "tokenizer": AutoTokenizer,
            "encoder": AutoModel
        },
        "ctx": {
            "tokenizer": AutoTokenizer,
            "encoder": AutoModel
        }
    },
    "nv": {
        "q": {
            "tokenizer": None,
            "encoder": AutoModel
        },
        "ctx": {
            "tokenizer": None,
            "encoder": AutoModel
        }
    },
    "bge": {
        "q": {
            "tokenizer": AutoTokenizer,
            "encoder": AutoModel
        },
        "ctx": {
            "tokenizer": AutoTokenizer,
            "encoder": AutoModel
        }
    },
    "llm2vec": {
        "q": {
            "tokenizer": AutoTokenizer,
            "encoder": LLM2Vec
        },
        "ctx": {
            "tokenizer": AutoTokenizer,
            "encoder": LLM2Vec
        }
    },
    "promptriever":
    {
        "q": {
            "tokenizer": None,
            "encoder": Promptriever
        },
        "ctx": {
            "tokenizer": None,
            "encoder": Promptriever
        }
    }
}

QUERY_PROMPTS = {
    "nv": {
        "q": "Instruct: Given a question, retrieve passages that answer the question\nQuery: ",
        "q_suffix": "",
        "ctx": "",
    },
    "bge": {
        "q": "<instruct>Given a question, retrieve passages that answer the question\n<query>",
        "q_suffix" : "\n<response>",
        "ctx": "",
    },
    "llm2vec": {
        "q": "Given a question, retrieve passages that answer the question:",
        "q_suffix" : "",
        "ctx": "",
    },
    "promptriever": {
        "q": "query:  Given a question, retrieve passages that answer the question. ",
        "q_suffix" : "",
        "ctx": "passage:  ",
    }
}

def load_encoder(encoder_name, encoder_class_name, encoder_type, encoder_name_2=None):
    """
    Loads encoder and tensorizer based on name and class_name and type
    """
    assert encoder_type in ["q", "ctx"], f"Model type {encoder_type} is not implemented yet. Choices: q, ctx"

    tokenizer_class, encoder_class = MODEL_MAP[encoder_class_name][encoder_type]["tokenizer"], MODEL_MAP[encoder_class_name][encoder_type]["encoder"]
    
    max_length = 256 if encoder_type == "ctx" else 128
    tensorizer = None
    if tokenizer_class is not None:
        tensorizer = Tensorizer(tokenizer_class.from_pretrained(encoder_name), max_length=max_length)
    if encoder_class_name == "llm2vec":
        encoder = LLM2Vec.from_pretrained(
            encoder_name,
            peft_model_name_or_path=encoder_name_2,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
        )
    elif encoder_class_name == "promptriever":
        encoder = encoder_class(encoder_name, max_length=max_length, torch_dtype=torch.float16)
    else:
        configs = {}
        if encoder_class_name == "nv":
            configs["torch_dtype"] = torch.float16
        encoder = encoder_class.from_pretrained(encoder_name, trust_remote_code=True, **configs)

    return tensorizer, encoder

def generate_embedding(batch_texts, tensorizer, encoder, prefix: str = "", suffix: str = "", encoder_class_name: str = None, device: str = None):
    if encoder_class_name == "nv":
        out = encoder.encode(batch_texts, instruction=prefix)
        return F.normalize(out, p=2, dim=1)
    
    if encoder_class_name == "llm2vec":
        if prefix != "":
            batch_texts_ = [[prefix, text] for text in batch_texts]
        else:
            batch_texts_ = batch_texts
        out = encoder.encode(batch_texts_, show_progress_bar=False)
        return F.normalize(out, p=2, dim=1)

    if encoder_class_name == "promptriever":
        batch_texts_ = [f"{prefix}{text}{suffix}" for text in batch_texts]
        return encoder.encode(batch_texts_)
    
    if encoder_class_name == "bge":
        batch_texts_ = [f"{prefix}{text}{suffix}" for text in batch_texts]
        batch_texts = batch_texts_
    
    tns = tensorizer.text_to_tensor(batch_texts)
    with torch.no_grad():
        encoder_input = move_to_device(tns, device)
        out = encoder(**encoder_input)
        if encoder_class_name == "bge":
            left_padding = (encoder_input['attention_mask'][:, -1].sum() == encoder_input['attention_mask'].shape[0])
            if left_padding:
                return out.last_hidden_state[:, -1]
            else:
                sequence_lengths = encoder_input['attention_mask'].sum(dim=1) - 1
                batch_size = out.last_hidden_state.shape[0]
                return out.last_hidden_state[torch.arange(batch_size, device=out.last_hidden_state.device), sequence_lengths]
        if encoder_class_name == "dpr":
            return out.pooler_output
        if encoder_class_name == "contriever":
            token_embeddings = out[0].masked_fill(~encoder_input["attention_mask"][..., None].bool(), 0.)
            return token_embeddings.sum(dim=1) / encoder_input["attention_mask"].sum(dim=1)[..., None]
    
def instantiate_index(indexer):
    index_class = INDEX_MAP[indexer]
    return index_class()

def setup_cfg_gpu(cfg):
    """
    Setup params for CUDA, GPU & distributed training
    """
    logging.info("CFG's local_rank=%s", cfg.local_rank)
    ws = os.environ.get("WORLD_SIZE")
    cfg.distributed_world_size = int(ws) if ws else 1
    logging.info("Env WORLD_SIZE=%s", ws)

    if cfg.distributed_port and cfg.distributed_port > 0:
        logging.info("distributed_port is specified, trying to init distributed mode from SLURM params ...")
        init_method, local_rank, world_size, device = _infer_slurm_init(cfg)

        logging.info(
            "Inferred params from SLURM: init_method=%s | local_rank=%s | world_size=%s",
            init_method,
            local_rank,
            world_size,
        )

        cfg.local_rank = local_rank
        cfg.distributed_world_size = world_size
        cfg.n_gpu = 1

        torch.cuda.set_device(device)
        device = str(torch.device("cuda", device))

        torch.distributed.init_process_group(
            backend="nccl", init_method=init_method, world_size=world_size, rank=local_rank
        )

    elif cfg.local_rank == -1 or cfg.no_cuda:  # single-node multi-gpu (or cpu) mode
        device = str(torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu"))
        cfg.n_gpu = torch.cuda.device_count()
    else:  # distributed mode
        torch.cuda.set_device(cfg.local_rank)
        device = str(torch.device("cuda", cfg.local_rank))
        torch.distributed.init_process_group(backend="nccl")
        cfg.n_gpu = 1

    cfg.device = device

    logging.info(
        "Initialized host %s as d.rank %d on device=%s, n_gpu=%d, world size=%d",
        socket.gethostname(),
        cfg.local_rank,
        cfg.device,
        cfg.n_gpu,
        cfg.distributed_world_size,
    )
    logging.info("16-bits training: %s ", cfg.fp16)
    return cfg

def _infer_slurm_init(cfg) -> Tuple[str, int, int, int]:

    node_list = os.environ.get("SLURM_STEP_NODELIST")
    if node_list is None:
        node_list = os.environ.get("SLURM_JOB_NODELIST")
    logging.info("SLURM_JOB_NODELIST: %s", node_list)

    if node_list is None:
        raise RuntimeError("Can't find SLURM node_list from env parameters")

    local_rank = None
    world_size = None
    distributed_init_method = None
    device_id = None
    try:
        hostnames = subprocess.check_output(["scontrol", "show", "hostnames", node_list])
        distributed_init_method = "tcp://{host}:{port}".format(
            host=hostnames.split()[0].decode("utf-8"),
            port=cfg.distributed_port,
        )
        nnodes = int(os.environ.get("SLURM_NNODES"))
        logging.info("SLURM_NNODES: %s", nnodes)
        ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
        if ntasks_per_node is not None:
            ntasks_per_node = int(ntasks_per_node)
            logging.info("SLURM_NTASKS_PER_NODE: %s", ntasks_per_node)
        else:
            ntasks = int(os.environ.get("SLURM_NTASKS"))
            logging.info("SLURM_NTASKS: %s", ntasks)
            assert ntasks % nnodes == 0
            ntasks_per_node = int(ntasks / nnodes)

        if ntasks_per_node == 1:
            gpus_per_node = torch.cuda.device_count()
            node_id = int(os.environ.get("SLURM_NODEID"))
            local_rank = node_id * gpus_per_node
            world_size = nnodes * gpus_per_node
            logging.info("node_id: %s", node_id)
        else:
            world_size = ntasks_per_node * nnodes
            proc_id = os.environ.get("SLURM_PROCID")
            local_id = os.environ.get("SLURM_LOCALID")
            logging.info("SLURM_PROCID %s", proc_id)
            logging.info("SLURM_LOCALID %s", local_id)
            local_rank = int(proc_id)
            device_id = int(local_id)

    except subprocess.CalledProcessError as e:  # scontrol failed
        raise e
    except FileNotFoundError:  # Slurm is not installed
        pass
    return distributed_init_method, local_rank, world_size, device_id

def setup_for_distributed_mode(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: object,
    n_gpu: int = 1,
    local_rank: int = -1,
    fp16: bool = False,
    fp16_opt_level: str = "O1",
) -> tuple[nn.Module, torch.optim.Optimizer]:
    print(model)
    model.to(device)
    if fp16:
        try:
            import apex
            from apex import amp

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device if device else local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    return model, optimizer

def move_to_device(sample, device):
    if len(sample) == 0:
        return {}

    def _move_to_device(maybe_tensor, device):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(device)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_device(value, device) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_device(x, device) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return [_move_to_device(x, device) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_device(sample, device)
