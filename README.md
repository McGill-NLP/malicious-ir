# *Exploiting Instruction-Following Retrievers for Malicious Information Retrieval*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/McGill-NLP/malicious-ir/blob/main/LICENSE)

This work studies the ability of retrievers to fulfill malicious queries, both when used *directly* and when used in a *retrieval augmented generation*-based setup. We find that given malicious requests, most retrievers can (for >50% of queries) select relevant harmful passages.
We further uncover an emerging risk with instruction-following retrievers, where highly relevant harmful information can be surfaced through fine-grained instructions.
Finally, we show that even safety-aligned LLMs, such as *Llama3*, can produce harmful responses when provided with harmful retrieved passages in-context.

<p align="center">
  <img src="https://github.com/user-attachments/assets/41b7e800-3e20-4e79-9a8d-a7de471ef8f1" width="100%" alt="Malicious-IR_figure1"/>
</p>

This repository includes code and data for reproducing our results and analysis.
## Installation
We need two separate environments for recent LLMs' generations and retrieval.
```bash
conda env create --name ret --file=retrieval_environments.yml

conda env create --name llm --file=vllm_environments.yml
```

## Getting Started
Our analysis consists of XXX general steps: corpus generation, corpus embedding, and retrieval inference. We go into more details for our further analysis below.

### 1. Corpus Generation
The first step includes generating the malicious passages for [AdvBench](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv) queries (i.e., **AdvBench-IR**). You can find the generated samples in [data/malicious.jsonl](data/malicious.jsonl) or on [HuggingFace](). Checkout the [create_demonstration.sh](scripts/create_demonstration.sh) script for the detailed and exact arguments. This script will generate and chunk the malicious passages for retrieval.

```bash
python generate_malicious_passages.py \
    --template_name "PsgWiki" \
    --extract_title \
    --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
    --data_file_path <address of the AdvBench queries>
```

### 2. Corpus Embedding
For a more real-world retrieval, we combine the malicious passages with the benign [DPR's Wikipedia retrieval corpus](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py) which can be downloaded from [this link](https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz).
Subsequently, we generate embeddings for the new retrieval corpus. Note that this step takes long, and you might want to run the code in multiple GPUs for different shards of the data. Checkout the [generate_embeddings.sh](scripts/generate_embeddings.sh) script for the detailed and exact arguments, and to see how to correctly combine the malicious and benign corpora. This script will generate embeddings for the retrieval corpus shards.
```bash
python generate_embeddings.py \
    --encoder_class_name "nv" \
    --encoder_name "nvidia/NV-Embed-v2" \
    --encoder_type "ctx" \
    --ctx_src <address of the retrieval corpus> \
    --shard_id <shard_id from 0 to 7> \
    --num_shards 8 \
    --out_file <dir of the corpus embeddings>/embed
```

### 3. Retrieval Inference
This step includes running the inference for the malicious AdvBench-IR dataset. Checkout the [retrieve.sh](scripts/retrieve.sh) script for the detailed and exact arguments.

```bash
python retrieve.py \
    --encoder_class_name "nv" \
    --encoder_name "nvidia/NV-Embed-v2" \
    --encoder_type "q" \
    --out_file <address of the output> \
    --qa_dataset <address of the AdvBench-IR samples> \
    --encoded_ctx_files <dir of the corpus embeddings>/embed_*_vec \
    --eval_unsafety 1 \
    --relevant_idx <address of the correct passage mappings, i.e., one of the outputs of the step 1> \
    --malicious_prefix <id prefix of the malicious corpus> \
    --ctx_sources <address of the retrieval corpus>
```

You can run the inference on benign datasets using the same script, but without the malicious arguments (e.g., `--eval_unsafety`, `--relevant_idx`, and `--malicious_prefix`). You can download TriviaQA and NaturalQuestions test samples from [DPR's repository](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py) (`nq-test.qa.csv` and `trivia-test.qa.csv`).

## Fine-Grained Query Analysis
## Retrieval-Augmented Generation Analysis