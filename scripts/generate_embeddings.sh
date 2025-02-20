#!/bin/bash

source /usr/local/pkgs/anaconda/bin/activate ragsafety

python prepare_corpora_for_embedding.py \
    --corpus_to_jsonl_paths data/advbench-ir_chunked.jsonl data/psgs_w100.tsv \
    --jsonl_path data/retrieval_corpus.jsonl

declare -A class_name=(
    ["dpr-ctx_encoder-single-nq-base"]="dpr"
    ["dpr-ctx_encoder-multiset-base"]="dpr"
    ["contriever-msmarco"]="contriever"
    ["NV-Embed-v2"]="nv"
    ["bge-en-icl"]="bge"
    ["LLM2Vec-Meta-Llama-31-8B-Instruct-mntp"]="llm2vec"
    ["promptriever-llama3.1-8b-instruct-v1"]="promptriever"
)

declare -A batchsize=(
    ["dpr-ctx_encoder-single-nq-base"]="2048"
    ["dpr-ctx_encoder-multiset-base"]="2048"
    ["contriever-msmarco"]="2048"
    ["NV-Embed-v2"]="32"
    ["bge-en-icl"]="32"
    ["LLM2Vec-Meta-Llama-31-8B-Instruct-mntp"]="48"
    ["promptriever-llama3.1-8b-instruct-v1"]="48"
)

declare -A dtype=(
    ["dpr-ctx_encoder-single-nq-base"]="float32"
    ["dpr-ctx_encoder-multiset-base"]="float32"
    ["contriever-msmarco"]="float32"
    ["NV-Embed-v2"]="float32"
    ["bge-en-icl"]="float32"
    ["LLM2Vec-Meta-Llama-31-8B-Instruct-mntp"]="float32"
    ["promptriever-llama3.1-8b-instruct-v1"]="float32"
)

mal_template="Wiki"
corpus="retrieval_corpus.jsonl"
model="samaya-ai/promptriever-llama3.1-8b-instruct-v1" # "nvidia/NV-Embed-v2" etc.

dir="results/demonstration/demonstration_t-${mal_template}_m-Mistral-7B-Instruct-v0.2_d-advbench_s-0/corpus_embedding/${model#*/}"

if [[ ! -e $dir ]]; then
    mkdir -p $dir
fi

CUDA_VISIBLE_DEVICES=0 \
python generate_embeddings.py \
    --encoder_class_name ${class_name[${model#*/}]} \
    --encoder_name ${model} \
    --encoder_type ctx \
    --ctx_src data/${corpus} \
    --shard_id 0 \
    --num_shards 1 \
    --batch_size ${batchsize[${model#*/}]} \
    --output_dtype ${dtype[${model#*/}]} \
    --out_file ${dir}/embed >> ${dir}/embed_0.log 2>&1 

