#!/bin/bash

source /usr/local/pkgs/anaconda/bin/activate ragsafety

declare -A class_name=(
    ["dpr-question_encoder-single-nq-base"]="dpr"
    ["dpr-question_encoder-multiset-base"]="dpr"
    ["contriever-msmarco"]="contriever"
    ["NV-Embed-v2"]="nv"
    ["bge-en-icl"]="bge"
    ["LLM2Vec-Meta-Llama-31-8B-Instruct-mntp"]="llm2vec"
    ["promptriever-llama3.1-8b-instruct-v1"]="promptriever"
)
declare -A ctx_encoder_model=(
    ["dpr-question_encoder-single-nq-base"]="dpr-ctx_encoder-single-nq-base"
    ["dpr-question_encoder-multiset-base"]="dpr-ctx_encoder-multiset-base"
    ["contriever-msmarco"]="contriever-msmarco"
    ["NV-Embed-v2"]="NV-Embed-v2"
    ["bge-en-icl"]="bge-en-icl"
    ["LLM2Vec-Meta-Llama-31-8B-Instruct-mntp"]="LLM2Vec-Meta-Llama-31-8B-Instruct-mntp"
    ["promptriever-llama3.1-8b-instruct-v1"]="promptriever-llama3.1-8b-instruct-v1"
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

# Unsafety on AdvBench
mal_template="Wiki"
corpus="retrieval_corpus.jsonl"
prefix="malicious_chunked"
model="samaya-ai/promptriever-llama3.1-8b-instruct-v1"  # model="nvidia/NV-Embed-v2" etc.
dir="results/demonstration/demonstration_t-${mal_template}_m-Mistral-7B-Instruct-v0.2_d-advbench_s-0/corpus_embedding/${ctx_encoder_model[${model#*/}]}"
CUDA_VISIBLE_DEVICES=0 \
python retrieve.py \
    --encoder_class_name ${class_name[${model#*/}]} \
    --encoder_name ${model} \
    --encoder_type q \
    --batch_size 64 \
    --out_file ${dir}/retrieve_advbench \
    --qa_dataset results/demonstration/demonstration_t-${mal_template}_m-Mistral-7B-Instruct-v0.2_d-advbench_s-0/malicious.jsonl \
    --encoded_ctx_files "${dir}/embed_*_vec" \
    --eval_unsafety 1 \
    --relevant_idx results/demonstration/demonstration_t-${mal_template}_m-Mistral-7B-Instruct-v0.2_d-advbench_s-0/malicious_to_chunked_map.json \
    --malicious_prefix ${prefix} \
    --output_dtype ${dtype[${ctx_encoder_model[${model#*/}]}]} \
    --ctx_sources data/${corpus} >> ${dir}/retrieve_advbench.log 2>&1


# Performance on safe datasets
declare -A qa_file=(
    ["triviaqa"]="data/trivia-test.qa.csv"
    ["nq"]="data/nq-test.qa.csv"
)
mal_template="Wiki"
model="nvidia/NV-Embed-v2"
corpus="retrieval_corpus.jsonl"
for ds in  "nq" "triviaqa"
do
    dir="results/demonstration/demonstration_t-${mal_template}_m-Mistral-7B-Instruct-v0.2_d-advbench_s-0/corpus_embedding/${ctx_encoder_model[${model#*/}]}"
    CUDA_VISIBLE_DEVICES=0 \
    python retrieve.py \
        --encoder_class_name ${class_name[${model#*/}]} \
        --encoder_name ${model} \
        --encoder_type q \
        --batch_size 64 \
        --out_file ${dir}/retrieve_${ds} \
        --qa_dataset ${qa_file[${ds}]} \
        --encoded_ctx_files "${dir}/embed_*_vec" \
        --eval_unsafety 0 \
        --validation_workers 1 \
        --output_dtype ${dtype[${ctx_encoder_model[${model#*/}]}]} \
        --ctx_sources data/${corpus} >> ${dir}/retrieve_${ds}.log 2>&1
done
