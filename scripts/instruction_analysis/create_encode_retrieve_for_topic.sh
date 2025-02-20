#!/bin/bash

### Generate fine-grained instructions for bomb samples
conda activate ragsafetyv2

topic="bomb"
tmp="Fine"

CUDA_VISIBLE_DEVICES=2 \
python -m instruction_analysis.create_finegrained_instructions \
    --template_name $tmp \
    --topic $topic \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --max_new_tokens 1024 \
    --data_file_path results/demonstration/demonstration_t-Wiki_m-Mistral-7B-Instruct-v0.2_d-advbench_s-0/malicious_chunked.jsonl


### Retrieve with and without fine-grained instructions
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

#### with instruction keys
mal_template="Wiki"
corpus="retrieval_corpus.jsonl"
prefix="malicious_chunked"
corpus_dir=corpus_embedding
topic="bomb"
instruction_template="Fine"
model="samaya-ai/promptriever-llama3.1-8b-instruct-v1" 
dir="results/demonstration/demonstration_t-${mal_template}_m-Mistral-7B-Instruct-v0.2_d-advbench_s-0/${corpus_dir}/${ctx_encoder_model[${model#*/}]}"
if [[ ! -e ${dir}/instruction/${instruction_template}_instruction ]]; then
    mkdir -p ${dir}/instruction/${instruction_template}_instruction
fi
CUDA_VISIBLE_DEVICES=0 \
python retrieve.py \
    --encoder_class_name ${class_name[${model#*/}]} \
    --encoder_name ${model} \
    --encoder_type q \
    --batch_size 64 \
    --out_file ${dir}/instruction/${instruction_template}_instruction/retrieve_${topic}_instruction \
    --qa_dataset results/instruction/instruction_t-${instruction_template}_m-Mistral-7B-Instruct-v0.2_d-malicious_chunked_topic-${topic}_s-0_ndocs-1/instructions.jsonl \
    --instruction_key "specific_instruction" \
    --encoded_ctx_files "${dir}/embed_*_vec" \
    --eval_unsafety 1 \
    --relevant_idx results/instruction/instruction_t-${instruction_template}_m-Mistral-7B-Instruct-v0.2_d-malicious_chunked_topic-${topic}_s-0_ndocs-1/instructions_to_chunked_map.json \
    --malicious_prefix ${prefix} \
    --output_dtype ${dtype[${ctx_encoder_model[${model#*/}]}]} \
    --ctx_sources data/${corpus} >> ${dir}/instruction/${instruction_template}_instruction/retrieve_${topic}_instruction.log 2>&1


#### without instruction keys
model="samaya-ai/promptriever-llama3.1-8b-instruct-v1"
dir="results/demonstration/demonstration_t-${mal_template}_m-Mistral-7B-Instruct-v0.2_d-advbench_s-0/${corpus_dir}/${ctx_encoder_model[${model#*/}]}"
if [[ ! -e ${dir}/instruction/${instruction_template}_instruction ]]; then
    mkdir -p ${dir}/instruction/${instruction_template}_instruction
fi
CUDA_VISIBLE_DEVICES=0 \
python retrieve.py \
    --encoder_class_name ${class_name[${model#*/}]} \
    --encoder_name ${model} \
    --encoder_type q \
    --batch_size 64 \
    --out_file ${dir}/instruction/${instruction_template}_instruction/retrieve_${topic} \
    --qa_dataset results/instruction/instruction_t-${instruction_template}_m-Mistral-7B-Instruct-v0.2_d-malicious_chunked_topic-${topic}_s-0_ndocs-1/instructions.jsonl \
    --encoded_ctx_files "${dir}/embed_*_vec" \
    --eval_unsafety 1 \
    --relevant_idx results/instruction/instruction_t-${instruction_template}_m-Mistral-7B-Instruct-v0.2_d-malicious_chunked_topic-${topic}_s-0_ndocs-1/instructions_to_chunked_map.json \
    --malicious_prefix ${prefix} \
    --output_dtype ${dtype[${ctx_encoder_model[${model#*/}]}]} \
    --ctx_sources data/${corpus} >> ${dir}/instruction/${instruction_template}_instruction/retrieve_${topic}.log 2>&1
