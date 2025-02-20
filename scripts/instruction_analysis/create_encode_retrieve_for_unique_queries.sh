#!/bin/bash

ndocs=10

######################################################################################################

conda activate ragsafetyv2
# Create 10x documents for unique Advbench queries
tmp="Wiki"
CUDA_VISIBLE_DEVICES=0 \
python -m instruction_analysis.create_malicious_demonstrations \
    --template_name $tmp \
    --extract_title \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --max_new_tokens 512 \
    --num $ndocs \
    --data_file_path data/diverse.jsonl
######################################################################################################

# Create instructions for all queries * 10 docs in the unique advbench queries
source /usr/local/pkgs/anaconda/bin/activate ragsafetyv2
tmp="Fine"
CUDA_VISIBLE_DEVICES=0 \
python -m instruction_analysis.create_finegrained_instructions \
    --name "instruction_demonstration" \
    --template_name $tmp \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --max_new_tokens 1024 \
    --num $ndocs \
    --data_file_path results/instruction_demonstration/instruction_demonstration_t-Wiki_m-Mistral-7B-Instruct-v0.2_d-diverse_s-0_ndocs-${ndocs}/malicious_for_unique_instructions.jsonl
######################################################################################################

# Prepare new dataset for encoding
source /usr/local/pkgs/anaconda/bin/activate ragsafetyv2
python prepare_corpora_for_embedding.py \
    --corpus_to_jsonl_paths results/instruction_demonstration/instruction_demonstration_t-Wiki_m-Mistral-7B-Instruct-v0.2_d-diverse_s-0_ndocs-${ndocs}/malicious_for_unique_instructions.jsonl \
    --jsonl_path data/retrieval_corpus_instructions_ndocs-${ndocs}.jsonl
######################################################################################################

# Encode the new documents
source /usr/local/pkgs/anaconda/bin/activate ragsafety
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
    ["LLM2Vec-Meta-Llama-31-8B-Instruct-mntp"]="32"
    ["promptriever-llama3.1-8b-instruct-v1"]="32"
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
corpus=retrieval_corpus_with_instructions_ndocs-${ndocs}
data_addr="data/retrieval_corpus_instructions_ndocs-${ndocs}.jsonl"
model="samaya-ai/promptriever-llama3.1-8b-instruct-v1"
dir="results/instruction_demonstration/instruction_demonstration_t-${mal_template}_m-Mistral-7B-Instruct-v0.2_d-diverse_s-0_ndocs-${ndocs}/${corpus}/${model#*/}"

if [[ ! -e $dir ]]; then
    mkdir -p $dir
fi

CUDA_VISIBLE_DEVICES=0 \
python generate_embeddings.py \
    --encoder_class_name ${class_name[${model#*/}]} \
    --encoder_name ${model} \
    --encoder_type ctx \
    --ctx_src $data_addr \
    --shard_id 0 \
    --num_shards 1 \
    --batch_size ${batchsize[${model#*/}]} \
    --output_dtype ${dtype[${model#*/}]} \
    --out_file ${dir}/embed_instr >> ${dir}/embed_instr.log 2>&1 
######################################################################################################

# Copy the encoded corpus to the new malicious data directory 
mal_template="Wiki"
model="samaya-ai/promptriever-llama3.1-8b-instruct-v1"
prev_corpus=corpus_embedding
corpus=retrieval_corpus_with_instructions_ndocs-${ndocs}
dir="results/instruction_demonstration/instruction_demonstration_t-${mal_template}_m-Mistral-7B-Instruct-v0.2_d-diverse_s-0_ndocs-${ndocs}/${corpus}/${model#*/}"
cp results/demonstration/demonstration_t-${mal_template}_m-Mistral-7B-Instruct-v0.2_d-advbench_s-0/${prev_corpus}/${model#*/}/embed_*_vec ${dir}/
######################################################################################################

# Create a new corpus with previous corpus + new corpus
cat data/retrieval_corpus.jsonl data/retrieval_corpus_instructions_ndocs-${ndocs}.jsonl > data/retrieval_corpus_with_instructions_ndocs-${ndocs}.jsonl
######################################################################################################

# Run retrieval on the new corpus
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
# with instruction keys
mal_template="Wiki"
corpus=retrieval_corpus_with_instructions_ndocs-${ndocs}
prefix="malicious_for_unique_instructions"
instruction_template="Fine"
model="samaya-ai/promptriever-llama3.1-8b-instruct-v1"
dir="results/instruction_demonstration/instruction_demonstration_t-${mal_template}_m-Mistral-7B-Instruct-v0.2_d-diverse_s-0_ndocs-${ndocs}/${corpus}/${model#*/}"
if [[ ! -e ${dir}/${instruction_template}_instruction ]]; then
    mkdir -p ${dir}/${instruction_template}_instruction
fi
CUDA_VISIBLE_DEVICES=0 \
python retrieve.py \
    --encoder_class_name ${class_name[${model#*/}]} \
    --encoder_name ${model} \
    --encoder_type q \
    --batch_size 64 \
    --out_file ${dir}/${instruction_template}_instruction/retrieve_instruction \
    --qa_dataset results/instruction_demonstration/instruction_demonstration_t-${instruction_template}_m-Mistral-7B-Instruct-v0.2_d-malicious_for_unique_instructions_topic-all_s-0_ndocs-${ndocs}/instructions.jsonl \
    --instruction_key "specific_instruction" \
    --encoded_ctx_files "${dir}/embed_*_vec" \
    --eval_unsafety 1 \
    --relevant_idx results/instruction_demonstration/instruction_demonstration_t-${instruction_template}_m-Mistral-7B-Instruct-v0.2_d-malicious_for_unique_instructions_topic-all_s-0_ndocs-${ndocs}/instructions_to_chunked_map.json \
    --malicious_prefix ${prefix} \
    --output_dtype ${dtype[${ctx_encoder_model[${model#*/}]}]} \
    --ctx_sources data/${corpus}.jsonl >> ${dir}/${instruction_template}_instruction/retrieve_instruction.log 2>&1

# without instruction keys
CUDA_VISIBLE_DEVICES=0 \
python retrieve.py \
    --encoder_class_name ${class_name[${model#*/}]} \
    --encoder_name ${model} \
    --encoder_type q \
    --batch_size 64 \
    --out_file ${dir}/${instruction_template}_instruction/retrieve \
    --qa_dataset results/instruction_demonstration/instruction_demonstration_t-${instruction_template}_m-Mistral-7B-Instruct-v0.2_d-malicious_for_unique_instructions_topic-all_s-0_ndocs-${ndocs}/instructions.jsonl \
    --encoded_ctx_files "${dir}/embed_*_vec" \
    --eval_unsafety 1 \
    --relevant_idx results/instruction_demonstration/instruction_demonstration_t-${instruction_template}_m-Mistral-7B-Instruct-v0.2_d-malicious_for_unique_instructions_topic-all_s-0_ndocs-${ndocs}/instructions_to_chunked_map.json \
    --malicious_prefix ${prefix} \
    --output_dtype ${dtype[${ctx_encoder_model[${model#*/}]}]} \
    --ctx_sources data/${corpus}.jsonl >> ${dir}/${instruction_template}_instruction/retrieve.log 2>&1

######################################################################################################
