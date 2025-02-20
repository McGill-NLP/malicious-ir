#!/bin/bash

source /usr/local/pkgs/anaconda/bin/activate ragsafetyv2

num_docs=10
retreiver="NV-Embed-v2"
mal_template="Wiki"
name="advbench_qa"
qa_template="QA"
corpus="corpus_embedding"
dir="results/demonstration/demonstration_t-${mal_template}_m-Mistral-7B-Instruct-v0.2_d-advbench_s-0/${corpus}/${retreiver}"
if [[ ! -e ${dir}/${name} ]]; then
    mkdir -p ${dir}/${name}
fi

for model in "meta-llama/Meta-Llama-3-8B-Instruct"  # "google/gemma-2-9b-it" "mistralai/Mistral-7B-Instruct-v0.2"
do
    for num_docs in $(seq 0 ${num_docs})
    do
        CUDA_VISIBLE_DEVICES=0 \
        python qa_with_rag.py \
            --persistent_dir ${dir} \
            --name ${name} \
            --model_name_or_path ${model} \
            --max_new_tokens 512 \
            --template_name ${qa_template} \
            --data_file_path ${dir}/retrieve_advbench \
            --malicious_prefix malicious_chunked_w_title_less_hypo \
            --num_docs ${num_docs} >> ${dir}/${name}/m-${model#*/}_s-0.log 2>&1
    done
done

num_docs=10
retreiver="NV-Embed-v2"
mal_template="Wiki"
name="advbench_qa"
qa_template="QA"
corpus="corpus_embedding"
dir="results/demonstration/demonstration_t-${mal_template}_m-Mistral-7B-Instruct-v0.2_d-advbench_s-0/${corpus}/${retreiver}"
if [[ ! -e ${dir}/${name}/eval ]]; then
    mkdir -p ${dir}/${name}/eval
fi
for model in "meta-llama/Meta-Llama-3-8B-Instruct"  # "google/gemma-2-9b-it" "mistralai/Mistral-7B-Instruct-v0.2"
do
    for num_gold_docs in $(seq 0 ${num_docs})
    do
        CUDA_VISIBLE_DEVICES=0 \
        python evaluate_qa_response_safety.py \
            --model_name_or_path meta-llama/Llama-Guard-3-8B \
            --response_file_path ${dir}/${name}/${name}_t-${qa_template}_m-${model#*/}_s-0_ndocs-${num_gold_docs}.jsonl \
            --eval_file_path ${dir}/${name}/eval/eval_t-${qa_template}_m-${model#*/}_s-0_ndocs-${num_gold_docs}.json \
            >> ${dir}/${name}/eval/eval_t-${qa_template}_m-${model#*/}_s-0.log 2>&1
    done
done
