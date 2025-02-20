#!/bin/bash

conda activate ragsafetyv2

python prepare_advbench_dataset.py \
    --data_file_path data/advbench.csv

CUDA_VISIBLE_DEVICES=0 \
python generate_malicious_passages.py \
    --template_name Wiki \
    --extract_title \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --max_new_tokens 1024 \
    --data_file_path data/advbench.jsonl >> results/demonstration/demonstration_t-Wiki_m-Mistral-7B-Instruct-v0.2_d-advbench_s-0/malicious.log 2>&1
