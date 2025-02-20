#!/usr/bin/env python3

import argparse
import logging
import os
import json
import random
from typing import Any, Dict

from datasets import Dataset
from transformers import AutoTokenizer, set_seed
from vllm import LLM, SamplingParams

from malicious_ir.template import (
    load_instruction_generation_template,
    load_chat_template,
    load_system_message,
)
from malicious_ir.utils import generate_experiment_id, get_file_name, log_args

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Generates harmful examples.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data is stored.",
)
parser.add_argument(
    "--name",
    action="store",
    type=str,
    default=None,
    help="If used, overrides the default name of the experiment.",
)
parser.add_argument(
    "--topic",
    action="store",
    type=str,
    default="all",
    help="If specified, to filter out specific samples from the dataset.",
)
parser.add_argument(
    "--template_name",
    action="store",
    type=str,
    required=True,
    help="Name of the template.",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="HuggingFaceH4/tiny-random-LlamaForCausalLM",
    help="The model to generate harmful examples with.",
)
parser.add_argument(
    "--data_file_path",
    action="store",
    type=str,
    required=True,
    help="Path to the JSONL file containing the harmful instructions to generate "
    "responses to.",
)
parser.add_argument(
    "--num",
    action="store",
    type=int,
    default=1,
    help="Number of generations per query",
)
parser.add_argument(
    "--max_new_tokens",
    action="store",
    type=int,
    default=512,
    help="The maximum number of tokens to generate in the responses.",
)
parser.add_argument(
    "--temperature",
    action="store",
    type=float,
    default=1.0,
    help="Temperature for sampling.",
)
parser.add_argument(
    "--top_p",
    action="store",
    type=float,
    default=0.95,
    help="Parameter for Nucleus Sampling.",
)
parser.add_argument(
    "--seed",
    action="store",
    type=int,
    default=0,
    help="Seed for RNG.",
)
parser.add_argument(
    "--min_chunk_len",
    action="store",
    type=int,
    default=40,
    help="Minimum length of the valid chunks.",
)

def chunk_dataset(dataset, l=100):
    lines = []
    mapping = {}
    for data in dataset:
        tokens = data["response"][0].split()
        for start in range(0, len(tokens), l):
            new_data = {}
            new_data["id_"] = len(lines)
            new_data["response"] = [" ".join(tokens[start: start + l])]
            if "title" in data:
                new_data["title"] = data["title"]

            for k, v in data.items():
                new_data["original_data_" + k] = v
            lines.append(new_data)

            if new_data["original_data_id_"] not in mapping:
                mapping[new_data["original_data_id_"]] = []
            mapping[new_data["original_data_id_"]].append(new_data["id_"])

    return lines, mapping
    
def extract_instr(response):
    def _extract(beg="<instruction>", end="</instruction>"):
        if beg in response:
            resp = response.index(beg)
            if end in response[resp:]:
                beg_idx = resp + len(beg)
                end_idx = resp + response[resp:].index(end)
                return response[beg_idx:end_idx]
        
        return None

    st_en = [("<instruction>", "</instruction>"), ("Instruction:", "\n"), ("<Instruction>", "\n")]
    for st, en in st_en:
        instr = _extract(beg=st, end=en)
        if instr is not None:
            return instr
    return response


if __name__ == "__main__":
    args = parser.parse_args()

    set_seed(args.seed)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    name = args.name or "instruction"
    dataset_name = get_file_name(args.data_file_path)
    template_name = args.template_name

    experiment_id = generate_experiment_id(
        name=name,
        template_name=template_name,
        topic=args.topic,
        model_name_or_path=args.model_name_or_path,
        dataset_name=dataset_name,
        seed=args.seed,
        num_docs=args.num,
    )

    logging.info("Generating harmful examples:")
    log_args(args)

    dataset_dict = {}
    with open(args.data_file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            id_key = "original_data_id_" if "original_data_id_" in data else "id_"
            if data[id_key] not in dataset_dict:
                dataset_dict[data[id_key]] = []
            dataset_dict[data[id_key]].append(data)

    new_dataset = []
    for k, v in dataset_dict.items():
        new_query_datas = []
        for data in v:
            if len(data["response"][0].split()) > args.min_chunk_len:
                new_query_datas.append(data)
        new_dataset.append(random.choice(new_query_datas))

    dataset = Dataset.from_list(new_dataset)
    if "original_data_instruction" in dataset.column_names:
        dataset = dataset.rename_column("original_data_instruction", "instruction")
    if args.topic != "all":
        dataset = dataset.filter(lambda example: args.topic in example['instruction'])

    logging.info("Dataset length after filtering topic %s: %d", args.topic, len(dataset))
    system_message = load_system_message(args.model_name_or_path)
    chat_template = load_chat_template(args.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.chat_template = chat_template

    def _format_prompt(example: Dict[str, Any]) -> Dict[str, Any]:
        user_content = load_instruction_generation_template(template_name).format(
            passage=example["response"], query=example["instruction"]
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]

        example["prompt"] = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        return example

    logging.info("Template: %s", load_instruction_generation_template(template_name))
    dataset = dataset.map(_format_prompt, batched=False)
    sampling_params = SamplingParams(
        top_p=args.top_p, temperature=args.temperature, max_tokens=args.max_new_tokens
    )

    engine = LLM(args.model_name_or_path, trust_remote_code=True)

    request_outputs = engine.generate(dataset["prompt"], sampling_params)
    responses = [output.outputs[0].text for output in request_outputs]
    responses = [response.strip() for response in responses]     

    new_responses = []
    for response in responses:
        new_resp = extract_instr(response)
        new_responses.append(new_resp)

    responses = new_responses    
    dataset = dataset.add_column("specific_instruction", responses)

    os.makedirs(f"{args.persistent_dir}/results/{name}/{experiment_id}", exist_ok=True)
    dataset.to_json(f"{args.persistent_dir}/results/{name}/{experiment_id}/instructions.jsonl")

    gold = {str(d["id_"]): [d["id_"]] for d in dataset}
    with open(f"{args.persistent_dir}/results/{name}/{experiment_id}/instructions_to_chunked_map.json", "w") as f:
        json.dump(gold, f)
