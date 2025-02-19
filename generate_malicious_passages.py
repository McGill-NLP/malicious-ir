#!/usr/bin/env python3

import argparse
import logging
import os
import json
from typing import Any, Dict

from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from vllm import LLM, SamplingParams

from retriever_safety.template import (
    load_harmful_template,
    load_chat_template,
    load_system_message,
)
from retriever_safety.utils import generate_experiment_id, get_file_name, log_args

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Generates harmful examples.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=".",
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
    "--chunk_len",
    action="store",
    type=int,
    default=100,
    help="Length of the chunks.",
)
parser.add_argument(
    "--extract_title",
    action="store_true",
    default=False,
    help="If the title should be extracted from the response.",
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
    
def extract_title(response):
    def _extract(beg="<title>", end="</title>"):
        if beg in response:
            resp = response.index(beg)
            if end in response[resp:]:
                beg_idx = resp + len(beg)
                end_idx = resp + response[resp:].index(end)
                return response[beg_idx:end_idx], (response[:resp] + "\n" + response[end_idx + len(end):]).strip()
        
        return None, None

    st_en = [("<title>", "</title>"), ("Title:", "\n"), ("<title>", "\n")]
    for st, en in st_en:
        title, other_response = _extract(beg=st, end=en)
        if title is not None:
            return title, other_response
    return None, response


if __name__ == "__main__":
    args = parser.parse_args()

    set_seed(args.seed)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    name = args.name or "demonstration"
    dataset_name = get_file_name(args.data_file_path)
    template_name = args.template_name

    experiment_id = generate_experiment_id(
        name=name,
        template_name=template_name,
        model_name_or_path=args.model_name_or_path,
        dataset_name=dataset_name,
        seed=args.seed,
    )

    logging.info("Generating harmful examples:")
    log_args(args)

    dataset = load_dataset("json", split="train", data_files=args.data_file_path)

    system_message = load_system_message(args.model_name_or_path)
    chat_template = load_chat_template(args.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.chat_template = chat_template

    def _format_prompt(example: Dict[str, Any]) -> Dict[str, Any]:
        user_content = load_harmful_template(template_name).format(
            instruction=example["instruction"], response=example["response"]
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

    logging.info("Template: %s", load_harmful_template(template_name))
    dataset = dataset.map(_format_prompt, batched=False)
    sampling_params = SamplingParams(
        top_p=args.top_p, temperature=args.temperature, max_tokens=args.max_new_tokens
    )

    engine = LLM(args.model_name_or_path, trust_remote_code=True)

    request_outputs = engine.generate(dataset["prompt"], sampling_params)
    responses = [output.outputs[0].text for output in request_outputs]
    responses = [[response.strip()] for response in responses]

    if args.extract_title:
        new_responses = []
        titles = []
        for response in responses:
            title, other_response = extract_title(response[0])
            new_responses.append([other_response])
            titles.append(title)

        dataset = dataset.add_column("title", titles)
        responses = new_responses     

    dataset = dataset.remove_columns(["response"])
    dataset = dataset.add_column("response", responses)

    os.makedirs(f"{args.persistent_dir}/results/{name}/{experiment_id}", exist_ok=True)
    dataset.to_json(f"{args.persistent_dir}/results/{name}/{experiment_id}/malicious.jsonl")

    lines, mapping = chunk_dataset(dataset, l=args.chunk_len)
    with open(f"{args.persistent_dir}/results/{name}/{experiment_id}/malicious_chunked.jsonl", "w") as file:
        file.writelines([json.dumps(line) + "\n" for line in lines])
    with open(f"{args.persistent_dir}/results/{name}/{experiment_id}/malicious_to_chunked_map.json", "w") as file:
        file.write(json.dumps(mapping))
