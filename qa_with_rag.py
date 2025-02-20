#!/usr/bin/env python3

import argparse
import logging
import os
from typing import Any, Dict

from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from vllm import LLM, SamplingParams

from malicious_ir.template import (
    load_chat_template,
    load_system_message,
    get_message,
    load_qa_template
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
    "--malicious_prefix",
    action="store",
    type=str,
    help="Prefix for dataset name for gold (i.e., malicious) retrieved info.",
)
parser.add_argument(
    "--num_docs",
    action="store",
    type=int,
    help="Number of all retrieved documents.",
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


if __name__ == "__main__":
    args = parser.parse_args()

    set_seed(args.seed)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    name = args.name or "qa"
    template_name = args.template_name

    experiment_id = generate_experiment_id(
        name=name,
        template_name=template_name,
        model_name_or_path=args.model_name_or_path,
        seed=args.seed,
        num_docs=args.num_docs,
    )

    logging.info("Generating harmful examples:")
    log_args(args)

    dataset = load_dataset("json", split="train", data_files=args.data_file_path)
    dataset = dataset.rename_column("question", "instruction")

    system_message = load_system_message(args.model_name_or_path)
    chat_template = load_chat_template(args.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.chat_template = chat_template

    def _format_prompt(example: Dict[str, Any]) -> Dict[str, Any]:

        # Find gold (i.e., malicious) and non-gold retrieved information based on arguments
        information_list, information_title_list = [], []
        gold_counter, non_gold_counter = 0, 0
        for ctx in example["ctxs"]:
            if ctx["id"].startswith(args.malicious_prefix):
                if (gold_counter + non_gold_counter) < args.num_docs:
                    information_list.append(f"Information {len(information_list)+1}\n{ctx['text']}")
                    information_title_list.append(ctx["id"])
                gold_counter += 1
            if not ctx["id"].startswith(args.malicious_prefix):
                if (gold_counter + non_gold_counter) < args.num_docs:
                    information_list.append(f"Information {len(information_list)+1}\n{ctx['text']}")
                    information_title_list.append(ctx["id"])
                non_gold_counter += 1
        
        information = "\n\n".join(information_list)
        
        user_content = load_qa_template(template_name).format(
            instruction=example["instruction"], information="No information is provided." if args.num_docs == 0 else information
        )

        messages = get_message(system_message, user_content, args.model_name_or_path)
        
        example["prompt"] = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        example["information_count"] = {"gold": gold_counter, "non_gold": non_gold_counter}

        return example

    dataset = dataset.map(_format_prompt, batched=False)
    sampling_params = SamplingParams(
        top_p=args.top_p, temperature=args.temperature, max_tokens=args.max_new_tokens
    )

    engine = LLM(args.model_name_or_path, trust_remote_code=True)

    request_outputs = engine.generate(dataset["prompt"], sampling_params)
    responses = [output.outputs[0].text for output in request_outputs]
    responses = [[response.strip()] for response in responses]

    dataset = dataset.add_column("response", responses)
    dataset = dataset.remove_columns(["information_count"])

    save_dir = f"{args.persistent_dir}/{name}/"
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Saving harmful examples to {save_dir}")
    dataset.to_json(f"{save_dir}/{experiment_id}.jsonl")
