import argparse
import logging
import os
import re
from typing import Any, Dict, Optional


def generate_experiment_id(
    name: str,
    template_name: Optional[str] = None,
    model_name_or_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    topic: Optional[str] = None,
    seed: Optional[int] = None,
    num_docs: Optional[int] = None,
    num_gold_docs: Optional[int] = None,
) -> str:
    """Generates an experiment ID from the given parameters.

    Args:
        name: The name of the experiment.
        model_name_or_path: The name or path to the model used for the experiment.
        dataset_name: The name of the dataset used for the experiment.
        seed: The seed used for the experiment.

    Returns:
        The experiment ID.
    """
    experiment_id = name
    if template_name is not None:
        experiment_id += f"_t-{template_name}"
    if model_name_or_path is not None:
        experiment_id += f"_m-{get_short_model_name(model_name_or_path)}"
    if dataset_name is not None:
        experiment_id += f"_d-{dataset_name}"
    if topic is not None:
        experiment_id += f"_topic-{topic}"
    if seed is not None:
        experiment_id += f"_s-{seed}"
    if num_docs is not None:
        experiment_id += f"_ndocs-{num_docs}"
    if num_gold_docs is not None:
        experiment_id += f"_ngolds-{num_gold_docs}"

    return experiment_id


def parse_experiment_id(experiment_id: str) -> Dict[str, Any]:
    """Dynamically parses and experiment ID.

    Args:
        experiment_id: The experiment ID to parse.

    Returns:
        A dictionary containing the experiment parameters parsed from the experiment ID.
    """

    parameter_to_regex = {
        "template_name": "([A-Za-z0-9-_]+)",
        "model_name_or_path": "([A-Za-z0-9-._]+)",
        "dataset_name": "([A-Za-z0-9-_]+)",
        "seed": "([0-9]+)",
        "num_docs": "([0-9]+)",
        "num_gold_docs": "([0-9]+)",
        "topic": "([A-Za-z0-9-_]+)",
    }

    parameter_to_code = {
        "template_name": "t",
        "model_name_or_path": "m",
        "dataset_name": "d",
        "seed": "s",
        "num_docs": "ndocs",
        "num_gold_docs": "ngolds",
        "topic": "topic",
    }

    parameter_to_type = {
        "template_name": str,
        "model_name_or_path": str,
        "dataset_name": str,
        "seed": int,
        "num_docs": int,
        "num_gold_docs": int,
        "topic": str,
    }

    # Check which parameters are in the experiment ID. This search is currently brittle
    # and can potentially return false positives.
    parameters_to_parse = []
    for parameter, code in parameter_to_code.items():
        if re.search(f"_{code}-", experiment_id):
            parameters_to_parse.append(parameter)

    # Build the regex. The experiment name is always included.
    regex = "([A-Za-z0-9-_]+)"
    for parameter in parameters_to_parse:
        regex += f"_{parameter_to_code[parameter]}-{parameter_to_regex[parameter]}"

    parts = re.match(regex, experiment_id).groups()

    # Cast the parameters to the correct type.
    results = {"name": parts[0]}
    for parameter, part in zip(parameters_to_parse, parts[1:]):
        results[parameter] = parameter_to_type[parameter](part)

    return results


def get_file_name(file_path: str) -> str:
    """Gets the file name from a file path without the extension.

    Args:
        file_path: The path to the file.

    Returns:
        The file name without the extension.
    """
    return os.path.splitext(os.path.basename(file_path))[0]


def get_short_model_name(model_name_or_path: str) -> str:
    """Gets a short model name from the model name or path.

    Args:
        model_name_or_path: The model name or path.

    Returns:
        The short model name.
    """
    return model_name_or_path.rstrip("/").split("/")[-1]


def log_args(args: argparse.Namespace) -> None:
    """Log the commandline arguments."""
    for arg in vars(args):
        logging.info(f" - {arg}: {getattr(args, arg)}")


def str2bool(s):
    s = s.lower()
    if s == "true" or s == "1" or s == "t":
        return True
    if s == "false" or s == "0" or s == "f":
        return False
    raise ValueError(f"{s} is not a valid format.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prints experiment ID for given arguments. Can be ran without "
        "installing the package."
    )
    parser.add_argument(
        "experiment_path",
        action="store",
        type=str,
        help="Path to the experiment script to generate the experiment ID for.",
    )
    parser.add_argument(
        "--model_name_or_path",
        action="store",
        required=True,
        type=str,
        help="Name or path to the model used in the experiment.",
    )
    parser.add_argument(
        "--data_file_path",
        action="store",
        default=None,
        type=str,
        help="Path to the JSONL data file used in the experiment.",
    )
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        help="Seed used for RNG.",
    )

    args, _ = parser.parse_known_args()

    name = get_file_name(args.experiment_path)

    if args.data_file_path is not None:
        dataset_name = get_file_name(args.data_file_path)
    else:
        dataset_name = None

    experiment_id = generate_experiment_id(
        name=name,
        model_name_or_path=args.model_name_or_path,
        dataset_name=dataset_name,
        seed=args.seed,
    )
    print(experiment_id)
