import argparse
import logging
import os
import csv
import json
import glob
from tqdm import tqdm
from pathlib import Path
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
parser = argparse.ArgumentParser(description="Prepare corpus to be loaded by Pyserini.")
parser.add_argument(
    "--corpus_to_jsonl_paths",
    action="store",
    nargs="+",
    required=True,
    help="Paths to files that are to be converted to jsonl.",
)
parser.add_argument(
    "--jsonl_path",
    action="store",
    type=str,
    required=True,
    help="Output jsonl path to which the data will be stored.",
)
class SubCorpus:
    def __init__(self, path, normalize=True) -> None:
        self.path = path
        self.normalize = normalize
        self.base_name = Path(path).stem
        if path.endswith("jsonl"):
            self.corpus_type = "json" 
        elif path.endswith("tsv"):
            self.corpus_type = "csv" 
        else:
            raise NotImplementedError(f"The format of file {path} is not implemented yet.")
    def reader(self):
        if self.corpus_type == "csv":
            reader = csv.reader(open(self.path), delimiter="\t")
        if self.corpus_type == "json":
            reader = open(self.path)
        
        for line in reader:
            if self.corpus_type == "csv":
                if len(line) == 3:
                    id, content, title = line[0], line[1], line[2]
                elif len(line) == 2:
                    id, content, title = line[0], line[1], None
                else:
                    raise NotImplementedError(f"The data loading is not implemented for {len(line)} elements in a row.")
                if id == "id":
                    continue
                content.strip('"')
                if self.normalize:
                    content = normalize_passage(content)
                if title is not None:
                    content = title + " # " + content
                yield {
                    "id": self.base_name + "_" + str(id),
                    "contents": content
                }
            if self.corpus_type == "json":
                line = json.loads(line)
                if "id_" in line:
                    id = line["id_"]
                elif "id" in line:
                    id = line["id"]
                else:
                    raise ValueError(f"Could not find id in data keys {list(line.keys())}.")
                if "response" in line:
                    content = line["response"]
                    if isinstance(content, list):
                        content = content[0]
                    if "title" in line and line["title"] is not None:
                        content = line["title"] + " # " + content
                else:
                    raise ValueError(f"Could not find content in data keys {list(line.keys())}.")
                
                content.strip('"')
                if self.normalize:
                    content = normalize_passage(content)
                yield {
                    "id": self.base_name + "_" + str(id),
                    "contents": content
                }

def normalize_passage(ctx_text: str):
    ctx_text = ctx_text.replace("\n", " ").replace("’", "'")
    if ctx_text.startswith('"'):
        ctx_text = ctx_text[1:]
    if ctx_text.endswith('"'):
        ctx_text = ctx_text[:-1]
    return ctx_text

def main(args):
    with open(args.jsonl_path, 'w') as ofile:
        logging.info("Reading files %s", args.corpus_to_jsonl_paths)
        for file in args.corpus_to_jsonl_paths:
            logging.info("Reading file %s", file)
            corpus = SubCorpus(file)
            for en, row in tqdm(enumerate(corpus.reader())):
                ofile.write(json.dumps(row) + "\n")
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)