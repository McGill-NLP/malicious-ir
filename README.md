# *Exploiting Instruction-Following Retrievers for Malicious Information Retrieval*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/McGill-NLP/malicious-ir/blob/main/LICENSE)

This work studies the ability of retrievers to fulfill malicious queries, both when used *directly* and when used in a *retrieval augmented generation*-based setup. We find that given malicious requests, most retrievers can (for >50% of queries) select relevant harmful passages.
We further uncover an emerging risk with instruction-following retrievers, where highly relevant harmful information can be surfaced through fine-grained instructions.
Finally, we show that even safety-aligned LLMs, such as *Llama3*, can produce harmful responses when provided with harmful retrieved passages in-context.


This repository ...
## Installation
```bash
conda env create --name ret --file=retrieval_environments.yml

conda env create --name llm --file=vllm_environments.yml
```

## Getting Started
