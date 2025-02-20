# *Exploiting Instruction-Following Retrievers for Malicious Information Retrieval*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/McGill-NLP/malicious-ir/blob/main/LICENSE)

This work studies the ability of retrievers to fulfill malicious queries, both when used *directly* and when used in a *retrieval augmented generation*-based setup. We find that given malicious requests, most retrievers can (for >50% of queries) select relevant harmful passages.
We further uncover an emerging risk with instruction-following retrievers, where highly relevant harmful information can be surfaced through fine-grained instructions.
Finally, we show that even safety-aligned LLMs, such as *Llama3*, can produce harmful responses when provided with harmful retrieved passages in-context.

<p align="center">
  <img src="https://github.com/user-attachments/assets/41b7e800-3e20-4e79-9a8d-a7de471ef8f1" width="100%" alt="Malicious-IR_figure1"/>
  <!-- ![ragsafety](https://github.com/user-attachments/assets/009e0e02-62ed-408a-8243-321d4256c43b) -->
</p>

This repository ...
## Installation
```bash
conda env create --name ret --file=retrieval_environments.yml

conda env create --name llm --file=vllm_environments.yml
```

## Getting Started
