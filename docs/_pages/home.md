---
permalink: /
layout: splash
header:
    overlay_color: rgb(237, 27, 47)
    actions:
        - label: "Paper"
          url: https://arxiv.org
          icon: "fas fa-book"
        - label: "Code"
          url: "https://github.com/McGill-NLP/malicious-ir"
          icon: "fab fa-github"
        - label: "Dataset"
          url: "https://huggingface.co/datasets/McGill-NLP/AdvBench-IR"
          icon: "fas fa-smile-beam"

title: "Exploiting Instruction-Following Retrievers for Malicious Information Retrieval"
excerpt: Parishad BehnamGhader, Nicholas Meade, Siva Reddy
---

This work studies the ability of retrievers to fulfill malicious queries, both when used *directly* and when used in a *retrieval augmented generation*-based setup. We design **AdvBench-IR** dataset for our experiments. We find that given malicious requests, most retrievers can (for >50% of queries) select relevant harmful passages.
We further uncover an emerging risk with instruction-following retrievers, where highly relevant harmful information can be surfaced through fine-grained instructions.
Finally, we show that even safety-aligned LLMs, such as *Llama3*, can produce harmful responses when provided with harmful retrieved passages in-context.

<p align="center">
  <img src="https://github.com/user-attachments/assets/41b7e800-3e20-4e79-9a8d-a7de471ef8f1" width="100%" alt="Malicious-IR_figure1"/>
</p>
