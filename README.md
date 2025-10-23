# Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning

<div align="center">

[![Python Version](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![ArXiv Paper](https://img.shields.io/badge/cs.CL-arXiv:2510.19807-b31b1b.svg)](https://arxiv.org/abs/2510.19807)

</div>

This is the official implementation of **"[Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning](https://arxiv.org/abs/2510.19807)"**.

**Paper**: [Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning](https://arxiv.org/abs/2510.19807)

<br>

Overview of the Scaf-GRPO framework:

<div align=center>
<img width="100%" src="assets/overview.png"/>
</div>

**Scaf-GRPO** is a progressive training framework designed to overcome the "learning cliff" in reinforcement learning for LLMs. When a model consistently fails on difficult problems, leading to zero-reward signals and stalled progress, Scaf-GRPO intervenes with minimal, hierarchical guidance. By injecting tiered in-prompt hints—from abstract concepts to concrete steps—it enables the model to construct a valid solution, restoring the learning gradient and unlocking its ability to solve problems previously beyond its reach. This on-policy scaffolding approach preserves the model's exploratory autonomy while effectively extending the frontier of its reasoning capabilities.

## Contents
- [Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning](#scaf-grpo-scaffolded-group-relative-policy-optimization-for-enhancing-llm-reasoning)
  - [Contents](#contents)
  - [Code](#code)
  - [Citation](#citation)
  - [Acknowledgement](#acknowledgement)

## Code

> The official source code for Scaf-GRPO is currently being prepared for public release and will be available here soon. The repository will include:
> - The full implementation of the Scaf-GRPO framework.
> - Scripts for data filtering and hierarchical hint generation.
> - A complete training and evaluation pipeline to replicate our results.
>
> Thank you for your interest and patience! Please stay tuned.

## Citation
If you find our work and resources useful in your research, please consider citing our paper:
```bibtex
@article{zhang2025scafgrpo,
  title={{Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning}},
  author={Xichen Zhang, Sitong Wu, Yinghao Zhu, Haoru Tan, Shaozuo Yu, Ziyi He, Jiaya Jia},
  journal={arXiv preprint arXiv:2510.19807},
  year={2025}
}
```

## Acknowledgement
We would like to thank the following projects for their great work and inspiration:
-   [verl](https://github.com/volcengine/verl) for their efficient and robust RL framework for LLMs.
-   The authors of [symeval](https://github.com/tongyx361/symeval) for the robust mathematical evaluation library.