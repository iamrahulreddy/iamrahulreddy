## Muskula Rahul

![Profile Views](https://komarev.com/ghpvc/?username=iamrahulreddy&color=blueviolet&style=flat-square)
[![Blog](https://img.shields.io/badge/Blog-neuralnets.dev-orange?style=flat-square&logo=astro)](https://neuralnets.dev)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-iamrahulreddy-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/iamrahulreddy/)
[![Hugging Face](https://img.shields.io/badge/🤗-HuggingFace-yellow?style=flat-square)](https://huggingface.co/iamrahulreddy)

I build LLM systems that sit close to the metal — MoE architectures, attention kernels, speculative decoding. The kind of work where a misaligned memory access costs you a day.

> [!NOTE]
> The workflow: read the paper, implement it, fix what broke. The CUDA race conditions weren't in the abstract.

## Projects

### [Keiro](https://huggingface.co/iamrahulreddy/Keiro) — Sparse MoE on Qwen2.5-3B

Retrofitted Sparse Mixture-of-Experts into Qwen2.5-3B. A Top-2 router activates 2 of 8 LoRA experts per transformer block, leaving the frozen FFN untouched and routing through Rank-16 adapters instead. Active compute stays identical to the dense baseline. The model adds **19.46M trainable parameters (0.63% of total)** and retains **95.4% of GSM8K** performance.

```
What actually needed fixing:
├── CUDA race condition in index_add_ with duplicate Top-K indices
├── BFloat16 cumsum upcast mismatch in the coalesce path
└── 4.7× autoregressive inference bottleneck — resolved by bypassing
    capacity buffers during single-token generation

lm-evaluation-harness results vs. base model:
├── HellaSwag     −0.13%
├── ARC-Challenge −0.17%
└── GSM8K         −3.19%
```

### [Prolepsis](https://github.com/iamrahulreddy/Prolepsis) — Speculative Decoding

A Qwen 1.7B draft model generates candidate tokens; a Qwen 8B target verifies them in a single parallel pass. A rejection sampling pipeline ensures the output distribution is mathematically identical to running the target model alone.

| Metric | Result |
|---|---|
| Speedup on A100 | **1.30×** |
| Acceptance Rate | **~56.5%** across mixed-domain prompts |
| Output Distribution | Identical to target |

### [FlashTile](https://github.com/iamrahulreddy/FlashTile) — Flash Attention V1/V2

Implements block-wise tiling, online softmax, and recomputation-based backward passes to cut attention storage from **O(N²) to O(N)**. Covers GQA and MQA variants, with a forward-only Triton kernel included for benchmarking.

### [Substrata9](https://github.com/iamrahulreddy/Substrata9) — Linux Introspection Toolkit

Pure Bash. No compilation, no dependencies. Reads `/proc` to surface memory maps, file descriptors, process hierarchies, and runtime anomalies. Outputs JSON — slots into observability, debugging, and forensics pipelines without modification.

### [Mission Cipher](https://github.com/iamrahulreddy/cipher) — GraphRAG App

Combines cosine-similarity search over semantic embeddings with a live knowledge graph (NetworkX) to answer questions with richer contextual grounding than plain RAG. Deployed on GCE behind NGINX, with Flask and Gunicorn communicating over a Unix socket.

## Writing

> [!TIP]
> [neuralnets.dev](https://neuralnets.dev) — LLM architecture, inference, GPU programming, and occasionally the math underneath all of it. The goal is precision over vibe — the writeups get into what the papers skip and what the code alone won't tell you.
