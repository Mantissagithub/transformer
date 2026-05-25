# Attention variants

This page keeps the attention family in one place. Each variant has its own architecture card, paper tag, local implementation/config links, and a short note on when to use it.

The diagrams use the same grammar throughout: blue blocks are query streams, green blocks are key streams, orange blocks are value streams, the matrix panel shows the visible attention pattern, and the memory panel shows the KV-cache shape relative to full MHA.

## `mha`

![MHA architecture](assets/mha.svg)

Multi-head attention is the baseline from [Attention Is All You Need](https://arxiv.org/abs/1706.03762). It projects the input into separate query, key, and value tensors, computes dense token-to-token attention, and applies the attention pattern to the values.

- Implementation: [`mha.py`](../../src/components/attention/mha.py)
- Config: [`mha.yaml`](../../configs/attention/mha.yaml)
- Use when: you want the clean reference path before testing cache or context-efficiency changes.

## `mqa`

![MQA architecture](assets/mqa.svg)

Multi-query attention follows [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150). Query heads stay separate, but all heads share one key stream and one value stream, reducing decode-time KV bandwidth.

- Implementation: [`mqa.py`](../../src/components/attention/mqa.py)
- Config: [`mqa.yaml`](../../configs/attention/mqa.yaml)
- Use when: KV-cache size and memory bandwidth matter more than per-head KV capacity.

## `gqa`

![GQA architecture](assets/gqa.svg)

Grouped-query attention follows [GQA](https://arxiv.org/abs/2305.13245). It keeps multiple KV heads, but fewer than the number of query heads, so it sits between MHA and MQA.

- Implementation: [`gqa.py`](../../src/components/attention/gqa.py)
- Config: [`gqa.yaml`](../../configs/attention/gqa.yaml)
- Use when: you want most of MHA's modeling shape with a smaller KV cache.

## `gqa_rope`

![GQA with RoPE architecture](assets/gqa_rope.svg)

`gqa_rope` combines [GQA](https://arxiv.org/abs/2305.13245) with rotary position embeddings from [RoFormer](https://arxiv.org/abs/2104.09864). In this repo, RoPE is applied inside attention to Q and K before grouped-KV attention runs.

- Implementation: [`gqa_rope.py`](../../src/components/attention/gqa_rope.py)
- Config: [`gqa_rope.yaml`](../../configs/attention/gqa_rope.yaml)
- Use when: you want grouped KV heads plus rotary positions.

## `sliding_window`

![Sliding-window attention architecture](assets/sliding_window.svg)

Sliding-window attention uses the local-window idea associated with [Longformer](https://arxiv.org/abs/2004.05150). It keeps dense heads, but the attention mask only allows nearby tokens through.

- Implementation: [`sliding_window.py`](../../src/components/attention/sliding_window.py)
- Config: [`sliding_window.yaml`](../../configs/attention/sliding_window.yaml)
- Use when: you want a local-context baseline without changing the head layout.

## `sliding_gqa`

![Sliding GQA architecture](assets/sliding_gqa.svg)

`sliding_gqa` combines local-window attention with grouped KV heads. It has both the reduced attention span of sliding-window attention and the smaller KV layout of GQA.

- Implementation: [`sliding_gqa.py`](../../src/components/attention/sliding_gqa.py)
- Config: [`sliding_gqa.yaml`](../../configs/attention/sliding_gqa.yaml)
- Use when: you want local attention and grouped cache savings together.

## `gemma3_hybrid`

![Gemma 3 hybrid attention architecture](assets/gemma3_hybrid.svg)

The Gemma-style hybrid pattern is tagged to the [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786). It composes existing repo variants: several local `sliding_gqa` layers followed by a global `gqa_rope` layer.

- Implementation: composed in [`builder.py`](../../src/model/builder.py)
- Config: [`gemma3_hybrid.yaml`](../../configs/attention/gemma3_hybrid.yaml)
- Use when: you want an interleaved local/global pattern without adding a new kernel.

## `mla`

![MLA architecture](assets/mla.svg)

Multi-head latent attention is tagged to [DeepSeek-V2](https://arxiv.org/abs/2405.04434). It caches a compact latent KV state plus a decoupled RoPE key, then reconstructs per-head K/V during attention.

- Implementation: [`mla.py`](../../src/components/attention/mla.py)
- Config: [`mla.yaml`](../../configs/attention/mla.yaml)
- Use when: KV-cache memory is the main target. The current implementation saves cache memory, but does not yet reduce decode-time projection FLOPs.

## `csa`

![CSA architecture](assets/csa.svg)

Compressed sparse attention is tagged to the [DeepSeek-V4 docs](https://huggingface.co/docs/transformers/main/model_doc/deepseek_v4). It compresses KV blocks, scores compressed entries with a sparse indexer, gathers top-k memory, keeps a local branch, and projects the result back through grouped output projection.

- Implementation: [`csa.py`](../../src/components/attention/csa.py)
- Config: [`csa.yaml`](../../configs/attention/csa.yaml)
- Use when: you want a long-context experiment that keeps local detail while sparsifying compressed long-range memory.

## `hca`

![HCA architecture](assets/hca.svg)

Heavily compressed attention is also tagged to the [DeepSeek-V4 docs](https://huggingface.co/docs/transformers/main/model_doc/deepseek_v4). It is the simpler compressed sibling of CSA: single-stream block compression, no sparse indexer, no local branch, then shared-KV attention over compressed entries.

- Implementation: [`hca.py`](../../src/components/attention/hca.py)
- Config: [`hca.yaml`](../../configs/attention/hca.yaml)
- Use when: you want a simpler compressed-memory attention path with heavier KV reduction.

## Model compatibility

The encoder-decoder path needs attention modules that support cross-attention and bidirectional encoder attention. In this repo, `mha`, `mqa`, `gqa`, `gqa_rope`, `sliding_window`, `sliding_gqa`, and `gemma3_hybrid` satisfy that path.

`csa`, `hca`, and `mla` are self-attention-only variants and should run through the causal-LM path. The builder enforces this split so an experiment does not silently train with the wrong topology.
