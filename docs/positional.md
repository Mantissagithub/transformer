# Positional variants

Position modules give the transformer information about token order. In this repo some variants modify embeddings directly, while RoPE/ALiBi store tables or biases for attention modules to consume.

## `sinusoidal`

Paper tag: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

Implementation: [`sinusoidal.py`](../src/components/positional/sinusoidal.py)
Config: [`sinusoidal.yaml`](../configs/positional/sinusoidal.yaml)

$$
\mathrm{PE}_{p,2i}=\sin\left(p / 10000^{2i/d}\right),\quad
\mathrm{PE}_{p,2i+1}=\cos\left(p / 10000^{2i/d}\right)
$$

The encoding is added to token embeddings before the transformer stack. It is deterministic and has no learned position table.

Use it for the original transformer baseline.

## `rope`

Paper tag: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

Implementation: [`rope.py`](../src/components/positional/rope.py)
Config: [`rope.yaml`](../configs/positional/rope.yaml)

$$
\mathrm{RoPE}(q_p) = R_p q_p,\quad
\mathrm{RoPE}(k_t)=R_t k_t,\quad
q_p^\top k_t \rightarrow q_p^\top R_{t-p} k_t
$$

RoPE rotates query and key channels by position-dependent angles, turning absolute position rotations into relative offsets inside the attention dot product. In this repo the module stores cosine/sine tables; RoPE-aware attention modules apply the rotation inside attention.

Use it for modern causal-LM attention, especially `gqa_rope` and `sliding_gqa`.

## `alibi`

Paper tag: [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)

Implementation: [`alibi.py`](../src/components/positional/alibi.py)
Config: [`alibi.yaml`](../configs/positional/alibi.yaml)

$$
\mathrm{score}_{h,i,j} = \frac{q_{h,i}^\top k_{h,j}}{\sqrt{d_h}} + m_h(i-j)
$$

ALiBi does not add a position vector to embeddings. It adds a head-specific linear distance bias to attention scores, usually favoring recent context.

Use it when extrapolation beyond the training length matters and you want fixed positional bias instead of rotation.

## `rope_nope_hybrid`

Paper tag: local config pattern inspired by modern partial-RoPE attention stacks.

Config: [`rope_nope_hybrid.yaml`](../configs/positional/rope_nope_hybrid.yaml)

This config keeps `name: rope` but adds `skip_every_n`. It is a configuration-level pattern for mixing RoPE-aware and non-RoPE behavior in selected layers when the builder/attention path supports it.

Use it only when the target attention variant knows how to interpret the hybrid setting.
