# Feed-forward variants

Transformer feed-forward blocks are position-wise MLPs applied independently to every token. For an input token state $x \in \mathbb{R}^{d}$, the block returns another $d$-dimensional state.

## `relu_ffn`

Paper tag: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

Implementation: [`vanilla.py`](../src/components/feedforward/vanilla.py)
Config: [`relu_ffn.yaml`](../configs/feedforward/relu_ffn.yaml)

$$
\operatorname{FFN}(x) = W_2\,\operatorname{ReLU}(W_1x + b_1) + b_2
$$

This is the original transformer-style position-wise feed-forward block. It is the simplest baseline: one expansion projection, one ReLU nonlinearity, dropout, then one projection back to `d_model`.

Use it when you want a conservative baseline or when comparing attention/connection changes without also changing the MLP.

## `geglu`

Paper tag: [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

Implementation: [`geglu.py`](../src/components/feedforward/geglu.py)
Config: [`geglu.yaml`](../configs/feedforward/geglu.yaml)

$$
\operatorname{GEGLU}(x) = W_{\text{down}}\left(\operatorname{GELU}(W_g x) \odot W_u x\right)
$$

`geglu` splits the expansion into a gate branch and an up branch. The GELU-activated gate decides which expanded channels pass through. In this repo both expansion projections are bias-free, then dropout is applied before the down projection.

Use it when you want a gated MLP with a smooth GELU gate.

## `swiglu`

Paper tag: [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

Implementation: [`swiglu.py`](../src/components/feedforward/swiglu.py)
Config: [`swiglu.yaml`](../configs/feedforward/swiglu.yaml)

$$
\operatorname{SwiGLU}(x) = W_{\text{down}}\left(\operatorname{SiLU}(W_g x) \odot W_u x\right)
$$

`swiglu` is the modern gated MLP default in many decoder-only transformer families. The SiLU gate is smoother than ReLU and keeps small negative inputs active instead of hard-zeroing them.

Use it when building a more modern decoder-style stack with RMSNorm, RoPE, GQA, and tied output embeddings.
