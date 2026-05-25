# Normalization variants

Normalization modules act on the last hidden dimension of each token state. They are usually placed inside the connection block before a sublayer in this repo.

## `layernorm`

Paper tag: [Layer Normalization](https://arxiv.org/abs/1607.06450)

Implementation: [`layernorm.py`](../src/components/normalization/layernorm.py)
Config: [`layernorm.yaml`](../configs/normalization/layernorm.yaml)

$$
\mu = \frac{1}{d}\sum_i x_i,\quad
\sigma = \sqrt{\frac{1}{d}\sum_i (x_i-\mu)^2},\quad
\mathrm{LN}(x) = \gamma \frac{x-\mu}{\sigma+\epsilon} + \beta
$$

`layernorm` centers and rescales each token independently. The local implementation keeps learned scale `alpha` and bias `bias`.

Use it for the original transformer baseline or when you want explicit mean-centering.

## `rmsnorm`

Paper tag: [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)

Implementation: [`rmsnorm.py`](../src/components/normalization/rmsnorm.py)
Config: [`rmsnorm.yaml`](../configs/normalization/rmsnorm.yaml)

$$
\mathrm{RMS}(x)=\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon},\quad
\mathrm{RMSNorm}(x)=g\frac{x}{\mathrm{RMS}(x)}
$$

`rmsnorm` removes the mean-centering step and keeps only RMS scaling. This reduces work and matches many modern LLM stacks.

Use it with `swiglu`, `gqa_rope`, and decoder-only causal models when you want the modern pre-norm path.
