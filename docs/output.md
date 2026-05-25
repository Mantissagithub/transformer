# Embeddings, projections, and loss

These components define how token ids enter the model and how hidden states become vocabulary logits.

## `learned` embedding

Paper tag: standard transformer embedding table from [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

Implementation: [`learned.py`](../src/components/embeddings/learned.py)

$$
E(x_i)=\sqrt{d_{\text{model}}}\,W_E[x_i]
$$

The local embedding optionally scales vectors by $\sqrt{d_{\text{model}}}$, matching the original transformer convention.

## `linear` projection

Implementation: [`linear.py`](../src/components/projection/linear.py)

$$
\ell_t = W_o h_t + b_o
$$

This is an untied vocabulary projection. With `log_softmax: true`, it returns log-probabilities directly.

## `tied` projection

Paper tag: [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)

Implementation: [`tied.py`](../src/components/projection/tied.py)

$$
\ell_t = h_t W_E^\top
$$

The output projection reuses the input embedding matrix. This reduces parameters and aligns input/output token spaces.

## `cross_entropy`

Implementation: [`cross_entropy.py`](../src/components/losses/cross_entropy.py)
Config: [`cross_entropy.yaml`](../configs/loss/cross_entropy.yaml)

$$
\mathcal{L} = -\log p_\theta(y_t \mid x_{\le t})
$$

The registered loss is PyTorch cross entropy with optional `label_smoothing` and `ignore_index`. The trainer sets `ignore_index` to the dataset pad token id.
