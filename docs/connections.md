# Connection variants

Connection modules define how each sublayer writes back into the residual stream. In this repo they also own whether the block state is a standard tensor `(batch, seq, d)` or multiple parallel streams `(batch, seq, n, d)`.

## `residual`

Paper tags: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

Implementation: [`residual.py`](../src/components/connections/residual.py)
Config: [`residual.yaml`](../configs/connection/residual.yaml)

$$
h_{\ell+1} = h_\ell + \operatorname{Dropout}(F(\operatorname{Norm}(h_\ell)))
$$

This is the normal pre-norm residual block. `F` is the attention or feed-forward sublayer. The state shape stays `(batch, seq, d)`.

Use it for the simplest stable transformer path.

## `residual_sandwich`

Paper tag: repo config variant, related to pre/post-norm transformer design.

Config: [`residual_sandwich.yaml`](../configs/connection/residual_sandwich.yaml)

$$
h_{\ell+1} = h_\ell + \operatorname{Dropout}(\operatorname{Norm}_2(F(\operatorname{Norm}_1(h_\ell))))
$$

This config uses the same `residual` implementation with `post_norm: true`. It normalizes before and after the sublayer.

Use it when testing whether extra normalization after the branch improves stability.

## `hyperconnection`

Paper tag: [Hyper-Connections](https://arxiv.org/abs/2409.19606)

Implementation: [`hyperconnection.py`](../src/components/connections/hyperconnection.py)
Config: [`hyperconnection.yaml`](../configs/connection/hyperconnection.yaml)

Hyper-connections widen the residual state into `n` parallel streams:

$$
H_\ell \in \mathbb{R}^{b\times s\times n\times d}
$$

The width connection mixes streams:

$$
\tilde H = \alpha(H)H
$$

Then one branch is passed through the sublayer and written back through depth coefficients:

$$
H_{\ell+1} = \operatorname{depth\_mix}(\tilde H, \beta(H)F(\tilde H_0))
$$

The local implementation has static alpha/beta parameters and optional dynamic alpha/beta predicted from normalized hidden states.

Use it when experimenting with widened residual streams and learned cross-layer mixing.

## `mhc`

Paper tag: [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880)

Implementation: [`mhc.py`](../src/components/connections/mhc.py)
Config: [`mhc.yaml`](../configs/connection/mhc.yaml)

mHC keeps the multi-stream residual state from hyper-connections but constrains the stream-mixing matrix with Sinkhorn-Knopp normalization:

$$
\alpha_{\text{mc}} = \operatorname{Sinkhorn}(\exp(\alpha))
$$

This pushes the connection matrix toward a doubly-stochastic manifold, restoring a more identity-like skip path while keeping learned stream mixing.

Use it when testing the hyper-connection idea with an explicit stability constraint.
