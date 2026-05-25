# Optimizer variants

Optimizers are built through `src.registry.OPTIMIZER` and return a list of PyTorch optimizers. The trainer steps every optimizer in the list, which is why `muon_adamw` can split parameter groups into multiple optimizer instances.

## `adamw`

Paper tag: [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)

Implementation: [`builders.py`](../src/components/optimizers/builders.py)
Config: [`adamw.yaml`](../configs/optimizer/adamw.yaml)

\[
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\quad
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
\]

\[
\theta_t = (1-\eta\lambda)\theta_{t-1} - \eta \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\]

This is the default optimizer for transformer training. Weight decay is decoupled from the adaptive gradient update.

## `lion`

Paper tag: [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675)

Implementation: [`lion.py`](../src/components/optimizers/lion.py)
Config: [`lion.yaml`](../configs/optimizer/lion.yaml)

\[
u_t = \operatorname{sign}(\beta_1 m_{t-1} + (1-\beta_1)g_t),\quad
\theta_t = (1-\eta\lambda)\theta_{t-1} - \eta u_t
\]

\[
m_t = \beta_2 m_{t-1} + (1-\beta_2)g_t
\]

Lion stores one momentum buffer and uses the sign of the blended update. It can be memory-lighter than AdamW but usually needs LR tuning.

## `adafactor`

Paper tag: [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235)

Implementation: [`builders.py`](../src/components/optimizers/builders.py)
Config: [`adafactor.yaml`](../configs/optimizer/adafactor.yaml)

Adafactor factorizes second-moment estimates for matrix-shaped parameters:

\[
V \approx \frac{r c^\top}{\operatorname{mean}(r)}
\]

The local builder uses `torch.optim.Adafactor` when available. It is useful when optimizer-state memory matters.

## `muon_adamw`

Paper tags: [Scaling Muon](https://arxiv.org/abs/2502.16982), [Muon optimizer analysis](https://arxiv.org/abs/2504.16041)

Implementation: [`builders.py`](../src/components/optimizers/builders.py)
Config: [`muon_adamw.yaml`](../configs/optimizer/muon_adamw.yaml)

This builder sends 2D parameters to `torch.optim.Muon` when that optimizer exists in the installed PyTorch build, and sends the remaining parameters to AdamW. If Muon is not available, 2D parameters fall back to AdamW.

Use it as an experimental matrix-parameter optimizer path; verify your PyTorch build before assuming actual Muon updates are active.

## `ademamix`

Paper tag: [The AdEMAMix Optimizer: Better, Faster, Older](https://arxiv.org/abs/2409.03137)

Implementation: [`ademamix.py`](../src/components/optimizers/ademamix.py)
Config: [`ademamix.yaml`](../configs/optimizer/ademamix.yaml)

\[
m_t^{fast}=\beta_1 m_{t-1}^{fast}+(1-\beta_1)g_t
\]

\[
m_t^{slow}=\beta_3 m_{t-1}^{slow}+(1-\beta_3)g_t
\]

\[
\theta_t = (1-\eta\lambda)\theta_{t-1} - \eta\frac{\hat m_t^{fast}+\alpha m_t^{slow}}{\sqrt{\hat v_t}+\epsilon}
\]

AdEMAMix extends AdamW with a slow gradient EMA so older gradients can keep influence over the update. This repo implements the AdamW-style variant with decoupled weight decay.

## `mars_adamw`

Paper tag: [MARS: Unleashing the Power of Variance Reduction for Training Large Models](https://arxiv.org/abs/2411.10438)

Implementation: [`mars.py`](../src/components/optimizers/mars.py)
Config: [`mars_adamw.yaml`](../configs/optimizer/mars_adamw.yaml)

\[
\tilde g_t = g_t + \gamma\frac{\beta_1}{1-\beta_1}(g_t-g_{t-1})
\]

Then AdamW moments are updated using \(\tilde g_t\) instead of \(g_t\).

Use it for variance-reduction experiments that should still fit the existing AdamW-like optimizer contract.
