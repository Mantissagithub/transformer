# Scheduler variants

Schedulers are built through `src.registry.SCHEDULER` and return one LambdaLR-style scheduler per optimizer. The trainer infers `total_steps` from `training.max_steps` or the dataloader length, then passes it into the scheduler builder.

## `none`

Config: [`none.yaml`](../configs/scheduler/none.yaml)

No scheduler is attached. The optimizer LR stays fixed unless the optimizer itself changes it.

Use it for debugging or schedule-free optimizer experiments.

## `linear_warmup`

Paper tag: common transformer warmup/decay recipe.

Config: [`linear_warmup.yaml`](../configs/scheduler/linear_warmup.yaml)

$$
\lambda(t)=t/T_w \quad \text{for } t<T_w
$$

$$
\lambda(t)=\max\left(r_{\min}, 1-\frac{t-T_w}{T-T_w}\right) \quad \text{for } t\ge T_w
$$

The LR warms up linearly, then decays linearly to `min_lr_ratio`.

## `cosine_warmup`

Paper tag: cosine annealing from [SGDR](https://arxiv.org/abs/1608.03983), used broadly in transformer training.

Config: [`cosine_warmup.yaml`](../configs/scheduler/cosine_warmup.yaml)

$$
\lambda(t)=r_{\min}+(1-r_{\min})\frac{1+\cos(\pi p)}{2},\quad
p=\frac{t-T_w}{T-T_w}
$$

This is a smooth alternative to linear decay. It is the default-style schedule for longer pretraining runs in this repo.

## `inverse_sqrt_warmup`

Paper tag: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

Config: [`inverse_sqrt_warmup.yaml`](../configs/scheduler/inverse_sqrt_warmup.yaml)

$$
\lambda(t)=t/T_w \quad \text{for } t<T_w
$$

$$
\lambda(t)=\sqrt{T_w/t} \quad \text{for } t\ge T_w
$$

This mirrors the original transformer decay shape after warmup, without including the `d_model^{-0.5}` factor because the base LR already carries scale.

## `polynomial_warmup`

Paper tag: polynomial decay is a standard large-scale training schedule family; see recent scaling-law schedule work such as [Optimal Learning-Rate Schedules under Functional Scaling Laws](https://arxiv.org/abs/2602.06797).

Config: [`polynomial_warmup.yaml`](../configs/scheduler/polynomial_warmup.yaml)

$$
\lambda(t)=r_{\min}+(1-r_{\min})(1-p)^\alpha
$$

`power: 1.0` gives linear decay. Larger powers hold LR higher early and decay harder near the end; smaller powers decay earlier.

## `wsd`

Paper tag: [Understanding Warmup-Stable-Decay Learning Rates](https://arxiv.org/abs/2410.05192)

Config: [`wsd.yaml`](../configs/scheduler/wsd.yaml)

$$
\lambda(t)=t/T_w \quad \text{for } t<T_w
$$

$$
\lambda(t)=1 \quad \text{for } T_w \le t < T_d
$$

$$
\lambda(t)=r_{\min}+(1-r_{\min})D\left(\frac{t-T_d}{T-T_d}\right) \quad \text{for } t\ge T_d
$$

`D` is cosine or linear decay. WSD is useful when the main training run should stay at peak LR for most of the budget, then decay only near the end.
