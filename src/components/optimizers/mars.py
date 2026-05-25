import torch
from torch.optim import Optimizer


class MarsAdamW(Optimizer):
    """MARS-AdamW optimizer (https://arxiv.org/abs/2411.10438).

    This is the AdamW-style MARS variant: it forms a variance-reduced gradient
    estimate from the current gradient and the previous gradient delta, then
    feeds that estimate through AdamW moments.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas=(0.9, 0.999),
        gamma: float = 0.025,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError(f"invalid lr: {lr}")
        if eps < 0:
            raise ValueError(f"invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"invalid weight_decay: {weight_decay}")
        if len(betas) != 2:
            raise ValueError("betas must contain (beta1, beta2)")
        if not all(0 <= beta < 1 for beta in betas):
            raise ValueError(f"invalid betas: {betas}")
        if gamma < 0:
            raise ValueError(f"invalid gamma: {gamma}")

        defaults = dict(lr=lr, betas=betas, gamma=gamma, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            gamma = group["gamma"]
            eps = group["eps"]
            wd = group["weight_decay"]
            correction_scale = gamma * beta1 / max(1e-16, 1 - beta1)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("MarsAdamW does not support sparse gradients")

                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["prev_grad"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                prev_grad = state["prev_grad"]

                if step == 1:
                    variance_reduced = grad
                else:
                    variance_reduced = grad.add(grad - prev_grad, alpha=correction_scale)
                prev_grad.copy_(grad)

                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                exp_avg.mul_(beta1).add_(variance_reduced, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(variance_reduced, variance_reduced, value=1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = lr / bias_correction1
                denom = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss
