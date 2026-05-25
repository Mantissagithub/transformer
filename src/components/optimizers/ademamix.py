import torch
from torch.optim import Optimizer


class AdEMAMix(Optimizer):
    """AdEMAMix optimizer (https://arxiv.org/abs/2409.03137).

    AdamW-style update with a fast EMA and a slow EMA of gradients. The slow
    EMA lets older gradients keep a controlled influence over the step.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas=(0.9, 0.999, 0.9999),
        alpha: float = 5.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError(f"invalid lr: {lr}")
        if eps < 0:
            raise ValueError(f"invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"invalid weight_decay: {weight_decay}")
        if len(betas) != 3:
            raise ValueError("betas must contain (beta1, beta2, beta3)")
        if not all(0 <= beta < 1 for beta in betas):
            raise ValueError(f"invalid betas: {betas}")
        if alpha < 0:
            raise ValueError(f"invalid alpha: {alpha}")

        defaults = dict(lr=lr, betas=betas, alpha=alpha, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2, beta3 = group["betas"]
            alpha = group["alpha"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdEMAMix does not support sparse gradients")

                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg_fast"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["exp_avg_slow"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]
                exp_avg_fast = state["exp_avg_fast"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg_slow = state["exp_avg_slow"]

                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                exp_avg_fast.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg_slow.mul_(beta3).add_(grad, alpha=1 - beta3)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                fast = exp_avg_fast / bias_correction1
                denom = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(eps)
                update = fast.add(exp_avg_slow, alpha=alpha)
                p.addcdiv_(update, denom, value=-lr)
        return loss
