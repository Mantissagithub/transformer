import math
from typing import List, Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

from src.registry import SCHEDULER


@SCHEDULER.register("none")
def build_none(optimizers: List[Optimizer], **kwargs) -> List[Optional[_LRScheduler]]:
    return [None for _ in optimizers]


@SCHEDULER.register("linear_warmup")
def build_linear_warmup(
    optimizers: List[Optimizer],
    warmup_steps: int = 1000,
    total_steps: int = 100_000,
    min_lr_ratio: float = 0.0,
    **kwargs,
) -> List[_LRScheduler]:
    def fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr_ratio, 1.0 - progress)

    return [LambdaLR(opt, lr_lambda=fn) for opt in optimizers]


@SCHEDULER.register("cosine_warmup")
def build_cosine_warmup(
    optimizers: List[Optimizer],
    warmup_steps: int = 1000,
    total_steps: int = 100_000,
    min_lr_ratio: float = 0.0,
    **kwargs,
) -> List[_LRScheduler]:
    def fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos

    return [LambdaLR(opt, lr_lambda=fn) for opt in optimizers]


@SCHEDULER.register("inverse_sqrt_warmup")
def build_inverse_sqrt_warmup(
    optimizers: List[Optimizer],
    warmup_steps: int = 1000,
    min_lr_ratio: float = 0.0,
    **kwargs,
) -> List[_LRScheduler]:
    def fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        ratio = math.sqrt(max(1, warmup_steps) / max(step, 1))
        return max(min_lr_ratio, ratio)

    return [LambdaLR(opt, lr_lambda=fn) for opt in optimizers]


@SCHEDULER.register("polynomial_warmup")
def build_polynomial_warmup(
    optimizers: List[Optimizer],
    warmup_steps: int = 1000,
    total_steps: int = 100_000,
    power: float = 1.0,
    min_lr_ratio: float = 0.0,
    **kwargs,
) -> List[_LRScheduler]:
    def fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        decay = (1.0 - min(progress, 1.0)) ** power
        return min_lr_ratio + (1.0 - min_lr_ratio) * decay

    return [LambdaLR(opt, lr_lambda=fn) for opt in optimizers]


@SCHEDULER.register("wsd")
def build_wsd(
    optimizers: List[Optimizer],
    warmup_steps: int = 1000,
    total_steps: int = 100_000,
    stable_ratio: float = 0.9,
    decay_steps: Optional[int] = None,
    min_lr_ratio: float = 0.0,
    decay: str = "cosine",
    **kwargs,
) -> List[_LRScheduler]:
    if not 0 <= stable_ratio <= 1:
        raise ValueError(f"stable_ratio must be in [0, 1], got {stable_ratio}")
    if decay_steps is None:
        decay_steps = max(1, int((total_steps - warmup_steps) * (1.0 - stable_ratio)))
    decay_start = max(warmup_steps, total_steps - decay_steps)

    def fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if step < decay_start:
            return 1.0
        progress = (step - decay_start) / max(1, total_steps - decay_start)
        progress = min(progress, 1.0)
        if decay == "linear":
            tail = 1.0 - progress
        elif decay == "cosine":
            tail = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"unknown wsd decay: {decay}")
        return min_lr_ratio + (1.0 - min_lr_ratio) * tail

    return [LambdaLR(opt, lr_lambda=fn) for opt in optimizers]
