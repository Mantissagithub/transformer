from typing import List

import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.registry import OPTIMIZER

from .ademamix import AdEMAMix
from .lion import Lion
from .mars import MarsAdamW


@OPTIMIZER.register("adamw")
def build_adamw(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    **kwargs,
) -> List[Optimizer]:
    return [torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=tuple(betas))]


@OPTIMIZER.register("lion")
def build_lion(model: nn.Module, lr: float = 1e-4, weight_decay: float = 0.0, **kwargs) -> List[Optimizer]:
    return [Lion(model.parameters(), lr=lr, weight_decay=weight_decay)]


@OPTIMIZER.register("ademamix")
def build_ademamix(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    betas: tuple = (0.9, 0.999, 0.9999),
    alpha: float = 5.0,
    eps: float = 1e-8,
    **kwargs,
) -> List[Optimizer]:
    return [
        AdEMAMix(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(betas),
            alpha=alpha,
            eps=eps,
        )
    ]


@OPTIMIZER.register("mars_adamw")
def build_mars_adamw(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    gamma: float = 0.025,
    eps: float = 1e-8,
    **kwargs,
) -> List[Optimizer]:
    return [
        MarsAdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(betas),
            gamma=gamma,
            eps=eps,
        )
    ]


@OPTIMIZER.register("adafactor")
def build_adafactor(model: nn.Module, lr: float = 1e-3, weight_decay: float = 0.0, **kwargs) -> List[Optimizer]:
    try:
        from torch.optim import Adafactor  # PyTorch >= 2.5
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Adafactor requires torch>=2.5; install a newer torch.") from exc
    return [Adafactor(model.parameters(), lr=lr, weight_decay=weight_decay)]


@OPTIMIZER.register("muon_adamw")
def build_muon_adamw(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    **kwargs,
) -> List[Optimizer]:
    """Split: 2D params → Muon, others → AdamW. Falls back to AdamW-only if Muon is unavailable."""
    muon_params = [p for p in model.parameters() if p.dim() == 2]
    other_params = [p for p in model.parameters() if p.dim() != 2]
    optimizers: List[Optimizer] = []
    Muon = getattr(torch.optim, "Muon", None)
    if Muon is not None and muon_params:
        optimizers.append(Muon(muon_params, lr=lr, weight_decay=weight_decay))
    elif muon_params:
        optimizers.append(torch.optim.AdamW(muon_params, lr=lr, weight_decay=weight_decay))
    if other_params:
        optimizers.append(torch.optim.AdamW(other_params, lr=lr, weight_decay=weight_decay))
    return optimizers
