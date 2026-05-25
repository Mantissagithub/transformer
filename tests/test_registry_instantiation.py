"""Every registered component must instantiate from its YAML defaults without error."""
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD
import yaml

from src.registry import (
    ATTENTION,
    CONNECTION,
    FFN,
    LOSS,
    NORM,
    OPTIMIZER,
    POS,
    SCHEDULER,
)

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
D_MODEL = 64


def _load(group: str, name: str) -> dict:
    cfg = yaml.safe_load((CONFIGS / group / f"{name}.yaml").read_text())
    cfg.pop("name", None)
    return cfg


@pytest.mark.parametrize("name", ATTENTION.names())
def test_attention(name: str) -> None:
    cfg = _load("attention", name)
    cfg.setdefault("d_model", D_MODEL)
    cfg.setdefault("dropout", 0.0)
    ATTENTION.build(name, **cfg)


@pytest.mark.parametrize("name", FFN.names())
def test_ffn(name: str) -> None:
    cfg = _load("feedforward", name)
    cfg.setdefault("d_model", D_MODEL)
    cfg.setdefault("dropout", 0.0)
    FFN.build(name, **cfg)


@pytest.mark.parametrize("name", NORM.names())
def test_norm(name: str) -> None:
    cfg = _load("normalization", name)
    NORM.build(name, d_model=D_MODEL, **cfg)


@pytest.mark.parametrize("name", POS.names())
def test_positional(name: str) -> None:
    cfg = _load("positional", name)
    cfg.setdefault("d_model", D_MODEL)
    cfg.setdefault("max_len", 32)
    if name == "alibi":
        cfg.setdefault("n_heads", 4)
    POS.build(name, **cfg)


@pytest.mark.parametrize("name", CONNECTION.names())
def test_connection(name: str) -> None:
    cfg = _load("connection", name)
    cfg.setdefault("d_model", D_MODEL)
    CONNECTION.build(name, **cfg)


@pytest.mark.parametrize("name", LOSS.names())
def test_loss(name: str) -> None:
    cfg = _load("loss", name)
    LOSS.build(name, **cfg)


@pytest.mark.parametrize("name", SCHEDULER.names())
def test_scheduler_yaml_loads(name: str) -> None:
    _load("scheduler", name)


@pytest.mark.parametrize("name", OPTIMIZER.names())
def test_optimizer_yaml_instantiates(name: str) -> None:
    cfg = _load("optimizer", name)
    model = nn.Linear(4, 2)
    optimizers = OPTIMIZER.build(name, model=model, **cfg)
    assert optimizers


@pytest.mark.parametrize("name", ["ademamix", "mars_adamw"])
def test_custom_optimizer_step(name: str) -> None:
    model = nn.Linear(4, 2)
    opt = OPTIMIZER.build(name, model=model, lr=1e-3)[0]
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    before = model.weight.detach().clone()
    opt.step()
    assert not torch.equal(model.weight, before)


@pytest.mark.parametrize("name", ["inverse_sqrt_warmup", "polynomial_warmup", "wsd"])
def test_scheduler_instantiates_and_steps(name: str) -> None:
    opt = SGD([torch.nn.Parameter(torch.ones(()))], lr=1.0)
    cfg = _load("scheduler", name)
    cfg["total_steps"] = 10
    cfg["warmup_steps"] = 2
    sched = SCHEDULER.build(name, optimizers=[opt], **cfg)[0]
    values = []
    for _ in range(4):
        opt.step()
        sched.step()
        values.append(opt.param_groups[0]["lr"])
    assert all(value >= 0 for value in values)
