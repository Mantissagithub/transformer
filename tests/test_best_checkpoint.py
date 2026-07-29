from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from src.training.checkpoint import best_checkpoint_path, load_checkpoint
from src.training.trainer import Trainer


class _Logger:
    def __init__(self) -> None:
        self.values = []

    def scalar(self, tag, value, step) -> None:
        self.values.append((tag, value, step))

    def flush(self) -> None:
        pass


class _FixedLogits(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(input_ids.shape[0], -1, -1)


def _trainer(tmp_path: Path) -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = OmegaConf.create(
        {
            "training": {
                "ckpt_dir": str(tmp_path),
                "ckpt_basename": "model",
            }
        }
    )
    trainer.dist_env = SimpleNamespace(is_main=True)
    trainer.model = torch.nn.Linear(2, 2)
    trainer.optimizers = [torch.optim.AdamW(trainer.model.parameters())]
    trainer.logger = _Logger()
    trainer.global_step = 10
    trainer.best_val_loss = float("inf")
    return trainer


def test_best_checkpoint_only_saves_improvements(tmp_path, monkeypatch) -> None:
    trainer = _trainer(tmp_path)
    losses = iter([2.0, 2.5, 1.5])
    monkeypatch.setattr(trainer, "_validation_loss", lambda: next(losses))

    first = trainer._validate_and_save_best(epoch=0, tui=None)
    worse = trainer._validate_and_save_best(epoch=0, tui=None)
    better = trainer._validate_and_save_best(epoch=1, tui=None)

    expected = best_checkpoint_path(str(tmp_path), "model")
    assert first == expected
    assert worse is None
    assert better == expected
    assert list(tmp_path.glob("*.pt")) == [expected]
    assert trainer.best_val_loss == 1.5

    restored_model = torch.nn.Linear(2, 2)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters())
    state = load_checkpoint(expected, restored_model, [restored_optimizer])
    assert state["epoch"] == 1
    assert state["global_step"] == 10
    assert state["best_val_loss"] == 1.5
    assert trainer.logger.values == [
        ("val/loss", 2.0, 10),
        ("val/loss", 2.5, 10),
        ("val/loss", 1.5, 10),
    ]


def test_validation_loss_is_token_weighted_and_restores_training_mode() -> None:
    logits = torch.tensor(
        [[[0.0, 3.0, 1.0], [2.0, 0.0, 1.0], [0.0, 1.0, 3.0]]]
    )
    labels = torch.tensor([[1, 0, 2]])
    trainer = Trainer.__new__(Trainer)
    trainer.kind = "causal_lm"
    trainer.device = torch.device("cpu")
    trainer.model = _FixedLogits(logits).train()
    trainer.data = {
        "val_loader": [{"input_ids": torch.ones_like(labels), "labels": labels}],
        "pad_token_id": 0,
    }
    trainer.autocast_dtype = None
    trainer.dist_env = SimpleNamespace(is_dist=False)

    actual = trainer._validation_loss()
    expected = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 3),
        labels.reshape(-1),
        ignore_index=0,
        reduction="sum",
    ).item() / 2

    assert actual == expected
    assert trainer.model.training
