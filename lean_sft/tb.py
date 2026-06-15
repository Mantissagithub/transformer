from __future__ import annotations

import os
import warnings
from typing import Mapping

from .config import ROOT


def log_scalars(cfg: dict, namespace: str, values: Mapping[str, float | int], step: int = 0) -> None:
    log_dir = ROOT / str(cfg.get("logging", {}).get("tensorboard_dir", "runs/lean-starcoder-sft")) / namespace
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    warnings.filterwarnings("ignore", category=FutureWarning, module="keras")
    try:
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
        import time
    except Exception:
        return

    writer = EventFileWriter(str(log_dir))
    wall_time = time.time()
    for key, value in values.items():
        summary = Summary(value=[Summary.Value(tag=key, simple_value=float(value))])
        writer.add_event(Event(wall_time=wall_time, step=step, summary=summary))
    writer.close()
