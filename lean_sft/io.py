from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def assign_splits(rows: list[dict], *, seed: int, validation_fraction: float, test_fraction: float) -> list[dict]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(n * test_fraction)
    n_val = int(n * validation_fraction)
    for idx, row in enumerate(shuffled):
        if idx < n_test:
            row["split"] = "test"
        elif idx < n_test + n_val:
            row["split"] = "validation"
        else:
            row["split"] = "train"
    return shuffled
