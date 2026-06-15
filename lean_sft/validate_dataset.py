from __future__ import annotations

import argparse

from .config import load_config
from .io import assign_splits, read_jsonl, write_jsonl
from .quality import build_example
from .schema import normalize_code
from .tb import log_scalars


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate, dedupe, and split Lean SFT JSONL.")
    parser.add_argument("input")
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    valid = []
    seen = set()
    rejected = 0
    for row in read_jsonl(args.input):
        try:
            phase = str(row["phase"])
            example = build_example(row, expected_phase=phase, model=str(row.get("generator_model", "")))
            key = normalize_code(example.lean_code)
            if key in seen:
                rejected += 1
                continue
            seen.add(key)
            valid.append(example.to_dict())
        except Exception:
            rejected += 1

    ds_cfg = cfg["dataset"]
    rows = assign_splits(
        valid,
        seed=int(ds_cfg["seed"]),
        validation_fraction=float(ds_cfg["validation_fraction"]),
        test_fraction=float(ds_cfg["test_fraction"]),
    )
    write_jsonl(args.out, rows)
    phase = rows[0]["phase"] if rows else "unknown"
    log_scalars(
        cfg,
        f"dataset/{phase}/validate",
        {
            "accepted": len(rows),
            "rejected": rejected,
            "train": sum(row["split"] == "train" for row in rows),
            "validation": sum(row["split"] == "validation" for row in rows),
            "test": sum(row["split"] == "test" for row in rows),
        },
    )
    print(f"accepted={len(rows)} rejected={rejected} out={args.out}")


if __name__ == "__main__":
    main()
