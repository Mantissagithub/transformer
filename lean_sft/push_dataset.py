from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, create_repo

from .config import load_config, require_env
from .io import read_jsonl
from .tb import log_scalars


def main() -> None:
    parser = argparse.ArgumentParser(description="Push processed Lean SFT data to Hugging Face.")
    parser.add_argument("--syntax", required=True)
    parser.add_argument("--proof", required=True)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    token = require_env("HF_TOKEN")
    repo_id = args.repo_id or cfg["dataset_repo"]
    rows = read_jsonl(args.syntax) + read_jsonl(args.proof)
    splits = {
        split: [row for row in rows if row.get("split") == split]
        for split in ("train", "validation", "test")
    }
    dataset = DatasetDict({name: Dataset.from_list(items) for name, items in splits.items()})
    create_repo(repo_id, repo_type="dataset", token=token, private=args.private, exist_ok=True)
    dataset.push_to_hub(repo_id, token=token, private=args.private)
    _write_card(repo_id, rows, token)
    log_scalars(
        cfg,
        "dataset/push",
        {
            "rows": len(rows),
            "train": len(splits["train"]),
            "validation": len(splits["validation"]),
            "test": len(splits["test"]),
        },
    )
    print(f"pushed {sum(len(v) for v in splits.values())} rows to {repo_id}")


def _write_card(repo_id: str, rows: list[dict], token: str) -> None:
    counts = {}
    for row in rows:
        key = (row.get("phase"), row.get("split"))
        counts[key] = counts.get(key, 0) + 1
    lines = [
        "# Lean StarCoder SFT Data",
        "",
        "Synthetic Lean syntax and proof-completion data generated through OpenRouter.",
        "",
        "| phase | split | rows |",
        "|---|---:|---:|",
    ]
    for (phase, split), count in sorted(counts.items()):
        lines.append(f"| {phase} | {split} | {count} |")
    path = Path("/tmp/lean_starcoder_dataset_README.md")
    path.write_text("\n".join(lines) + "\n")
    HfApi(token=token).upload_file(
        path_or_fileobj=str(path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )


if __name__ == "__main__":
    main()
