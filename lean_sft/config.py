from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "lean_sft.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    data = yaml.safe_load(cfg_path.read_text()) or {}
    data["dataset_repo"] = os.getenv("HF_DATASET_REPO", data["dataset_repo"])
    data["phase1_repo"] = os.getenv("HF_PHASE1_REPO", data["phase1_repo"])
    data["phase2_repo"] = os.getenv("HF_PHASE2_REPO", data["phase2_repo"])
    data["merged_repo"] = os.getenv("HF_MERGED_REPO", data["merged_repo"])
    return data


def require_env(name: str) -> str:
    load_dotenv(ROOT / ".env")
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required in env or .env")
    return value


def redact(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 10:
        return "***"
    return f"{value[:4]}...{value[-4:]}"
