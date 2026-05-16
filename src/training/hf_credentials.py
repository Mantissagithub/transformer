from __future__ import annotations

import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.prompt import Prompt


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def ensure_hf_credentials(
    cfg: DictConfig,
    *,
    console: Console | None = None,
    prompt_if_missing: bool = False,
) -> None:
    console = console or Console()
    hf_cfg = cfg.training.get("hf", {})
    push_enabled = bool(hf_cfg.get("push", False))

    token = _ensure_env_value(
        env_var="HF_TOKEN",
        prompt="Enter HF write token",
        console=console,
        required=push_enabled,
        prompt_if_missing=prompt_if_missing,
        password=True,
    )
    username = _ensure_env_value(
        env_var="HF_USERNAME",
        prompt="Enter HF username",
        console=console,
        required=push_enabled and not hf_cfg.get("repo_id"),
        prompt_if_missing=prompt_if_missing,
        password=False,
    )

    if not push_enabled:
        return

    repo_id = hf_cfg.get("repo_id")
    if not repo_id:
        if not username:
            raise RuntimeError("HF_USERNAME is required when training.hf.push=true")

        from huggingface_hub import HfApi

        api = HfApi(token=token)
        base = f"{cfg.experiment_name}-{_config_suffix(cfg)}-{_final_checkpoint_index(cfg)}"
        candidate = f"{username}/{base}"
        if not cfg.training.get("preload") and _repo_exists(api, candidate):
            n = 2
            while _repo_exists(api, f"{username}/{base}-{n}"):
                n += 1
            base = f"{base}-{n}"
            candidate = f"{username}/{base}"
            console.print(f"[yellow]Repo exists; using {base} instead.[/]")
        OmegaConf.set_struct(cfg, False)
        cfg.experiment_name = base
        cfg.training.ckpt_basename = base
        OmegaConf.set_struct(cfg, True)

        repo_id = candidate
        OmegaConf.set_struct(cfg, False)
        cfg.training.hf.repo_id = repo_id
        OmegaConf.set_struct(cfg, True)

    from huggingface_hub import HfApi

    try:
        user = HfApi(token=token).whoami()["name"]
    except Exception as exc:
        raise RuntimeError(f"HF token validation failed: {exc}") from exc
    console.print(f"[green]✓ HF authenticated as {user} → {repo_id}[/]")


def _ensure_env_value(
    *,
    env_var: str,
    prompt: str,
    console: Console,
    required: bool,
    prompt_if_missing: bool,
    password: bool,
) -> str | None:
    value = os.environ.get(env_var)
    if value:
        return value
    if not required and not prompt_if_missing:
        return None
    console.print(f"[yellow]{env_var} not found in env or .env.[/]")
    value = Prompt.ask(prompt, password=password).strip()
    if not value:
        if required:
            raise RuntimeError(f"{env_var} is required for this run")
        return None
    os.environ[env_var] = value
    _append_env(PROJECT_ROOT / ".env", env_var, value)
    console.print(f"[green]Saved {env_var} to .env[/]")
    return value


def _append_env(path: Path, key: str, value: str) -> None:
    lines: list[str] = []
    if path.exists():
        lines = [
            line
            for line in path.read_text().splitlines()
            if not line.startswith(f"{key}=")
        ]
    lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n")


def _repo_exists(api, repo_id: str) -> bool:
    try:
        api.repo_info(repo_id, repo_type="model")
        return True
    except Exception:
        return False


def _config_suffix(cfg: DictConfig) -> str:
    t = cfg.training
    bs = int(t.batch_size)
    accum = int(t.get("gradient_accumulation_steps", 1))
    parts = [str(cfg.data.get("name", "")), f"bs{bs * accum}"]
    max_steps = int(t.get("max_steps", 0))
    if max_steps > 0:
        parts.append(f"s{max_steps}")
    else:
        parts.append(f"e{int(t.get('num_epochs', 1))}")
    parts.append(str(t.get("precision", "fp32")))
    return "-".join(p for p in parts if p)


def _final_checkpoint_index(cfg: DictConfig) -> str:
    num_epochs = int(cfg.training.get("num_epochs", 1))
    return str(max(0, num_epochs - 1))
