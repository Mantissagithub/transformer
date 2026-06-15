from __future__ import annotations

import argparse
from rich.progress import Progress

from .config import load_config, require_env
from .io import write_jsonl
from .openrouter import chat_json, resolve_model
from .prompts import proof_prompt, syntax_prompt
from .quality import build_example
from .tb import log_scalars


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Lean SFT JSONL with OpenRouter.")
    parser.add_argument("--phase", choices=["syntax", "proof"], required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    api_key = "dry-run" if args.dry_run else require_env("OPENROUTER_API_KEY")
    model = cfg["openrouter"]["requested_model"]
    if not args.dry_run:
        model = resolve_model(api_key, cfg)

    rows = []
    attempts = 0
    max_attempts = max(args.count * 3, args.count + 10)
    with Progress() as progress:
        task = progress.add_task(f"generate {args.phase}", total=args.count)
        for idx in range(max_attempts):
            if len(rows) >= args.count:
                break
            attempts += 1
            if args.dry_run:
                raw = _fake_row(args.phase, idx, model)
            else:
                messages = syntax_prompt(idx) if args.phase == "syntax" else proof_prompt(idx)
                raw = chat_json(
                    api_key=api_key,
                    base_url=cfg["openrouter"]["base_url"],
                    model=model,
                    messages=messages,
                    timeout_sec=int(cfg["openrouter"]["timeout_sec"]),
                    max_retries=int(cfg["openrouter"]["max_retries"]),
                )
            try:
                example = build_example(raw, expected_phase=args.phase, model=model)
            except ValueError:
                continue
            rows.append(example.to_dict())
            progress.update(task, completed=len(rows))

    write_jsonl(args.out, rows)
    if len(rows) < args.count:
        raise RuntimeError(f"only accepted {len(rows)} rows after {attempts} attempts")
    log_scalars(
        cfg,
        f"dataset/{args.phase}/generate",
        {"accepted": len(rows), "attempts": attempts, "rejected": attempts - len(rows)},
    )
    print(f"wrote {len(rows)} rows to {args.out} using {model}")


def _fake_row(phase: str, idx: int, model: str) -> dict:
    if phase == "syntax":
        code = f"def lean_syntax_sample_{idx} (n : Nat) : Nat := n + 1"
        instruction = "show a small Lean definition"
        response = f"Here is a small definition:\n{code}"
    else:
        code = f"theorem lean_proof_sample_{idx} (p : Prop) (h : p) : p := by\n  exact h"
        instruction = "prove this simple Lean theorem"
        response = code
    return {
        "phase": phase,
        "instruction": instruction,
        "response": response,
        "lean_code": code,
        "source": "dry_run",
        "generator_model": model,
        "quality_flags": ["dry_run"],
    }


if __name__ == "__main__":
    main()
