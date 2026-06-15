from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from datasets import load_dataset

from .config import load_config, require_env
from .schema import format_for_sft
from .tb import log_scalars


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a limited Lean SFT benchmark.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-repo", default=None)
    parser.add_argument("--out", default="benchmarks/lean/report.md")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    token = require_env("HF_TOKEN")
    ds = load_dataset(args.dataset_repo or cfg["dataset_repo"], split="test", token=token)
    rows = [row for row in ds if row["phase"] == "proof"][: args.limit]
    lean_path = shutil.which("lean")
    target_compile_ok = 0
    target_compile_fail = 0
    for idx, row in enumerate(rows):
        if lean_path and _check_lean(lean_path, row["lean_code"], idx):
            target_compile_ok += 1
        elif lean_path:
            target_compile_fail += 1

    generated = []
    generated_compile_ok = 0
    generated_compile_fail = 0
    if args.generate:
        generated = _generate(args.model, rows, token, cfg["base_model"])
        for idx, code in enumerate(generated):
            if lean_path and _check_lean(lean_path, code, 10_000 + idx):
                generated_compile_ok += 1
            elif lean_path:
                generated_compile_fail += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Lean SFT Benchmark",
        "",
        f"- Model: `{args.model}`",
        f"- Dataset: `{args.dataset_repo or cfg['dataset_repo']}`",
        f"- Proof test rows sampled: `{len(rows)}`",
        f"- Lean executable: `{lean_path or 'not found'}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| target_lean_compile_ok | {target_compile_ok} |",
        f"| target_lean_compile_fail | {target_compile_fail} |",
        f"| generated_rows | {len(generated)} |",
        f"| generated_lean_compile_ok | {generated_compile_ok} |",
        f"| generated_lean_compile_fail | {generated_compile_fail} |",
        f"| text_only_rows | {len(rows) if not lean_path else 0} |",
        "",
        "This report treats Lean compilation as the real correctness signal. If Lean is not installed, only dataset-format coverage is reported.",
    ]
    out.write_text("\n".join(lines) + "\n")
    log_scalars(
        cfg,
        "benchmark",
        {
            "target_lean_compile_ok": target_compile_ok,
            "target_lean_compile_fail": target_compile_fail,
            "generated_rows": len(generated),
            "generated_lean_compile_ok": generated_compile_ok,
            "generated_lean_compile_fail": generated_compile_fail,
            "text_only_rows": len(rows) if not lean_path else 0,
        },
    )
    print(f"wrote {out}")


def _check_lean(lean_path: str, code: str, idx: int) -> bool:
    path = Path("/tmp") / f"lean_sft_bench_{idx}.lean"
    path.write_text(code + "\n")
    proc = subprocess.run([lean_path, str(path)], text=True, capture_output=True, timeout=20)
    return proc.returncode == 0


def _generate(model_name: str, rows: list[dict], token: str, base_model: str) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    outputs = []
    for row in rows:
        prompt = format_for_sft({**row, "response": ""})
        batch = tokenizer(prompt, return_tensors="pt").to(model.device)
        generated = model.generate(**batch, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(generated[0][batch["input_ids"].shape[1] :], skip_special_tokens=True)
        outputs.append(_extract_lean(text))
    return outputs


def _extract_lean(text: str) -> str:
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            body = parts[1]
            return body.removeprefix("lean").strip()
    return text.strip()


if __name__ == "__main__":
    main()
