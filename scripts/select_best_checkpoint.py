from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from benchmark_hf_collection import (
    _build_data,
    _clear_cuda,
    _evaluate_core,
    _prepare_cfg,
    _resolve_device,
    _set_vocab_sizes,
)
from src.model.builder import build_causal_lm, build_transformer


def main() -> None:
    args = _parse_args()
    cfg = OmegaConf.load(args.config)
    _prepare_cfg(cfg, Path(args.tokenizer_dir), args)
    data = _build_data(cfg, args)
    _set_vocab_sizes(cfg, data)

    model_kind = str(cfg.model.get("kind", "encoder_decoder"))
    device = _resolve_device(args.device)
    model = build_causal_lm(cfg) if model_kind == "causal_lm" else build_transformer(cfg)
    model = model.to(device)
    model.eval()

    results = []
    for checkpoint in [Path(path) for path in args.checkpoint]:
        started = time.monotonic()
        state = torch.load(checkpoint, map_location="cpu")
        state_dict = state["model_state_dict"] if "model_state_dict" in state else state
        model.load_state_dict(state_dict)

        core, performance = _evaluate_core(model, data, cfg, model_kind, device, args)
        result = {
            "checkpoint": str(checkpoint),
            "epoch": state.get("epoch"),
            "global_step": state.get("global_step"),
            **core,
            **performance,
            "wall_time_sec": time.monotonic() - started,
        }
        results.append(result)
        print(
            f"{checkpoint.name}: val_loss={result['eval_loss']:.6f}, "
            f"perplexity={result['perplexity']:.4f}, "
            f"token_accuracy={result['token_accuracy']:.4%}"
        )
        _clear_cuda()

    best = min(results, key=lambda result: result["eval_loss"])
    payload = {"selection_metric": "eval_loss", "best": best, "checkpoints": results}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"best checkpoint: {best['checkpoint']} ({best['eval_loss']:.6f})")
    print(f"wrote {output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a checkpoint using full validation loss.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer-dir", default="tokenizers")
    parser.add_argument("--dataset-hf-path", default="huuuyeah/meetingbank")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    return parser.parse_args()


if __name__ == "__main__":
    main()
