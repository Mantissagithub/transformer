#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
BASE_DIR="${BASE_DIR:-}"
OLD_DIR="${OLD_DIR:-}"
NEW_DIR="${NEW_DIR:-}"
OLD_CKPT="${OLD_CKPT:-}"
NEW_CKPT="${NEW_CKPT:-}"
OUT_DIR="${OUT_DIR:-}"

BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-100}"
GENERATION_SAMPLES="${GENERATION_SAMPLES:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-0}"
DEVICE="${DEVICE:-auto}"
PRECISION="${PRECISION:-fp32}"
DATA_LIMIT="${DATA_LIMIT:-0}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

usage() {
  cat <<'USAGE'
Evaluate old vs new MSA checkpoints from a training comparison run.

Typical:
  BASE_DIR=/tmp/msa_meetingbank_compare_<user>_<timestamp> ./scripts/eval_msa_versions.sh

Or explicit:
  OLD_DIR=/path/to/old NEW_DIR=/path/to/new ./scripts/eval_msa_versions.sh
  OLD_CKPT=/path/old.pt NEW_CKPT=/path/new.pt ./scripts/eval_msa_versions.sh

Defaults:
  BATCH_SIZE=8
  MAX_EVAL_BATCHES=100     # 0 = full validation
  GENERATION_SAMPLES=16    # 0 = skip generation overlap
  MAX_NEW_TOKENS=0         # 0 = model/data max_tgt_len
  DEVICE=auto              # auto|cpu|cuda
  PRECISION=fp32           # fp32|bf16|fp16
  DATA_LIMIT=0

Outputs:
  <BASE_DIR>/eval/old_metrics.json
  <BASE_DIR>/eval/new_metrics.json
  <BASE_DIR>/eval/summary.json
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "$BASE_DIR" ]]; then
  BASE_DIR="$(find /tmp -maxdepth 1 -type d -name "msa_meetingbank_compare_${USER}_*" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | awk '{print $2}')"
fi
if [[ -z "$BASE_DIR" ]]; then
  echo "BASE_DIR not provided and no /tmp/msa_meetingbank_compare_${USER}_* directory found" >&2
  exit 1
fi

OLD_DIR="${OLD_DIR:-$BASE_DIR/old}"
NEW_DIR="${NEW_DIR:-$BASE_DIR/new}"
OUT_DIR="${OUT_DIR:-$BASE_DIR/eval}"
mkdir -p "$OUT_DIR"

latest_ckpt() {
  local dir="$1"
  local prefix="$2"
  find "$dir/weights" -maxdepth 1 -type f -name "${prefix}*.pt" 2>/dev/null | sort | tail -1
}

OLD_CKPT="${OLD_CKPT:-$(latest_ckpt "$OLD_DIR" compare_msa_meetingbank_old)}"
NEW_CKPT="${NEW_CKPT:-$(latest_ckpt "$NEW_DIR" compare_msa_meetingbank_new)}"

if [[ ! -f "$OLD_CKPT" ]]; then
  echo "old checkpoint not found. Set OLD_CKPT=/path/to/checkpoint.pt" >&2
  exit 1
fi
if [[ ! -f "$NEW_CKPT" ]]; then
  echo "new checkpoint not found. Set NEW_CKPT=/path/to/checkpoint.pt" >&2
  exit 1
fi

eval_one() {
  local label="$1"
  local dir="$2"
  local ckpt="$3"
  local out="$OUT_DIR/${label}_metrics.json"

  echo "==> [$label] evaluating $ckpt"
  (
    cd "$dir"
    export HF_HUB_DISABLE_XET
    set -a
    [[ -f .env ]] && source .env
    set +a
    "$PYTHON_BIN" - "$ckpt" "$out" "$label" "$ROOT/tokenizers" <<'PY'
from __future__ import annotations

import json
import math
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tqdm import tqdm

import src  # noqa: F401
from src.model.builder import build_causal_lm, build_transformer
from src.registry import DATASET


ckpt_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
label = sys.argv[3]
tokenizer_dir = Path(sys.argv[4])

batch_size = int(__import__("os").environ.get("BATCH_SIZE", "8"))
max_eval_batches = int(__import__("os").environ.get("MAX_EVAL_BATCHES", "100"))
generation_samples = int(__import__("os").environ.get("GENERATION_SAMPLES", "16"))
max_new_tokens = int(__import__("os").environ.get("MAX_NEW_TOKENS", "0"))
device_arg = __import__("os").environ.get("DEVICE", "auto")
precision = __import__("os").environ.get("PRECISION", "fp32")
data_limit = int(__import__("os").environ.get("DATA_LIMIT", "0"))


def resolve_device() -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def autocast(device: torch.device):
    if precision == "fp32":
        from contextlib import nullcontext
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def build_cfg():
    with initialize_config_dir(config_dir=str(Path.cwd() / "configs"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiment=meeting_summarization_causal",
                "attention=msa",
                "logging=tensorboard",
                "+logging.prompt=false",
                f"training.batch_size={batch_size}",
                "+training.max_steps=0",
                f"+training.precision={precision}",
                f"data.limit={data_limit}",
                f"data.tokenizer_dir={tokenizer_dir}",
            ],
        )
    return cfg


def build_data(cfg):
    kwargs = OmegaConf.to_container(cfg.data, resolve=True)
    name = kwargs.pop("name")
    kwargs["batch_size"] = batch_size
    kwargs["val_batch_size"] = batch_size
    kwargs["num_workers"] = 0
    return DATASET.build(name, **kwargs)


def set_vocab_sizes(cfg, data):
    OmegaConf.set_struct(cfg, False)
    cfg.model.src_vocab_size = data["src_vocab_size"]
    cfg.model.tgt_vocab_size = data["tgt_vocab_size"]
    OmegaConf.set_struct(cfg, True)


def load_model(cfg, device):
    kind = str(cfg.model.get("kind", "encoder_decoder"))
    model = build_causal_lm(cfg) if kind == "causal_lm" else build_transformer(cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state["model_state_dict"] if "model_state_dict" in state else state
    model.load_state_dict(state_dict)
    return model.to(device).eval(), kind, state


def update_core(metrics, logits, labels, ignore_index):
    vocab = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab).float()
    flat_labels = labels.reshape(-1)
    mask = flat_labels != ignore_index
    count = int(mask.sum().item())
    if count == 0:
        return
    metrics["loss_sum"] += float(F.cross_entropy(flat_logits, flat_labels, ignore_index=ignore_index, reduction="sum").item())
    metrics["tokens"] += count
    scored = flat_logits[mask]
    target = flat_labels[mask]
    metrics["correct_1"] += int((scored.argmax(dim=-1) == target).sum().item())
    k = min(5, vocab)
    metrics["correct_5"] += int((scored.topk(k=k, dim=-1).indices == target[:, None]).any(dim=-1).sum().item())


def evaluate_core(model, data, cfg, kind, device):
    metrics = {"loss_sum": 0.0, "tokens": 0, "correct_1": 0, "correct_5": 0}
    latencies = []
    pad_id = int(data["pad_token_id"])
    start = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(data["val_loader"], desc=f"{label} core", unit="batch")):
            if max_eval_batches > 0 and batch_idx >= max_eval_batches:
                break
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.monotonic()
            with autocast(device):
                if kind == "causal_lm":
                    ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    logits = model(ids)
                else:
                    src = batch["encoder_input"].to(device)
                    tgt = batch["decoder_input"].to(device)
                    src_mask = batch["encoder_mask"].to(device)
                    tgt_mask = batch["decoder_mask"].to(device)
                    labels = batch["label"].to(device)
                    logits = model.project(model.decode(model.encode(src, src_mask), src_mask, tgt, tgt_mask))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies.append(time.monotonic() - t0)
            update_core(metrics, logits, labels, pad_id)

    elapsed = max(time.monotonic() - start, 1e-9)
    if metrics["tokens"]:
        loss = metrics["loss_sum"] / metrics["tokens"]
        ppl = math.inf if loss > 88 else math.exp(loss)
        tok_acc = metrics["correct_1"] / metrics["tokens"]
        top5 = metrics["correct_5"] / metrics["tokens"]
    else:
        loss = ppl = tok_acc = top5 = None
    return {
        "eval_loss": loss,
        "perplexity": ppl,
        "token_accuracy": tok_acc,
        "top_5_accuracy": top5,
        "non_pad_tokens": metrics["tokens"],
        "eval_tokens_per_sec": metrics["tokens"] / elapsed if metrics["tokens"] else None,
        "avg_forward_latency_ms": (sum(latencies) / len(latencies) * 1000) if latencies else None,
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(device) / 1_000_000 if device.type == "cuda" else None,
    }


def strip_after_eos(ids, eos_id):
    if eos_id in ids:
        return ids[: ids.index(eos_id)]
    return ids


def text_metrics(preds, refs):
    out = {"rouge1": None, "rouge2": None, "rougeL": None, "bleu": None}
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        for pred, ref in zip(preds, refs):
            scores = scorer.score(ref, pred)
            for key in totals:
                totals[key] += scores[key].fmeasure
        for key, val in totals.items():
            out[key] = val / max(len(preds), 1)
    except Exception as exc:
        out["rouge_error"] = str(exc)
    try:
        import sacrebleu
        out["bleu"] = float(sacrebleu.corpus_bleu(preds, [refs]).score)
    except Exception as exc:
        out["bleu_error"] = str(exc)
    return out


def evaluate_generation(model, data, cfg, kind, device):
    if generation_samples <= 0:
        return {}
    raw = load_dataset(str(cfg.data.hf_path), split="validation")
    sample_count = min(generation_samples, len(raw))
    tok = data["tgt_tokenizer"]
    max_src_len = int(cfg.data.get("max_src_len", 512))
    max_tgt_len = max_new_tokens if max_new_tokens > 0 else int(cfg.data.get("max_tgt_len", 256))
    bos_id = tok.token_to_id("[SOS]")
    sep_id = tok.token_to_id("[SEP]") or tok.token_to_id("[EOS]")
    eos_id = tok.token_to_id("[EOS]")
    preds, refs = [], []
    generated = 0
    start = time.monotonic()
    with torch.inference_mode():
        for row in tqdm(raw.select(range(sample_count)), desc=f"{label} gen", unit="sample"):
            src_ids = tok.encode(row["transcript"]).ids[: max_src_len - 2]
            prompt = torch.tensor([[bos_id, *src_ids, sep_id]], dtype=torch.long, device=device)
            with autocast(device):
                out = model.generate(prompt, max_new_tokens=max_tgt_len, temperature=0.0, eos_id=eos_id)
            new_ids = out[0, prompt.shape[1] :].detach().cpu().tolist()
            generated += len(new_ids)
            preds.append(tok.decode(strip_after_eos(new_ids, eos_id), skip_special_tokens=True))
            refs.append(row["summary"])
    elapsed = max(time.monotonic() - start, 1e-9)
    result = text_metrics(preds, refs)
    result["generation_samples"] = sample_count
    result["generation_tokens_per_sec"] = generated / elapsed if generated else None
    return result


device = resolve_device()
cfg = build_cfg()
data = build_data(cfg)
set_vocab_sizes(cfg, data)
model, kind, state = load_model(cfg, device)

result = {
    "label": label,
    "checkpoint": str(ckpt_path),
    "checkpoint_epoch": state.get("epoch"),
    "checkpoint_global_step": state.get("global_step"),
    "device": str(device),
    "precision": precision,
    "parameters": sum(p.numel() for p in model.parameters()),
}
result.update(evaluate_core(model, data, cfg, kind, device))
result.update(evaluate_generation(model, data, cfg, kind, device))
out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
PY
  )
}

echo "Evaluating MSA checkpoints"
echo "  base dir: $BASE_DIR"
echo "  old dir: $OLD_DIR"
echo "  new dir: $NEW_DIR"
echo "  old ckpt: $OLD_CKPT"
echo "  new ckpt: $NEW_CKPT"
echo "  output dir: $OUT_DIR"

export BATCH_SIZE MAX_EVAL_BATCHES GENERATION_SAMPLES MAX_NEW_TOKENS DEVICE PRECISION DATA_LIMIT
eval_one old "$OLD_DIR" "$OLD_CKPT"
eval_one new "$NEW_DIR" "$NEW_CKPT"

"$PYTHON_BIN" - "$OUT_DIR/old_metrics.json" "$OUT_DIR/new_metrics.json" "$OUT_DIR/summary.json" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

old = json.loads(Path(sys.argv[1]).read_text())
new = json.loads(Path(sys.argv[2]).read_text())
summary_path = Path(sys.argv[3])

keys = [
    "eval_loss",
    "perplexity",
    "token_accuracy",
    "top_5_accuracy",
    "eval_tokens_per_sec",
    "avg_forward_latency_ms",
    "rouge1",
    "rouge2",
    "rougeL",
    "bleu",
    "generation_tokens_per_sec",
]
summary = {"old": old, "new": new, "delta_new_minus_old": {}}
for key in keys:
    ov = old.get(key)
    nv = new.get(key)
    summary["delta_new_minus_old"][key] = (nv - ov) if isinstance(ov, (int, float)) and isinstance(nv, (int, float)) else None

summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

print("\nmetric,old,new,delta_new_minus_old")
for key in keys:
    print(f"{key},{old.get(key)},{new.get(key)},{summary['delta_new_minus_old'][key]}")
print(f"\nwrote {summary_path}")
PY
