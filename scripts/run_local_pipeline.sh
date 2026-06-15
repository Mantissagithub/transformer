#!/usr/bin/env bash
set -euo pipefail

SYNTAX_COUNT="${SYNTAX_COUNT:-8000}"
PROOF_COUNT="${PROOF_COUNT:-8000}"

python -m lean_sft.generate_dataset --phase syntax --count "$SYNTAX_COUNT" --out data/generated/syntax.jsonl
python -m lean_sft.generate_dataset --phase proof --count "$PROOF_COUNT" --out data/generated/proof.jsonl
python -m lean_sft.validate_dataset data/generated/syntax.jsonl --out data/processed/syntax.jsonl
python -m lean_sft.validate_dataset data/generated/proof.jsonl --out data/processed/proof.jsonl
python -m lean_sft.push_dataset --syntax data/processed/syntax.jsonl --proof data/processed/proof.jsonl

python -m lean_sft.train \
  --phase syntax \
  --output-dir outputs/phase1 \
  --push-repo "${HF_PHASE1_REPO:-Pradheep1647/starcoderbase-1b-lean-syntax-lora}"

python -m lean_sft.train \
  --phase proof \
  --base-or-adapter "${HF_PHASE1_REPO:-Pradheep1647/starcoderbase-1b-lean-syntax-lora}" \
  --output-dir outputs/phase2 \
  --push-repo "${HF_PHASE2_REPO:-Pradheep1647/starcoderbase-1b-lean-proofs-lora}"

python -m lean_sft.benchmark \
  --model "${HF_MERGED_REPO:-Pradheep1647/starcoderbase-1b-lean-sft}" \
  --out benchmarks/lean/report.md
