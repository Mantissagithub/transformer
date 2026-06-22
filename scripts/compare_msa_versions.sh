#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
if [[ -f "$ROOT/.env" ]]; then
  set -a
  source "$ROOT/.env"
  set +a
fi

resolve_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s\n' "$PYTHON_BIN"
    return
  fi

  local candidates=()
  candidates+=("$(command -v python || true)")
  candidates+=("$(command -v python3 || true)")
  candidates+=("/home/pradheep/.local/share/mise/installs/python/3.11.14/bin/python")

  local py
  for py in "${candidates[@]}"; do
    [[ -n "$py" && -x "$py" ]] || continue
    if "$py" -c 'import hydra' >/dev/null 2>&1; then
      printf '%s\n' "$py"
      return
    fi
  done

  echo "could not find a Python interpreter with hydra installed; set PYTHON_BIN=/path/to/python" >&2
  exit 1
}

PYTHON_BIN="$(resolve_python)"
OLD_REF="${OLD_REF:-HEAD~1}"
NEW_REF="${NEW_REF:-HEAD}"
BASE_DIR="${BASE_DIR:-/tmp/msa_meetingbank_compare_${USER}_$(date +%Y%m%d_%H%M%S)}"

# Real MeetingBank summarization training settings. Set MAX_STEPS=0 to let the
# repo config run normally for NUM_EPOCHS.
NUM_EPOCHS="${NUM_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-200}"
DATA_LIMIT="${DATA_LIMIT:-0}"
LR="${LR:-1e-4}"
PRECISION="${PRECISION:-fp32}"
TUI="${TUI:-true}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
KEEP_WORKTREES="${KEEP_WORKTREES:-true}"
HF_REPO_NAMESPACE="${HF_REPO_NAMESPACE:-${HF_USERNAME:-}}"
HF_REPO_PREFIX="${HF_REPO_PREFIX:-compare-msa-meetingbank}"

usage() {
  cat <<'USAGE'
Compare old vs new MSA on the actual MeetingBank summarization training path.

Defaults:
  OLD_REF=HEAD~1
  NEW_REF=HEAD
  NUM_EPOCHS=1
  BATCH_SIZE=8
  MAX_STEPS=200
  DATA_LIMIT=0
  LR=1e-4
  PRECISION=fp32
  TUI=true
  HF_HUB_DISABLE_XET=1
  KEEP_WORKTREES=true
  HF_REPO_NAMESPACE=<from HF_USERNAME in .env>
  HF_REPO_PREFIX=compare-msa-meetingbank

Examples:
  ./scripts/compare_msa_versions.sh
  MAX_STEPS=1000 BATCH_SIZE=4 ./scripts/compare_msa_versions.sh
  MAX_STEPS=0 NUM_EPOCHS=20 ./scripts/compare_msa_versions.sh
  TUI=false MAX_STEPS=200 ./scripts/compare_msa_versions.sh
  OLD_REF=b56ecd4 NEW_REF=e1a3421 ./scripts/compare_msa_versions.sh
  HF_REPO_PREFIX=msa-meetingbank-paper-compare ./scripts/compare_msa_versions.sh

Outputs:
  /tmp/msa_meetingbank_compare_<user>_<timestamp>/
    old_train.log and new_train.log when TUI=false
    old/ and new/ detached git worktrees
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "$HF_REPO_NAMESPACE" ]]; then
  echo "HF_REPO_NAMESPACE is empty; set HF_USERNAME in .env or export HF_REPO_NAMESPACE." >&2
  exit 1
fi

mkdir -p "$BASE_DIR"
OLD_DIR="$BASE_DIR/old"
NEW_DIR="$BASE_DIR/new"

cleanup() {
  if [[ "$KEEP_WORKTREES" != "true" ]]; then
    git -C "$ROOT" worktree remove --force "$OLD_DIR" >/dev/null 2>&1 || true
    git -C "$ROOT" worktree remove --force "$NEW_DIR" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

git -C "$ROOT" worktree add --detach "$OLD_DIR" "$OLD_REF" >/dev/null
git -C "$ROOT" worktree add --detach "$NEW_DIR" "$NEW_REF" >/dev/null

if [[ -f "$ROOT/.env" ]]; then
  cp "$ROOT/.env" "$OLD_DIR/.env"
  cp "$ROOT/.env" "$NEW_DIR/.env"
  chmod 600 "$OLD_DIR/.env" "$NEW_DIR/.env"
fi

disable_logging_prompt() {
  local dir="$1"
  "$PYTHON_BIN" - "$dir" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1]) / "src/training/logging.py"
text = path.read_text()
needle = '    backend = cfg.logging.get("backend", "tensorboard")\n'
insert = '    if cfg.logging.get("prompt", True) is False:\n        return\n\n'
if insert not in text:
    if needle not in text:
        raise SystemExit(f"could not patch logging prompt in {path}")
    text = text.replace(needle, insert + needle, 1)
    path.write_text(text)
PY
}

disable_logging_prompt "$OLD_DIR"
disable_logging_prompt "$NEW_DIR"

# The msa-old/new refs predate the tui.event() closed-file fix, so the post-
# training hf-push event crashes on a closed log file. Patch it into both
# worktrees (orthogonal to the attention comparison).
apply_tui_event_fix() {
  local dir="$1"
  "$PYTHON_BIN" - "$dir" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1]) / "src/training/tui.py"
text = path.read_text()
needle = '        self._logf.write(line + "\\n")\n        self._logf.flush()\n'
insert = (
    '        if self._logf.closed:\n'
    '            print(line, file=self._orig_stdout, flush=True)\n'
    '            return\n'
    '        self._logf.write(line + "\\n")\n'
    '        self._logf.flush()\n'
)
if 'self._logf.closed' not in text:
    if needle not in text:
        raise SystemExit(f"could not patch tui.event in {path}")
    text = text.replace(needle, insert, 1)
    path.write_text(text)
PY
}

apply_tui_event_fix "$OLD_DIR"
apply_tui_event_fix "$NEW_DIR"

train_ref() {
  local dir="$1"
  local label="$2"
  local log="$BASE_DIR/${label}_train.log"
  local repo_id="${HF_REPO_NAMESPACE}/${HF_REPO_PREFIX}-${label}"

  echo "==> [$label] MeetingBank causal summarization training"
  echo "    dir: $dir"
  echo "    hf repo: $repo_id"
  echo "    HF_HUB_DISABLE_XET: $HF_HUB_DISABLE_XET"
  if [[ "$TUI" == "true" ]]; then
    echo "    tui: enabled"
  else
    echo "    log: $log"
  fi

  local cmd=(
    "$PYTHON_BIN" -m src.cli.train
    +experiment=meeting_summarization_causal
    attention=msa
    experiment_name="compare_msa_meetingbank_${label}"
    training.ckpt_basename="compare_msa_meetingbank_${label}"
    training.num_epochs="$NUM_EPOCHS"
    training.batch_size="$BATCH_SIZE"
    +training.max_steps="$MAX_STEPS"
    training.lr="$LR"
    +training.precision="$PRECISION"
    training.save_every_n_epochs=0
    training.tui="$TUI"
    training.hf.push=true
    training.hf.repo_id="$repo_id"
    training.hf.commit_message="compare msa meetingbank ${label}"
    data.limit="$DATA_LIMIT"
    data.tokenizer_dir="$ROOT/tokenizers"
    logging=tensorboard
    +logging.prompt=false
  )

  if [[ "$TUI" == "true" ]]; then
    (
      cd "$dir"
      export HF_HUB_DISABLE_XET
      set -a
      [[ -f .env ]] && source .env
      set +a
      set -x
      "${cmd[@]}"
    )
    return
  fi

  (
    cd "$dir"
    export HF_HUB_DISABLE_XET
    set -a
    [[ -f .env ]] && source .env
    set +a
    set -x
    "${cmd[@]}"
  ) 2>&1 | tee "$log"
}

echo "MeetingBank MSA training comparison"
echo "  old ref: $OLD_REF -> $OLD_DIR"
echo "  new ref: $NEW_REF -> $NEW_DIR"
echo "  num epochs: $NUM_EPOCHS"
echo "  batch size: $BATCH_SIZE"
echo "  max steps: $MAX_STEPS"
echo "  data limit: $DATA_LIMIT"
echo "  tui: $TUI"
echo "  python: $PYTHON_BIN"
echo "  HF_HUB_DISABLE_XET: $HF_HUB_DISABLE_XET"
echo "  HF repo old: ${HF_REPO_NAMESPACE}/${HF_REPO_PREFIX}-old"
echo "  HF repo new: ${HF_REPO_NAMESPACE}/${HF_REPO_PREFIX}-new"
echo "  keep worktrees: $KEEP_WORKTREES"
echo "  base dir: $BASE_DIR"

train_ref "$OLD_DIR" old
train_ref "$NEW_DIR" new

if [[ "$TUI" != "true" ]]; then
  echo
  echo "Logs:"
  echo "  old: $BASE_DIR/old_train.log"
  echo "  new: $BASE_DIR/new_train.log"
  echo
  echo "Quick loss tail:"
  grep -h "loss=" "$BASE_DIR/old_train.log" | tail -n 5 | sed 's/^/[old] /' || true
  grep -h "loss=" "$BASE_DIR/new_train.log" | tail -n 5 | sed 's/^/[new] /' || true
fi

if [[ "$KEEP_WORKTREES" == "true" ]]; then
  echo
  echo "Kept worktrees and checkpoints under:"
  echo "  $BASE_DIR"
  echo "Run eval with:"
  echo "  BASE_DIR=$BASE_DIR ./scripts/eval_msa_versions.sh"
fi
