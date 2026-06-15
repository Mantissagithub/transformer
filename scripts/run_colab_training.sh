#!/usr/bin/env bash
set -euo pipefail

SESSION="${COLAB_SESSION:-lean-starcoder-sft}"
REMOTE_ROOT="/content/lean-starcoder-sft"
ARCHIVE="/tmp/lean-starcoder-sft.tar.gz"

command -v colab >/dev/null
colab --auth=oauth2 sessions >/dev/null

tar \
  --exclude .git \
  --exclude .venv \
  --exclude data/generated \
  --exclude data/processed \
  --exclude outputs \
  --exclude artifacts \
  -czf "$ARCHIVE" .

colab --auth=oauth2 new -s "$SESSION" --gpu T4 || true
colab --auth=oauth2 upload -s "$SESSION" "$ARCHIVE" /content/lean-starcoder-sft.tar.gz

cat > /tmp/lean_starcoder_colab_bootstrap.py <<'PY'
import os
import subprocess
from pathlib import Path

root = Path("/content/lean-starcoder-sft")
root.mkdir(parents=True, exist_ok=True)
subprocess.run(["tar", "-xzf", "/content/lean-starcoder-sft.tar.gz", "-C", str(root)], check=True)
env = os.environ.copy()
env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
subprocess.run(["python", "-m", "pip", "install", "-U", "pip"], check=True, env=env)
subprocess.run(["python", "-m", "pip", "install", "-e", ".[quant]"], cwd=root, check=True, env=env)
subprocess.run(["bash", "scripts/run_local_pipeline.sh"], cwd=root, check=True, env=env)
PY

colab --auth=oauth2 exec -s "$SESSION" -f /tmp/lean_starcoder_colab_bootstrap.py
colab --auth=oauth2 log -s "$SESSION" -o artifacts/colab_log.md || true
colab --auth=oauth2 download -s "$SESSION" /content/lean-starcoder-sft/benchmarks/lean/report.md benchmarks/lean/report.md || true
colab --auth=oauth2 download -s "$SESSION" /content/lean-starcoder-sft/runs/lean-starcoder-sft artifacts/tensorboard-runs || true
colab --auth=oauth2 stop -s "$SESSION"
