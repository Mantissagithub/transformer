from __future__ import annotations

import argparse
import csv
from html import escape
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml
from huggingface_hub import HfApi, create_repo
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


OWNER = "Pradheep1647"
COLLECTION_TITLE = "transformer-lab"
COLLECTION_SLUG = "Pradheep1647/transformer-lab-6a07fe3185f5728e217997e0"
EXTRA_REPOS = [
    "Pradheep1647/run_sliding_gqa-meetingbank-bs8-e20-fp32-19",
    "Pradheep1647/run_hca-meetingbank-bs8-e20-fp32-19",
    "Pradheep1647/run_csa-meetingbank-bs8-e20-fp32-19",
]
LOSS_TAG = "train/loss"


@dataclass(frozen=True)
class ModelTarget:
    repo_id: str
    run_name: str
    config_path: Path
    event_file: Path
    checkpoint_file: str


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    token = _hf_token(root)
    api = HfApi(token=token)

    collection_slug, collection_repos = _collection_repos(api)
    repo_ids = args.repo_id or _ordered_unique([*collection_repos, *EXTRA_REPOS])
    publishable_targets = []
    skipped_repo_ids = []
    for repo_id in repo_ids:
        checkpoint_file = _repo_checkpoint_file(api, repo_id)
        if checkpoint_file:
            publishable_targets.append((repo_id, checkpoint_file))
        else:
            skipped_repo_ids.append(repo_id)

    for repo_id in skipped_repo_ids:
        print(f"skipping {repo_id}: missing checkpoint .pt file")
        if not args.dry_run:
            _remove_from_collection(api, collection_slug, repo_id)

    repo_ids = [repo_id for repo_id, _ in publishable_targets]
    targets = [_resolve_target(root, repo_id, checkpoint_file) for repo_id, checkpoint_file in publishable_targets]

    if args.dry_run:
        for target in targets:
            print(f"{target.repo_id}")
            print(f"  config: {target.config_path}")
            print(f"  events: {target.event_file}")
        return

    metrics = _load_metrics(Path(args.metrics_file)) if args.metrics_file else {}
    with tempfile.TemporaryDirectory(prefix="transformer-hf-publish-") as tmp:
        tmp_root = Path(tmp)
        available_table = _available_models_table(targets)
        template = (root / "hf_readme.md").read_text()
        for target in targets:
            out_dir = tmp_root / target.repo_id.replace("/", "__")
            out_dir.mkdir(parents=True, exist_ok=True)
            artifacts = _build_artifacts(
                root,
                target,
                targets,
                available_table,
                template,
                out_dir,
                metrics.get(target.repo_id),
            )
            _upload_artifacts(api, token, target.repo_id, artifacts)

        _ensure_collection_membership(api, collection_slug, repo_ids)

    print(f"published {len(targets)} model repos")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish production HF model-card artifacts for transformer-lab repos."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve targets and local artifact sources without uploading.",
    )
    parser.add_argument(
        "--repo-id",
        action="append",
        default=[],
        help="Publish a specific repo; repeatable.",
    )
    parser.add_argument(
        "--metrics-file",
        help="Optional benchmark results.jsonl used to populate evaluation metrics.",
    )
    return parser.parse_args()


def _hf_token(root: Path) -> str:
    token = os.environ.get("HF_TOKEN")
    env_path = root / ".env"
    if not token and env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip()
                break
    if not token:
        raise RuntimeError("HF_TOKEN is required in the environment or .env")
    return token


def _collection_repos(api: HfApi) -> tuple[str, list[str]]:
    try:
        collection = api.get_collection(COLLECTION_SLUG)
        repos = [
            item.item_id
            for item in collection.items
            if item.item_type == "model"
        ]
        return collection.slug, repos
    except Exception:
        pass

    for collection in api.list_collections(owner=OWNER):
        if collection.title == COLLECTION_TITLE:
            repos = [
                item.item_id
                for item in collection.items
                if item.item_type == "model"
            ]
            return collection.slug, repos
    return COLLECTION_SLUG, []


def _repo_checkpoint_file(api: HfApi, repo_id: str) -> str | None:
    try:
        files = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    except Exception:
        return None

    checkpoints = sorted(path for path in files if path.endswith(".pt"))
    if not checkpoints:
        return None
    if len(checkpoints) == 1:
        return checkpoints[0]

    epoch_suffix = re.compile(r"(\d+)\.pt$")
    ranked = []
    for path in checkpoints:
        match = epoch_suffix.search(path)
        ranked.append((int(match.group(1)) if match else -1, path))
    ranked.sort()
    return ranked[-1][1]


def _remove_from_collection(api: HfApi, collection_slug: str, repo_id: str) -> None:
    collection = api.get_collection(collection_slug)
    for item in collection.items:
        if item.item_type == "model" and item.item_id == repo_id:
            api.delete_collection_item(
                collection_slug=collection_slug,
                item_object_id=item.item_object_id,
                missing_ok=True,
            )
            print(f"removed incomplete repo from collection: {repo_id}")
            return


def _resolve_target(root: Path, repo_id: str, checkpoint_file: str) -> ModelTarget:
    run_name = _run_name(repo_id)
    config_path = _latest_config(root / "outputs" / run_name)
    event_file = _event_file(root / "runs", run_name)
    return ModelTarget(
        repo_id=repo_id,
        run_name=run_name,
        config_path=config_path,
        event_file=event_file,
        checkpoint_file=checkpoint_file,
    )


def _run_name(repo_id: str) -> str:
    name = repo_id.split("/", 1)[1]
    match = re.match(r"(.+)-meetingbank-bs\d+-(?:e|s)\d+-.+-\d+$", name)
    if not match:
        raise RuntimeError(f"cannot infer run name from repo id: {repo_id}")
    return match.group(1)


def _latest_config(output_dir: Path) -> Path:
    configs = sorted(output_dir.glob("*/.hydra/config.yaml"))
    if not configs:
        raise RuntimeError(f"no Hydra config found under {output_dir}")
    return max(configs, key=lambda path: path.stat().st_mtime)


def _event_file(runs_dir: Path, run_name: str) -> Path:
    direct = runs_dir / run_name
    direct_events = list(direct.glob("events.out.tfevents*")) if direct.exists() else []
    if direct_events:
        return max(direct_events, key=lambda path: path.stat().st_mtime)

    candidates = [
        event
        for path in runs_dir.glob(f"{run_name}*")
        if path.is_dir()
        for event in path.glob("events.out.tfevents*")
    ]
    if not candidates:
        raise RuntimeError(f"no TensorBoard event file found for {run_name}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _build_artifacts(
    root: Path,
    target: ModelTarget,
    targets: list[ModelTarget],
    available_table: str,
    template: str,
    out_dir: Path,
    metrics: dict | None,
) -> list[Path]:
    cfg = yaml.safe_load(target.config_path.read_text())
    model_meta = _model_meta(target.repo_id, cfg)

    config_json = out_dir / "config.json"
    config_json.write_text(json.dumps(cfg, indent=2, sort_keys=False) + "\n")

    architecture_png = out_dir / "architecture.png"
    _write_architecture_diagram(architecture_png, cfg, model_meta)

    loss_rows = _loss_rows(target.event_file)
    loss_csv = out_dir / "loss_curve.csv"
    _write_loss_csv(loss_csv, loss_rows)
    _write_loss_svg(out_dir / "loss_curve.svg", loss_rows, model_meta["attention"])
    model_meta["training_steps"] = f"{loss_rows[-1][1]:,}"
    model_meta["logged_training_time"] = _format_duration(loss_rows[-1][0] - loss_rows[0][0])

    readme = template.format(
        available_models_table=available_table,
        checkpoint_file=target.checkpoint_file,
        model_title=_model_title(target.run_name),
        repo_id=target.repo_id,
        architecture_file=architecture_png.name,
        evaluation_table=_evaluation_table(metrics),
        model_index=_model_index(target, metrics),
        tokenizer_files=_tokenizer_files_table(model_meta["model_kind"]),
        **model_meta,
    )
    (out_dir / "README.md").write_text(readme)

    tokenizer_artifacts = _copy_tokenizers(root, out_dir, cfg)
    return [
        out_dir / "README.md",
        config_json,
        architecture_png,
        loss_csv,
        out_dir / "loss_curve.svg",
        *tokenizer_artifacts,
    ]


def _model_meta(repo_id: str, cfg: dict) -> dict[str, str]:
    attention = str(cfg["attention"]["name"])
    training = cfg["training"]
    model = cfg["model"]
    data = cfg["data"]
    return {
        "attention": attention,
        "dataset": str(data.get("name", "unknown")),
        "n_layers": str(model.get("n_layers", "unknown")),
        "d_model": str(model.get("d_model", "unknown")),
        "n_heads": str(cfg["attention"].get("n_heads", "unknown")),
        "batch_size": str(training.get("batch_size", "unknown")),
        "effective_batch_size": str(
            int(training.get("batch_size", 1))
            * int(training.get("gradient_accumulation_steps", 1))
        ),
        "num_epochs": str(training.get("num_epochs", "unknown")),
        "precision": str(training.get("precision", "fp32")),
        "model_kind": str(model.get("kind", "encoder_decoder")),
        "builder_function": "build_causal_lm"
        if str(model.get("kind", "encoder_decoder")) == "causal_lm"
        else "build_transformer",
    }


def _load_metrics(path: Path) -> dict[str, dict]:
    metrics = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "ok" and row.get("repo_id"):
            metrics[row["repo_id"]] = row
    return metrics


def _evaluation_table(metrics: dict | None) -> str:
    if not metrics:
        return "Evaluation metrics have not been attached yet."

    rows = ["| Metric | Value |", "|---|---:|"]
    fields = [
        ("Validation loss", "eval_loss", 4),
        ("Perplexity", "perplexity", 4),
        ("Token accuracy", "token_accuracy", 4),
        ("Top-5 accuracy", "top_5_accuracy", 4),
        ("ROUGE-1", "rouge1", 4),
        ("ROUGE-2", "rouge2", 4),
        ("ROUGE-L", "rougeL", 4),
        ("BLEU", "bleu", 2),
        ("BERTScore F1", "bertscore_f1", 4),
        ("Evaluation tokens/s", "eval_tokens_per_sec", 1),
        ("Generation tokens/s", "generation_tokens_per_sec", 1),
        ("Forward latency (ms)", "avg_forward_latency_ms", 2),
        ("Peak CUDA memory (MB)", "peak_cuda_memory_mb", 1),
    ]
    for label, key, digits in fields:
        value = metrics.get(key)
        if value is not None:
            rows.append(f"| {label} | {float(value):.{digits}f} |")
    return "\n".join(rows)


def _model_index(target: ModelTarget, metrics: dict | None) -> str:
    if not metrics:
        return ""

    fields = [
        ("loss", "Validation loss", "eval_loss"),
        ("perplexity", "Perplexity", "perplexity"),
        ("accuracy", "Token accuracy", "token_accuracy"),
        ("accuracy", "Top-5 accuracy", "top_5_accuracy"),
        ("rouge", "ROUGE-1", "rouge1"),
        ("rouge", "ROUGE-2", "rouge2"),
        ("rouge", "ROUGE-L", "rougeL"),
        ("bleu", "BLEU", "bleu"),
        ("bertscore", "BERTScore F1", "bertscore_f1"),
    ]
    lines = [
        "model-index:",
        f"- name: {_model_title(target.run_name)}",
        "  results:",
        "  - task:",
        "      type: summarization",
        "      name: Meeting summarization",
        "    dataset:",
        "      type: huuuyeah/meetingbank",
        "      name: MeetingBank",
        "      split: validation",
        "    metrics:",
    ]
    for metric_type, name, key in fields:
        value = metrics.get(key)
        if value is None:
            continue
        lines.extend(
            [
                f"    - type: {metric_type}",
                f"      name: {name}",
                f"      value: {float(value)}",
            ]
        )
    return "\n".join(lines)


def _format_duration(seconds: float) -> str:
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"


def _tokenizer_files_table(model_kind: str) -> str:
    if model_kind == "causal_lm":
        return "\n".join(
            [
                "| `tokenizer.json` | Unified MeetingBank transcript and summary tokenizer. |",
                "| `causal_tokenizer.json` | Explicit alias of the unified causal tokenizer. |",
            ]
        )
    return "\n".join(
        [
            "| `tokenizer.json` | MeetingBank transcript tokenizer alias for source inputs. |",
            "| `transcript_tokenizer.json` | Explicit MeetingBank transcript tokenizer. |",
            "| `summary_tokenizer.json` | MeetingBank summary tokenizer for target text. |",
        ]
    )


def _model_title(run_name: str) -> str:
    return run_name.replace("_", " ").title()


def _available_models_table(targets: Iterable[ModelTarget]) -> str:
    rows = ["| Variant | Repository |", "|---|---|"]
    for target in targets:
        variant = target.run_name.replace("run_", "")
        rows.append(f"| `{variant}` | [`{target.repo_id}`](https://huggingface.co/{target.repo_id}) |")
    return "\n".join(rows)


def _loss_rows(event_dir: Path) -> list[tuple[float, int, float]]:
    accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    if LOSS_TAG not in tags:
        raise RuntimeError(f"{LOSS_TAG} not found in {event_dir}; available tags: {tags}")

    rows = [
        (float(event.wall_time), int(event.step), float(event.value))
        for event in accumulator.Scalars(LOSS_TAG)
    ]
    if not rows:
        raise RuntimeError(f"{LOSS_TAG} is empty in {event_dir}")
    return rows


def _write_loss_csv(path: Path, rows: list[tuple[float, int, float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wall_time", "step", "loss"])
        writer.writerows(rows)


def _write_loss_svg(path: Path, rows: list[tuple[float, int, float]], title: str) -> None:
    width, height = 960, 540
    margin_left, margin_top, margin_right, margin_bottom = 72, 48, 28, 72
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    steps = [row[1] for row in rows]
    losses = [row[2] for row in rows if math.isfinite(row[2])]
    min_step, max_step = min(steps), max(steps)
    min_loss, max_loss = min(losses), max(losses)
    if max_step <= min_step:
        max_step = min_step + 1
    if max_loss <= min_loss:
        max_loss = min_loss + 1.0

    max_points = 900
    if len(rows) > max_points:
        stride = max(1, len(rows) // max_points)
        sampled = rows[::stride]
    else:
        sampled = rows

    points = []
    for _wall_time, step, loss in sampled:
        if not math.isfinite(loss):
            continue
        x = margin_left + (step - min_step) / (max_step - min_step) * plot_w
        y = margin_top + (max_loss - loss) / (max_loss - min_loss) * plot_h
        points.append(f"{x:.2f},{y:.2f}")

    polyline = " ".join(points)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Training loss curve">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{margin_left}" y="30" font-family="Arial, sans-serif" font-size="20" fill="#111111">{title} training loss</text>
  <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#222222" stroke-width="1.5"/>
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#222222" stroke-width="1.5"/>
  <text x="{width / 2:.0f}" y="{height - 22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#333333">step</text>
  <text x="24" y="{height / 2:.0f}" text-anchor="middle" transform="rotate(-90 24 {height / 2:.0f})" font-family="Arial, sans-serif" font-size="14" fill="#333333">loss</text>
  <text x="{margin_left}" y="{height - margin_bottom + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#555555">{min_step:,}</text>
  <text x="{width - margin_right}" y="{height - margin_bottom + 24}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#555555">{max_step:,}</text>
  <text x="{margin_left - 10}" y="{margin_top + 4}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#555555">{max_loss:.2f}</text>
  <text x="{margin_left - 10}" y="{height - margin_bottom + 4}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#555555">{min_loss:.2f}</text>
  <polyline fill="none" stroke="#0f766e" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round" points="{polyline}"/>
</svg>
"""
    path.write_text(svg)


def _write_architecture_diagram(path: Path, cfg: dict, model_meta: dict[str, str]) -> None:
    html = _architecture_html(cfg, model_meta)
    with tempfile.TemporaryDirectory(prefix="arch-html-") as tmp:
        html_path = Path(tmp) / "architecture.html"
        html_path.write_text(html)
        _render_html_to_png(html_path, path)


def _render_html_to_png(html_path: Path, out_path: Path) -> None:
    cmd = [
        "google-chrome",
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--force-device-scale-factor=2",
        "--window-size=1600,980",
        f"--screenshot={out_path}",
        html_path.as_uri(),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _architecture_html(cfg: dict, model_meta: dict[str, str]) -> str:
    model = cfg["model"]
    training = cfg["training"]
    attention = cfg["attention"]
    if model_meta["model_kind"] == "causal_lm":
        body = _causal_architecture_html(cfg, model, training, attention)
        subtitle = "Meeting summarization causal language model topology"
    else:
        body = _encoder_decoder_architecture_html(cfg, model, training, attention)
        subtitle = "Meeting summarization encoder-decoder topology"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Architecture</title>
  <style>
    :root {{
      --bg: #f3f6fb;
      --panel: rgba(255,255,255,0.84);
      --panel-border: #d8e1f0;
      --text: #0f172a;
      --muted: #52627c;
      --line: #4b5563;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(99, 102, 241, 0.08), transparent 30%),
        radial-gradient(circle at top right, rgba(14, 165, 233, 0.08), transparent 24%),
        linear-gradient(180deg, #f8fbff 0%, #eef3fb 100%);
      color: var(--text);
    }}
    .page {{
      width: 1600px;
      min-height: 980px;
      padding: 28px 34px 34px;
    }}
    .frame {{
      min-height: 918px;
      border: 1px solid rgba(216, 225, 240, 0.9);
      border-radius: 28px;
      background: rgba(255,255,255,0.72);
      box-shadow: 0 24px 60px rgba(15, 23, 42, 0.08);
      padding: 26px 28px 28px;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 18px;
    }}
    .title {{
      font-size: 32px;
      font-weight: 800;
      letter-spacing: 0;
      margin: 0;
    }}
    .subtitle {{
      margin: 10px 0 0;
      font-size: 18px;
      color: #36507a;
      text-align: right;
      max-width: 420px;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 310px 1fr;
      gap: 26px;
      align-items: start;
    }}
    .side, .main {{
      border: 1px solid var(--panel-border);
      border-radius: 24px;
      background: var(--panel);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);
    }}
    .side {{
      padding: 20px 20px 22px;
      min-height: 760px;
    }}
    .main {{
      padding: 20px 22px 22px;
      min-height: 760px;
    }}
    h2 {{
      margin: 4px 0 18px;
      font-size: 18px;
      font-weight: 800;
    }}
    .kv {{
      display: grid;
      gap: 12px;
    }}
    .kv-row {{
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: center;
      background: rgba(255,255,255,0.9);
      border: 1px solid #dfe7f3;
      border-radius: 14px;
      padding: 9px 12px;
      font-size: 13px;
    }}
    .kv-row .k {{ font-weight: 700; color: #334155; }}
    .kv-row .v {{ text-align: right; color: #0f172a; }}
    .stats {{
      margin-top: 18px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    .stat {{
      border-radius: 14px;
      border: 1px solid #c7d2fe;
      background: #eef2ff;
      color: #3730a3;
      font-size: 13px;
      font-weight: 700;
      text-align: center;
      padding: 11px 8px;
    }}
    .chip-section {{
      margin-top: 22px;
    }}
    .chip-title {{
      margin: 0 0 12px;
      font-size: 15px;
      font-weight: 800;
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .chip {{
      border-radius: 999px;
      border: 1px solid #bfdbfe;
      background: #eff6ff;
      color: #1d4ed8;
      font-size: 12px;
      padding: 7px 10px;
      line-height: 1;
      white-space: nowrap;
    }}
    .ed-grid {{
      display: grid;
      grid-template-columns: 400px 140px 400px;
      gap: 24px;
      align-items: start;
    }}
    .column-title {{
      font-size: 18px;
      font-weight: 800;
      margin: 0 0 14px;
    }}
    .flow {{
      display: grid;
      gap: 14px;
    }}
    .arrow {{
      width: 0;
      height: 24px;
      border-left: 3px solid #475569;
      margin: 0 auto;
      position: relative;
    }}
    .arrow::after {{
      content: "";
      position: absolute;
      left: -8px;
      bottom: -1px;
      border-left: 8px solid transparent;
      border-right: 8px solid transparent;
      border-top: 16px solid #475569;
    }}
    .node, .stack, .context, .note-card {{
      border-radius: 20px;
      background: rgba(255,255,255,0.86);
      border: 2px solid #dbe5f5;
      padding: 16px 18px;
      overflow: hidden;
    }}
    .node h3, .stack h3, .context h3, .note-card h3 {{
      margin: 0 0 8px;
      font-size: 16px;
      font-weight: 800;
    }}
    .shape {{
      color: #334155;
      font-size: 13px;
    }}
    .node.blue {{ background: #eff6ff; border-color: #4f78ff; }}
    .node.green {{ background: #ecfdf5; border-color: #10b981; }}
    .node.amber {{ background: #fefce8; border-color: #f59e0b; }}
    .node.orange {{ background: #fff7ed; border-color: #f97316; }}
    .node.violet {{ background: #faf5ff; border-color: #8b5cf6; }}
    .node.red {{ background: #fef2f2; border-color: #ef4444; }}
    .stack.blue {{ border-color: #4f78ff; }}
    .stack.orange {{ border-color: #f97316; }}
    .stack-inner {{
      border-radius: 18px;
      background: rgba(255,255,255,0.88);
      border: 1px solid #dbe5f5;
      padding: 14px 16px 12px;
    }}
    .stack-lines {{
      display: grid;
      gap: 8px;
      font-size: 13px;
      color: #334155;
    }}
    .stack-foot {{
      margin-top: 14px;
      text-align: right;
      font-size: 12px;
      color: #64748b;
    }}
    .context-col {{
      padding-top: 262px;
      display: grid;
      gap: 14px;
      align-content: start;
    }}
    .context {{
      background: rgba(255,255,255,0.94);
      border-color: #cbd5e1;
      padding: 14px 14px 16px;
    }}
    .context p {{
      margin: 0;
      font-size: 13px;
      line-height: 1.45;
      color: #334155;
    }}
    .context .connector {{
      margin: 14px -14px 0;
      padding: 10px 14px 0;
      border-top: 1px solid #e2e8f0;
      font-size: 12px;
      color: #0f766e;
      font-weight: 700;
    }}
    .causal-grid {{
      display: grid;
      grid-template-columns: 1fr 320px;
      gap: 24px;
      align-items: start;
    }}
    .legend {{
      margin-top: 22px;
      border-radius: 18px;
      border: 1px solid #d7dee8;
      background: rgba(255,255,255,0.92);
      padding: 14px 16px;
      display: grid;
      gap: 10px;
    }}
    .legend-row {{
      display: grid;
      grid-template-columns: 132px 1fr;
      gap: 10px;
      font-size: 13px;
      color: #334155;
    }}
    .legend-row strong {{
      color: #0f172a;
      font-size: 13px;
    }}
    .footer {{
      margin-top: 18px;
      text-align: right;
      font-size: 13px;
      color: #64748b;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="frame">
      <div class="header">
        <h1 class="title">Transformer Lab Architecture</h1>
        <p class="subtitle">{escape(subtitle)}</p>
      </div>
      <div class="layout">
        {_config_panel_html(cfg, model_meta)}
        <div class="main">
          {body}
        </div>
      </div>
      <div class="footer">Dimensions are inferred from config.json for this checkpoint.</div>
    </div>
  </div>
</body>
</html>"""


def _config_panel_html(cfg: dict, model_meta: dict[str, str]) -> str:
    model = cfg["model"]
    training = cfg["training"]
    attention = cfg["attention"]
    rows = [
        ("Model", "Causal LM Transformer" if model_meta["model_kind"] == "causal_lm" else "Encoder-Decoder Transformer"),
        ("Attention", str(attention.get("name", "unknown"))),
        ("Connection", str(cfg["connection"].get("name", "unknown"))),
        ("FFN", f'{cfg["feedforward"].get("name", "unknown")} d_ff={model.get("d_ff", "?")}'),
        ("Norm", str(cfg["normalization"].get("name", "unknown"))),
        ("Positional", str(cfg["positional"].get("name", "unknown"))),
        ("Batch / Epochs", f'{training.get("batch_size", "?")} / {training.get("num_epochs", "?")}'),
    ]
    stats = [
        f'd_model {model.get("d_model", "?")}',
        f'layers {model.get("n_layers", "?")}',
        f'heads {attention.get("n_heads", "?")}',
        f'precision {training.get("precision", "fp32")}',
    ]
    chips = "".join(f'<span class="chip">{escape(item)}</span>' for item in _attention_params(attention))
    kv_rows = "".join(
        f'<div class="kv-row"><span class="k">{escape(k)}</span><span class="v">{escape(v)}</span></div>'
        for k, v in rows
    )
    stat_boxes = "".join(f'<div class="stat">{escape(item)}</div>' for item in stats)
    return f"""
    <aside class="side">
      <h2>Run Configuration</h2>
      <div class="kv">{kv_rows}</div>
      <div class="stats">{stat_boxes}</div>
      <div class="chip-section">
        <p class="chip-title">Attention Parameters</p>
        <div class="chips">{chips}</div>
      </div>
    </aside>"""


def _encoder_decoder_architecture_html(cfg: dict, model: dict, training: dict, attention: dict) -> str:
    attn = _join_params(_attention_params(attention), limit=3)
    return f"""
    <div class="ed-grid">
      <section>
        <p class="column-title">Encoder</p>
        <div class="flow">
          {_node_html("blue", "Source Tokens", f'[{model.get("max_src_len", "?")} x {model.get("src_vocab_size", "?")}]')}
          <div class="arrow"></div>
          {_node_html("green", "Embedding + Positional Encoding", f'[{model.get("max_src_len", "?")} x {model.get("d_model", "?")}]')}
          <div class="arrow"></div>
          {_stack_html("blue", "Encoder Stack", [
            f'{model.get("n_layers", "?")} stacked encoder blocks',
            f'self-attn: {attention.get("name", "unknown")}  heads={attention.get("n_heads", "?")}',
            f'attention params: {attn}',
            f'ffn: {cfg["feedforward"].get("name", "unknown")}  d_ff={model.get("d_ff", "?")}',
            f'residual + {cfg["normalization"].get("name", "unknown")}',
          ])}
          <div class="arrow"></div>
          {_node_html("violet", "Encoder Memory", f'[{model.get("max_src_len", "?")} x {model.get("d_model", "?")}]')}
        </div>
      </section>
      <section class="context-col">
        <div class="context">
          <h3>Cross-Attention Context</h3>
          <p>Decoder queries attend over encoder memory instead of reusing source tokens directly.</p>
          <div class="connector">memory width d_model={model.get("d_model", "?")}  |  source length {model.get("max_src_len", "?")}</div>
        </div>
      </section>
      <section>
        <p class="column-title">Decoder</p>
        <div class="flow">
          {_node_html("amber", "Target Tokens", f'[{model.get("max_tgt_len", "?")} x {model.get("tgt_vocab_size", "?")}]')}
          <div class="arrow"></div>
          {_node_html("orange", "Embedding + Positional Encoding", f'[{model.get("max_tgt_len", "?")} x {model.get("d_model", "?")}]')}
          <div class="arrow"></div>
          {_stack_html("orange", "Decoder Stack", [
            f'{model.get("n_layers", "?")} stacked decoder blocks',
            'masked self-attn + encoder cross-attn',
            f'attention params: {attn}',
            f'ffn: {cfg["feedforward"].get("name", "unknown")}  d_ff={model.get("d_ff", "?")}',
            f'projection to vocab {model.get("tgt_vocab_size", "?")}',
          ])}
          <div class="arrow"></div>
          {_node_html("red", "Output Projection", f'[{model.get("max_tgt_len", "?")} x {model.get("tgt_vocab_size", "?")}]')}
        </div>
      </section>
    </div>"""


def _causal_architecture_html(cfg: dict, model: dict, training: dict, attention: dict) -> str:
    seq_len = model.get("max_seq_len", model.get("max_src_len", "?"))
    attn = _join_params(_attention_params(attention), limit=5)
    return f"""
    <div class="causal-grid">
      <section>
        <p class="column-title">Causal Pipeline</p>
        <div class="flow">
          {_node_html("blue", "Packed Transcript + Summary Tokens", f'[{seq_len} x {model.get("src_vocab_size", "?")}]')}
          <div class="arrow"></div>
          {_node_html("green", "Embedding + Positional Encoding", f'[{seq_len} x {model.get("d_model", "?")}]')}
          <div class="arrow"></div>
          {_stack_html("blue", "Causal Transformer Stack", [
            f'{model.get("n_layers", "?")} repeated causal blocks',
            f'self-attn: {attention.get("name", "unknown")}  heads={attention.get("n_heads", "?")}',
            f'attention params: {attn}',
            f'ffn: {cfg["feedforward"].get("name", "unknown")}  d_ff={model.get("d_ff", "?")}',
            f'connection: {cfg["connection"].get("name", "unknown")}',
          ])}
          <div class="arrow"></div>
          {_node_html("red", "Tied Projection + Logits", f'[{seq_len} x {model.get("tgt_vocab_size", "?")}]')}
        </div>
        <div class="legend">
          <div class="legend-row"><strong>Context packing</strong><span>MeetingBank causal mode concatenates transcript context and summary target in one sequence.</span></div>
          <div class="legend-row"><strong>Training objective</strong><span>Prefix tokens are masked from loss; supervision is on the summary continuation.</span></div>
          <div class="legend-row"><strong>Projection</strong><span>Tied embeddings reduce parameter duplication at the token interface.</span></div>
        </div>
      </section>
      <section style="display:grid; gap:16px;">
        {_note_card_html("Architecture Notes", [
          "causal packing: transcript prefix + summary continuation",
          "loss is applied on continuation tokens only",
          "input embedding weights are reused for output projection",
          f"max sequence length: {seq_len}",
        ])}
        {_note_card_html("Attention Detail", [
          f"variant: {attention.get('name', 'unknown')}",
          attn,
          f"normalization: {cfg['normalization'].get('name', 'unknown')}",
          f"batch size: {training.get('batch_size', '?')}   epochs: {training.get('num_epochs', '?')}",
        ])}
        {_note_card_html("Output", [f'logits over vocab size {model.get("tgt_vocab_size", "?")}'])}
      </section>
    </div>"""


def _node_html(kind: str, title: str, shape: str) -> str:
    return f'<div class="node {kind}"><h3>{escape(title)}</h3><div class="shape">{escape(shape)}</div></div>'


def _stack_html(kind: str, title: str, lines: list[str]) -> str:
    items = "".join(f'<div>{escape(line)}</div>' for line in lines)
    return (
        f'<div class="stack {kind}"><h3>{escape(title)}</h3>'
        f'<div class="stack-inner"><div class="stack-lines">{items}</div>'
        f'<div class="stack-foot">residual path around each sublayer</div></div></div>'
    )


def _note_card_html(title: str, lines: list[str]) -> str:
    items = "".join(f'<div>{escape(line)}</div>' for line in lines)
    return f'<div class="note-card"><h3>{escape(title)}</h3><div class="stack-lines">{items}</div></div>'


def _encoder_decoder_architecture_svg(
    cfg: dict,
    model: dict,
    training: dict,
    attention: dict,
) -> str:
    width, height = 1500, 940
    card_y = 116
    card_h = 728
    left_x = 72
    left_w = 312
    gap_left = 28
    center_x = left_x + left_w + gap_left
    center_w = 418
    connector_gap = 104
    right_x = center_x + center_w + connector_gap
    right_w = 418
    panel_y = card_y + 18

    attn_params = _attention_params(attention)
    left_meta = [
        ("Model", "Encoder-Decoder Transformer"),
        ("Attention", str(attention.get("name", "unknown"))),
        ("Connection", str(cfg["connection"].get("name", "unknown"))),
        ("FFN", f'{cfg["feedforward"].get("name", "unknown")}  d_ff={model.get("d_ff", "?")}'),
        ("Norm", str(cfg["normalization"].get("name", "unknown"))),
        ("Positional", str(cfg["positional"].get("name", "unknown"))),
        ("Batch / Epochs", f'{training.get("batch_size", "?")} / {training.get("num_epochs", "?")}'),
    ]
    enc_lines = [
        f"{model.get('n_layers', '?')} stacked encoder blocks",
        f"self-attn: {attention.get('name', 'unknown')}  heads={attention.get('n_heads', '?')}",
        f"attention params: {_join_params(attn_params, limit=3)}",
        f"ffn: {cfg['feedforward'].get('name', 'unknown')}  d_ff={model.get('d_ff', '?')}",
        f"residual + {cfg['normalization'].get('name', 'unknown')}",
    ]
    dec_lines = [
        f"{model.get('n_layers', '?')} stacked decoder blocks",
        "masked self-attn + encoder cross-attn",
        f"attention params: {_join_params(attn_params, limit=3)}",
        f"ffn: {cfg['feedforward'].get('name', 'unknown')}  d_ff={model.get('d_ff', '?')}",
        f"projection to vocab {model.get('tgt_vocab_size', '?')}",
    ]

    parts = [_svg_open(width, height, "Encoder-decoder transformer architecture")]
    parts.extend(_diagram_defs())
    parts.append(_svg_backdrop(width, height))
    parts.append(_title_block("Transformer Lab Architecture", "Meeting summarization encoder-decoder topology", width))
    parts.append(_card(left_x, card_y, left_w, card_h, "#f8fafc"))
    parts.append(_card(center_x, card_y, center_w, card_h, "#f8fafc"))
    parts.append(_card(right_x, card_y, right_w, card_h, "#f8fafc"))

    parts.append(_section_label(left_x + 24, panel_y + 18, "Run Configuration"))
    for idx, (label, value) in enumerate(left_meta):
        y = panel_y + 56 + idx * 48
        parts.append(_kv_row(left_x + 24, y, left_w - 48, label, value))
    parts.append(_stat_grid(left_x + 24, panel_y + 414, left_w - 48, [
        f"d_model {model.get('d_model', '?')}",
        f"layers {model.get('n_layers', '?')}",
        f"heads {attention.get('n_heads', '?')}",
        f"precision {training.get('precision', 'fp32')}",
    ]))
    parts.append(_attention_chip_group(left_x + 24, panel_y + 554, left_w - 48, attention))

    parts.append(_section_label(center_x + 24, panel_y + 18, "Encoder"))
    parts.append(_token_box(center_x + 24, panel_y + 54, center_w - 48, 74, "#eff6ff", "#3b82f6", "Source Tokens", f"[{model.get('max_src_len', '?')} x {model.get('src_vocab_size', '?')}]"))
    parts.append(_arrow(center_x + center_w / 2, panel_y + 128, center_x + center_w / 2, panel_y + 164))
    parts.append(_token_box(center_x + 24, panel_y + 164, center_w - 48, 74, "#ecfdf5", "#10b981", "Embedding + Positional Encoding", f"[{model.get('max_src_len', '?')} x {model.get('d_model', '?')}]"))
    parts.append(_arrow(center_x + center_w / 2, panel_y + 238, center_x + center_w / 2, panel_y + 276))
    parts.append(_stack_box(center_x + 24, panel_y + 276, center_w - 48, 250, "#f8fafc", "#2563eb", "Encoder Stack", enc_lines))
    parts.append(_arrow(center_x + center_w / 2, panel_y + 526, center_x + center_w / 2, panel_y + 564))
    parts.append(_token_box(center_x + 24, panel_y + 564, center_w - 48, 72, "#faf5ff", "#8b5cf6", "Encoder Memory", f"[{model.get('max_src_len', '?')} x {model.get('d_model', '?')}]"))

    parts.append(_section_label(right_x + 24, panel_y + 18, "Decoder"))
    parts.append(_token_box(right_x + 24, panel_y + 54, right_w - 48, 74, "#fefce8", "#f59e0b", "Target Tokens", f"[{model.get('max_tgt_len', '?')} x {model.get('tgt_vocab_size', '?')}]"))
    parts.append(_arrow(right_x + right_w / 2, panel_y + 128, right_x + right_w / 2, panel_y + 164))
    parts.append(_token_box(right_x + 24, panel_y + 164, right_w - 48, 74, "#fff7ed", "#f97316", "Embedding + Positional Encoding", f"[{model.get('max_tgt_len', '?')} x {model.get('d_model', '?')}]"))
    parts.append(_arrow(right_x + right_w / 2, panel_y + 238, right_x + right_w / 2, panel_y + 276))
    parts.append(_stack_box(right_x + 24, panel_y + 276, right_w - 48, 250, "#fffaf5", "#ea580c", "Decoder Stack", dec_lines))
    parts.append(_arrow(right_x + right_w / 2, panel_y + 526, right_x + right_w / 2, panel_y + 564))
    parts.append(_token_box(right_x + 24, panel_y + 564, right_w - 48, 72, "#fef2f2", "#ef4444", "Output Projection", f"[{model.get('max_tgt_len', '?')} x {model.get('tgt_vocab_size', '?')}]"))

    lane_x = center_x + center_w + 16
    lane_w = connector_gap - 32
    parts.append(_connector_panel(lane_x, panel_y + 308, lane_w, 162, "Cross-Attention Context", [
        "decoder queries attend over encoder memory",
        f"memory width: d_model={model.get('d_model', '?')}",
        f"source length: {model.get('max_src_len', '?')} tokens",
    ]))
    parts.append(_cross_arrow(center_x + center_w, panel_y + 600, lane_x, panel_y + 388, "memory"))
    parts.append(_cross_arrow(lane_x + lane_w, panel_y + 388, right_x, panel_y + 388, "context"))
    parts.append(_footer_note(width, height, "Dimensions are inferred from config.json for this checkpoint."))
    parts.append("</svg>")
    return "".join(parts)


def _causal_architecture_svg(
    cfg: dict,
    model: dict,
    training: dict,
    attention: dict,
) -> str:
    width, height = 1500, 920
    parts = [_svg_open(width, height, "Causal transformer architecture")]
    parts.extend(_diagram_defs())
    parts.append(_svg_backdrop(width, height))
    parts.append(_title_block("Transformer Lab Architecture", "Meeting summarization causal language model topology", width))

    left_x, top_y, left_w, left_h = 72, 122, 312, 690
    flow_x, flow_y, flow_w, flow_h = 412, 122, 1016, 690
    parts.append(_card(left_x, top_y, left_w, left_h, "#f8fafc"))
    parts.append(_card(flow_x, flow_y, flow_w, flow_h, "#f8fafc"))

    parts.append(_section_label(left_x + 24, top_y + 34, "Run Configuration"))
    meta_rows = [
        ("Model", "Causal LM Transformer"),
        ("Attention", str(attention.get("name", "unknown"))),
        ("Connection", str(cfg["connection"].get("name", "unknown"))),
        ("FFN", f'{cfg["feedforward"].get("name", "unknown")}  d_ff={model.get("d_ff", "?")}'),
        ("Norm", str(cfg["normalization"].get("name", "unknown"))),
        ("Sequence", f'max_seq_len={model.get("max_seq_len", model.get("max_src_len", "?"))}'),
        ("Batch / Epochs", f'{training.get("batch_size", "?")} / {training.get("num_epochs", "?")}'),
    ]
    for idx, (label, value) in enumerate(meta_rows):
        parts.append(_kv_row(left_x + 24, top_y + 74 + idx * 48, left_w - 48, label, value))
    parts.append(_stat_grid(left_x + 24, top_y + 414, left_w - 48, [
        f"d_model {model.get('d_model', '?')}",
        f"layers {model.get('n_layers', '?')}",
        f"heads {attention.get('n_heads', '?')}",
        f"precision {training.get('precision', 'fp32')}",
    ]))
    parts.append(_attention_chip_group(left_x + 24, top_y + 554, left_w - 48, attention))

    parts.append(_section_label(flow_x + 24, flow_y + 34, "Causal Pipeline"))
    lane_y = flow_y + 98
    lane_x = flow_x + 34
    node_w = flow_w - 68
    detail_w = 312
    flow_w_main = node_w - detail_w - 28
    main_x = lane_x
    detail_x = lane_x + flow_w_main + 28
    seq_shape = f"[{model.get('max_seq_len', model.get('max_src_len', '?'))} x {model.get('src_vocab_size', '?')}]"
    hidden_shape = f"[{model.get('max_seq_len', model.get('max_src_len', '?'))} x {model.get('d_model', '?')}]"
    logits_shape = f"[{model.get('max_seq_len', model.get('max_src_len', '?'))} x {model.get('tgt_vocab_size', '?')}]"
    parts.append(_token_box(main_x, lane_y, flow_w_main, 84, "#eff6ff", "#3b82f6", "Packed Transcript + Summary Tokens", seq_shape))
    parts.append(_arrow(main_x + flow_w_main / 2, lane_y + 84, main_x + flow_w_main / 2, lane_y + 124))
    parts.append(_token_box(main_x, lane_y + 124, flow_w_main, 84, "#ecfdf5", "#10b981", "Embedding + Positional Encoding", hidden_shape))
    parts.append(_arrow(main_x + flow_w_main / 2, lane_y + 208, main_x + flow_w_main / 2, lane_y + 248))
    parts.append(_stack_box(main_x, lane_y + 248, flow_w_main, 244, "#f8fafc", "#2563eb", "Causal Transformer Stack", [
        f"{model.get('n_layers', '?')} repeated causal blocks",
        f"self-attn: {attention.get('name', 'unknown')}  heads={attention.get('n_heads', '?')}",
        f"attention params: {_join_params(_attention_params(attention), limit=3)}",
        f"ffn: {cfg['feedforward'].get('name', 'unknown')}  d_ff={model.get('d_ff', '?')}",
        f"connection: {cfg['connection'].get('name', 'unknown')}",
        f"block width: d_model={model.get('d_model', '?')}",
    ]))
    parts.append(_arrow(main_x + flow_w_main / 2, lane_y + 492, main_x + flow_w_main / 2, lane_y + 532))
    parts.append(_token_box(main_x, lane_y + 532, flow_w_main, 84, "#fef2f2", "#ef4444", "Tied Projection + Logits", logits_shape))
    parts.append(_connector_panel(detail_x, lane_y, detail_w, 264, "Architecture Notes", [
        "causal packing: transcript prefix + summary continuation",
        "loss is applied on continuation tokens only",
        "input embedding weights are reused for output projection",
        f"max sequence length: {model.get('max_seq_len', model.get('max_src_len', '?'))}",
    ]))
    parts.append(_connector_panel(detail_x, lane_y + 292, detail_w, 212, "Attention Detail", [
        f"variant: {attention.get('name', 'unknown')}",
        _join_params(_attention_params(attention), limit=5),
        f"normalization: {cfg['normalization'].get('name', 'unknown')}",
        f"batch size: {training.get('batch_size', '?')}   epochs: {training.get('num_epochs', '?')}",
    ]))
    parts.append(_connector_panel(detail_x, lane_y + 532, detail_w, 84, "Output", [
        f"logits over vocab size {model.get('tgt_vocab_size', '?')}",
    ]))
    parts.append(_legend_strip(flow_x + 34, flow_y + 640, flow_w - 68, [
        ("Context packing", "MeetingBank causal mode concatenates transcript context and summary target in one sequence."),
        ("Training objective", "Prefix tokens are masked from loss; supervision is on the summary continuation."),
        ("Projection", "Tied embeddings reduce parameter duplication at the token interface."),
    ]))
    parts.append(_footer_note(width, height, "Dimensions are inferred from config.json for this checkpoint."))
    parts.append("</svg>")
    return "".join(parts)


def _attention_params(attention: dict) -> list[str]:
    keys = ("n_heads", "m", "k", "n_win", "g", "c", "c_I", "d_c", "d_g", "n_I_h", "rope_dim")
    out = []
    for key in keys:
        if key in attention:
            out.append(f"{key}={attention[key]}")
    return out


def _join_params(params: list[str], limit: int = 4) -> str:
    if not params:
        return "standard"
    visible = params[:limit]
    extra = len(params) - len(visible)
    if extra > 0:
        visible.append(f"+{extra} more")
    return "  ".join(visible)


def _svg_open(width: int, height: int, label: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(label)}">'
    )


def _diagram_defs() -> list[str]:
    return [
        """
<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="#f8fafc"/>
    <stop offset="100%" stop-color="#eef2ff"/>
  </linearGradient>
  <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
    <feDropShadow dx="0" dy="10" stdDeviation="16" flood-color="#0f172a" flood-opacity="0.08"/>
  </filter>
  <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#475569"/>
  </marker>
</defs>
"""
    ]


def _svg_backdrop(width: int, height: int) -> str:
    return (
        f'<rect width="{width}" height="{height}" fill="url(#bg)"/>'
        f'<rect x="24" y="24" width="{width - 48}" height="{height - 48}" rx="28" fill="#ffffff" opacity="0.68"/>'
    )


def _title_block(title: str, subtitle: str, width: int) -> str:
    return (
        f'<text x="72" y="74" font-family="Inter, Arial, sans-serif" font-size="30" font-weight="700" fill="#0f172a">{escape(title)}</text>'
        f'<text x="{width - 72}" y="74" text-anchor="end" font-family="Inter, Arial, sans-serif" font-size="16" fill="#475569">{escape(subtitle)}</text>'
    )


def _card(x: int | float, y: int | float, w: int | float, h: int | float, fill: str) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="24" fill="{fill}" stroke="#d7dee8" stroke-width="1.2" filter="url(#shadow)"/>'
    )


def _section_label(x: int | float, y: int | float, text: str) -> str:
    return f'<text x="{x}" y="{y}" font-family="Inter, Arial, sans-serif" font-size="20" font-weight="700" fill="#0f172a">{escape(text)}</text>'


def _kv_row(x: int | float, y: int | float, w: int | float, label: str, value: str) -> str:
    return (
        f'<rect x="{x}" y="{y - 22}" width="{w}" height="34" rx="12" fill="#ffffff" stroke="#e2e8f0"/>'
        f'<text x="{x + 14}" y="{y}" font-family="Inter, Arial, sans-serif" font-size="13" font-weight="700" fill="#334155">{escape(label)}</text>'
        f'<text x="{x + w - 14}" y="{y}" text-anchor="end" font-family="Inter, Arial, sans-serif" font-size="13" fill="#0f172a">{escape(value)}</text>'
    )


def _pill_row(x: int | float, y: int | float, items: list[str]) -> str:
    parts = []
    cursor = x
    for item in items:
        width = max(110, 12 + len(item) * 7)
        parts.append(
            f'<rect x="{cursor}" y="{y}" width="{width}" height="34" rx="17" fill="#e0e7ff" stroke="#c7d2fe"/>'
            f'<text x="{cursor + width / 2}" y="{y + 22}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="13" font-weight="700" fill="#3730a3">{escape(item)}</text>'
        )
        cursor += width + 10
    return "".join(parts)


def _stat_grid(x: int | float, y: int | float, w: int | float, items: list[str]) -> str:
    cols = 2
    gap = 12
    cell_w = (w - gap) / cols
    cell_h = 42
    parts = []
    for idx, item in enumerate(items):
        row = idx // cols
        col = idx % cols
        cell_x = x + col * (cell_w + gap)
        cell_y = y + row * (cell_h + gap)
        parts.append(
            f'<rect x="{cell_x}" y="{cell_y}" width="{cell_w}" height="{cell_h}" rx="14" fill="#eef2ff" stroke="#c7d2fe"/>'
            f'<text x="{cell_x + cell_w / 2}" y="{cell_y + 27}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="13" font-weight="700" fill="#3730a3">{escape(item)}</text>'
        )
    return "".join(parts)


def _attention_chip_group(x: int | float, y: int | float, w: int | float, attention: dict) -> str:
    chips = _attention_params(attention)
    parts = [
        f'<text x="{x}" y="{y}" font-family="Inter, Arial, sans-serif" font-size="16" font-weight="700" fill="#0f172a">Attention Parameters</text>'
    ]
    cursor_x = x
    cursor_y = y + 18
    max_x = x + w
    for chip in chips:
        chip_w = max(88, 18 + len(chip) * 7)
        if cursor_x + chip_w > max_x:
            cursor_x = x
            cursor_y += 38
        parts.append(
            f'<rect x="{cursor_x}" y="{cursor_y}" width="{chip_w}" height="30" rx="15" fill="#eff6ff" stroke="#bfdbfe"/>'
            f'<text x="{cursor_x + chip_w / 2}" y="{cursor_y + 20}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="12" fill="#1d4ed8">{escape(chip)}</text>'
        )
        cursor_x += chip_w + 8
    return "".join(parts)


def _token_box(x: int | float, y: int | float, w: int | float, h: int | float, fill: str, stroke: str, title: str, subtitle: str) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'
        f'<text x="{x + 18}" y="{y + 32}" font-family="Inter, Arial, sans-serif" font-size="16" font-weight="700" fill="#0f172a">{escape(title)}</text>'
        f'<text x="{x + 18}" y="{y + 56}" font-family="Inter, Arial, sans-serif" font-size="13" fill="#334155">{escape(subtitle)}</text>'
    )


def _stack_box(x: int | float, y: int | float, w: int | float, h: int | float, fill: str, stroke: str, title: str, lines: list[str]) -> str:
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="20" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
        f'<rect x="{x + 14}" y="{y + 14}" width="{w - 28}" height="{h - 28}" rx="16" fill="#ffffff" opacity="0.78" stroke="#dbeafe"/>',
        f'<text x="{x + 18}" y="{y + 34}" font-family="Inter, Arial, sans-serif" font-size="18" font-weight="700" fill="#0f172a">{escape(title)}</text>',
    ]
    for idx, line in enumerate(lines):
        parts.append(
            f'<text x="{x + 18}" y="{y + 68 + idx * 26}" font-family="Inter, Arial, sans-serif" font-size="13" fill="#334155">{escape(line)}</text>'
        )
    parts.append(
        f'<text x="{x + w - 18}" y="{y + h - 18}" text-anchor="end" font-family="Inter, Arial, sans-serif" font-size="12" fill="#64748b">residual path around each sublayer</text>'
    )
    return "".join(parts)


def _arrow(x1: int | float, y1: int | float, x2: int | float, y2: int | float) -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#475569" stroke-width="2.2" marker-end="url(#arrowhead)"/>'


def _cross_arrow(x1: int | float, y1: int | float, x2: int | float, y2: int | float, label: str) -> str:
    mid_x = (x1 + x2) / 2
    mid_y = min(y1, y2) - 42
    return (
        f'<path d="M {x1} {y1} C {mid_x} {mid_y}, {mid_x} {mid_y}, {x2} {y2}" fill="none" stroke="#0f766e" stroke-width="2.4" marker-end="url(#arrowhead)"/>'
        f'<text x="{mid_x}" y="{mid_y - 8}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="13" fill="#0f766e">{escape(label)}</text>'
    )


def _connector_panel(x: int | float, y: int | float, w: int | float, h: int | float, title: str, lines: list[str]) -> str:
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="#ffffff" stroke="#d7dee8"/>',
        f'<text x="{x + 18}" y="{y + 30}" font-family="Inter, Arial, sans-serif" font-size="16" font-weight="700" fill="#0f172a">{escape(title)}</text>',
    ]
    for idx, line in enumerate(lines):
        parts.append(
            f'<text x="{x + 18}" y="{y + 58 + idx * 22}" font-family="Inter, Arial, sans-serif" font-size="13" fill="#334155">{escape(line)}</text>'
        )
    return "".join(parts)


def _legend_strip(x: int | float, y: int | float, w: int | float, items: list[tuple[str, str]]) -> str:
    parts = [f'<rect x="{x}" y="{y}" width="{w}" height="168" rx="20" fill="#ffffff" stroke="#d7dee8"/>']
    for idx, (label, value) in enumerate(items):
        row_y = y + 34 + idx * 44
        parts.append(
            f'<text x="{x + 18}" y="{row_y}" font-family="Inter, Arial, sans-serif" font-size="14" font-weight="700" fill="#0f172a">{escape(label)}</text>'
            f'<text x="{x + 132}" y="{row_y}" font-family="Inter, Arial, sans-serif" font-size="14" fill="#334155">{escape(value)}</text>'
        )
    return "".join(parts)


def _footer_note(width: int, height: int, text: str) -> str:
    return f'<text x="{width - 72}" y="{height - 56}" text-anchor="end" font-family="Inter, Arial, sans-serif" font-size="13" fill="#64748b">{escape(text)}</text>'


def _copy_tokenizers(root: Path, out_dir: Path, cfg: dict) -> list[Path]:
    transcript = root / "tokenizers" / "meetingbank_transcript.json"
    summary = root / "tokenizers" / "meetingbank_summary.json"
    causal = root / "tokenizers" / "meetingbank_causal.json"
    model_kind = str(cfg.get("model", {}).get("kind", "encoder_decoder"))

    tokenizer = out_dir / "tokenizer.json"
    if model_kind == "causal_lm":
        if not causal.exists():
            raise RuntimeError("MeetingBank causal tokenizer file is missing")
        causal_out = out_dir / "causal_tokenizer.json"
        shutil.copyfile(causal, tokenizer)
        shutil.copyfile(causal, causal_out)
        return [tokenizer, causal_out]

    if not transcript.exists() or not summary.exists():
        raise RuntimeError("MeetingBank encoder-decoder tokenizer files are missing")
    transcript_out = out_dir / "transcript_tokenizer.json"
    summary_out = out_dir / "summary_tokenizer.json"
    shutil.copyfile(transcript, tokenizer)
    shutil.copyfile(transcript, transcript_out)
    shutil.copyfile(summary, summary_out)
    return [tokenizer, transcript_out, summary_out]


def _upload_artifacts(api: HfApi, token: str, repo_id: str, artifacts: list[Path]) -> None:
    create_repo(repo_id, token=token, exist_ok=True, private=False)
    for path in artifacts:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add production model card artifact {path.name}",
        )
    print(f"uploaded artifacts to {repo_id}")


def _ensure_collection_membership(api: HfApi, collection_slug: str, repo_ids: list[str]) -> None:
    collection = api.get_collection(collection_slug)
    existing = {item.item_id for item in collection.items if item.item_type == "model"}
    for repo_id in repo_ids:
        if repo_id not in existing:
            api.add_collection_item(
                collection_slug=collection_slug,
                item_id=repo_id,
                item_type="model",
            )
            print(f"added {repo_id} to {collection_slug}")


def _ordered_unique(items: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


if __name__ == "__main__":
    main()
