from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
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
EXTRA_REPOS = ["Pradheep1647/run_sliding_gqa-meetingbank-bs8-e20-fp32-19"]
LOSS_TAG = "train/loss"
CHECKPOINT_NAME = "meeting_model19.pt"


@dataclass(frozen=True)
class ModelTarget:
    repo_id: str
    run_name: str
    config_path: Path
    event_dir: Path


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    token = _hf_token(root)
    api = HfApi(token=token)

    collection_slug, collection_repos = _collection_repos(api)
    repo_ids = _ordered_unique([*collection_repos, *EXTRA_REPOS])
    publishable_repo_ids = []
    skipped_repo_ids = []
    for repo_id in repo_ids:
        if _repo_has_checkpoint(api, repo_id):
            publishable_repo_ids.append(repo_id)
        else:
            skipped_repo_ids.append(repo_id)

    for repo_id in skipped_repo_ids:
        print(f"skipping {repo_id}: missing {CHECKPOINT_NAME}")
        if not args.dry_run:
            _remove_from_collection(api, collection_slug, repo_id)

    repo_ids = publishable_repo_ids
    targets = [_resolve_target(root, repo_id) for repo_id in repo_ids]

    if args.dry_run:
        for target in targets:
            print(f"{target.repo_id}")
            print(f"  config: {target.config_path}")
            print(f"  events: {target.event_dir}")
        return

    with tempfile.TemporaryDirectory(prefix="transformer-hf-publish-") as tmp:
        tmp_root = Path(tmp)
        available_table = _available_models_table(targets)
        template = (root / "hf_readme.md").read_text()
        for target in targets:
            out_dir = tmp_root / target.repo_id.replace("/", "__")
            out_dir.mkdir(parents=True, exist_ok=True)
            artifacts = _build_artifacts(root, target, targets, available_table, template, out_dir)
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


def _repo_has_checkpoint(api: HfApi, repo_id: str) -> bool:
    try:
        files = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    except Exception:
        return False
    return CHECKPOINT_NAME in files


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


def _resolve_target(root: Path, repo_id: str) -> ModelTarget:
    run_name = _run_name(repo_id)
    config_path = _latest_config(root / "outputs" / run_name)
    event_dir = _event_dir(root / "runs", run_name)
    return ModelTarget(
        repo_id=repo_id,
        run_name=run_name,
        config_path=config_path,
        event_dir=event_dir,
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


def _event_dir(runs_dir: Path, run_name: str) -> Path:
    direct = runs_dir / run_name
    if direct.exists() and list(direct.glob("events.out.tfevents*")):
        return direct

    candidates = sorted(
        path
        for path in runs_dir.glob(f"{run_name}*")
        if path.is_dir() and list(path.glob("events.out.tfevents*"))
    )
    if not candidates:
        raise RuntimeError(f"no TensorBoard event dir found for {run_name}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _build_artifacts(
    root: Path,
    target: ModelTarget,
    targets: list[ModelTarget],
    available_table: str,
    template: str,
    out_dir: Path,
) -> list[Path]:
    cfg = yaml.safe_load(target.config_path.read_text())
    model_meta = _model_meta(target.repo_id, cfg)

    config_json = out_dir / "config.json"
    config_json.write_text(json.dumps(cfg, indent=2, sort_keys=False) + "\n")

    loss_rows = _loss_rows(target.event_dir)
    loss_csv = out_dir / "loss_curve.csv"
    _write_loss_csv(loss_csv, loss_rows)
    _write_loss_svg(out_dir / "loss_curve.svg", loss_rows, model_meta["attention"])

    readme = template.format(
        available_models_table=available_table,
        checkpoint_file=CHECKPOINT_NAME,
        model_title=_model_title(target.run_name),
        repo_id=target.repo_id,
        **model_meta,
    )
    (out_dir / "README.md").write_text(readme)

    tokenizer_artifacts = _copy_tokenizers(root, out_dir)
    return [
        out_dir / "README.md",
        config_json,
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
        "num_epochs": str(training.get("num_epochs", "unknown")),
        "precision": str(training.get("precision", "fp32")),
    }


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


def _copy_tokenizers(root: Path, out_dir: Path) -> list[Path]:
    transcript = root / "tokenizers" / "meetingbank_transcript.json"
    summary = root / "tokenizers" / "meetingbank_summary.json"
    if not transcript.exists() or not summary.exists():
        raise RuntimeError("MeetingBank tokenizer files are missing")

    tokenizer = out_dir / "tokenizer.json"
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
