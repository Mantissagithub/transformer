# Lean StarCoder SFT

Focused SFT pipeline for teaching `bigcode/starcoderbase-1b` Lean syntax first, then simple Lean proofs.

## Shape

- Phase 1: synthetic Lean syntax examples.
- Phase 2: synthetic proof-completion examples, trained from the phase-1 adapter.
- Generator: OpenRouter chat completions.
- Trainer: TRL `SFTTrainer` with LoRA/QLoRA.
- Remote compute: Google Colab CLI.
- Outputs: Hugging Face datasets, phase adapters, merged final model, and benchmark report.

## Secrets

Put credentials in `.env` or the shell. `.env` is ignored by git.

```bash
OPENROUTER_API_KEY=replace_with_openrouter_key
HF_TOKEN=replace_with_hf_write_token
HF_USERNAME=Pradheep1647
```

The generator uses `deepseek/deepseek-v4-flash` only. If OpenRouter rejects that exact id, the run fails instead of silently falling back.

## Local smoke checks

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m lean_sft.generate_dataset --phase syntax --count 5 --out data/generated/syntax.jsonl --dry-run
python -m lean_sft.validate_dataset data/generated/syntax.jsonl --out data/processed/syntax_valid.jsonl
```

## Colab run

```bash
scripts/run_colab_training.sh
```

The script checks the Colab CLI, creates a named T4 session, uploads this checkout, runs the full pipeline remotely, downloads logs/artifacts, and stops the session.

## TensorBoard

Every stage writes metrics under `runs/lean-starcoder-sft`:

- dataset generation accepted/attempted rows
- validation accepted/rejected rows
- dataset push row counts
- TRL train/eval metrics
- benchmark compile/generation metrics

```bash
tensorboard --logdir runs/lean-starcoder-sft
```

## Hugging Face targets

Defaults:

- Dataset: `Pradheep1647/lean-starcoder-sft-data`
- Phase 1 adapter: `Pradheep1647/starcoderbase-1b-lean-syntax-lora`
- Phase 2 adapter: `Pradheep1647/starcoderbase-1b-lean-proofs-lora`
- Merged final model: `Pradheep1647/starcoderbase-1b-lean-sft`

## Benchmark

The benchmark report separates real Lean verification from text-only metrics. If `lean` is unavailable, the report says so and only records generation/format metrics.
