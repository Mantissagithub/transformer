# Transformer

A modular PyTorch transformer for research. Every component — attention, FFN, normalization, positional encoding, residual connection, optimizer, dataset — is registered by name and selected from a YAML. Swap anything without touching the trainer.

Supports both encoder-decoder summarization (MeetingBank, Multi-News) and decoder-only causal-LM pretraining at the ~500M-parameter scale (FineWeb-Edu, FSDP, bf16, KV-cache generation).

## Why this exists

Started as a hand-rolled transformer for meeting summarization. Trying any new attention or connection variant meant editing several files. This rewrite makes the base composable — experiments are one YAML each, components are decoupled, the trainer doesn't know what attention you picked.

## Quick start

```bash
pip install -e .

# default: residual transformer on MeetingBank
python -m src.cli.train

# ~500M decoder-only pretrain on FineWeb-Edu
python -m src.cli.train +experiment=pretrain_500m

# mix and match anything inline
python -m src.cli.train attention=gqa_rope feedforward=swiglu normalization=rmsnorm
```

## Adding a new component

Same four steps for every kind (attention, FFN, norm, optimizer, dataset, ...):

1. Write the class. Inherit from the base, decorate with `@<KIND>.register("name")`.

   ```python
   # src/components/attention/my_attn.py
   @ATTENTION.register("my_attn")
   class MyAttention(AttentionBase):
       ...
   ```

2. Import the module in the package's `__init__.py` so the decorator runs at import time.
3. Add `configs/attention/my_attn.yaml` with `name: my_attn` and any kwargs.
4. Use it: `python -m src.cli.train attention=my_attn`.

No trainer or builder edits needed.

## What's available

| Group | Choices |
|-------|---------|
| Attention | mha, gqa, gqa_rope, mqa, sliding_window, sliding_gqa, gemma3_hybrid, csa, hca, mla, msa. See [`docs/attention.md`](docs/attention.md) for the variant notes and diagrams. |
| FFN | relu, swiglu, geglu. See [`docs/feedforward.md`](docs/feedforward.md). |
| Normalization | layernorm, rmsnorm. See [`docs/normalization.md`](docs/normalization.md). |
| Positional | sinusoidal, rope, alibi, rope/nope hybrid. See [`docs/positional.md`](docs/positional.md). |
| Connection | residual, residual_sandwich, hyperconnection, mhc. See [`docs/connections.md`](docs/connections.md). |
| Optimizer | adamw, muon_adamw, lion, adafactor, ademamix, mars_adamw. See [`docs/optimizers.md`](docs/optimizers.md). |
| Scheduler | cosine_warmup, linear_warmup, inverse_sqrt_warmup, polynomial_warmup, wsd, none. See [`docs/schedulers.md`](docs/schedulers.md). |
| Dataset | meetingbank, multi_news, fineweb_edu, c4, wikitext, wikipedia. See [`docs/datasets.md`](docs/datasets.md). |

Plus: bf16/fp16 autocast, gradient accumulation, `torch.compile`, DDP/FSDP, HF Hub push, KV-cache `.generate()` across every attention variant, Rich TUI.

## Tests

```bash
pytest tests/ -q
```

## License

Apache-2.0. See [`LICENSE`](LICENSE).
