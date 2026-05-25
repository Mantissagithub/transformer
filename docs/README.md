# Transformer Lab docs

These docs describe the registered component families in this repo. The attention page uses rendered architecture diagrams; the other component pages use equations, paper tags, and implementation notes.

| Component group | Docs |
|---|---|
| Attention | [`attention.md`](attention.md) |
| Feed-forward blocks | [`feedforward.md`](feedforward.md) |
| Normalization | [`normalization.md`](normalization.md) |
| Positional encoding | [`positional.md`](positional.md) |
| Connections | [`connections.md`](connections.md) |
| Optimizers | [`optimizers.md`](optimizers.md) |
| Schedulers | [`schedulers.md`](schedulers.md) |
| Datasets and tokenization | [`datasets.md`](datasets.md) |
| Embeddings, projections, loss | [`output.md`](output.md) |

The local source of truth remains the registry plus YAML configs under `configs/`. These pages explain what each choice is meant to do and where the paper-backed variants come from.
