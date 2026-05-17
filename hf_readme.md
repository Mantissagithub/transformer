---
library_name: pytorch
license: apache-2.0
datasets:
- huuuyeah/meetingbank
tags:
- pytorch
- transformer
- meeting-summarization
- custom-code
- attention-variant
---

# {model_title}

Custom PyTorch Transformer checkpoint trained on MeetingBank for meeting summarization research. This repository is part of the [`transformer-lab`](https://huggingface.co/collections/Pradheep1647/transformer-lab-6a07fe3185f5728e217997e0) collection.

## Model Details

| Field | Value |
|---|---|
| Repository | `{repo_id}` |
| Attention | `{attention}` |
| Dataset | `{dataset}` |
| Layers | `{n_layers}` |
| Hidden size | `{d_model}` |
| Heads | `{n_heads}` |
| Batch size | `{batch_size}` |
| Epochs | `{num_epochs}` |
| Precision | `{precision}` |
| Checkpoint | `{checkpoint_file}` |

## Architecture

![Architecture diagram]({architecture_file})

Static architecture diagram generated from this run's `config.json`, including model width, depth, sequence dimensions, and attention-specific settings.

## Training Loss

![Training loss](loss_curve.svg)

Raw curve data is available in [`loss_curve.csv`](loss_curve.csv).

## Available Models

{available_models_table}

## Files

| File | Purpose |
|---|---|
| `{checkpoint_file}` | PyTorch checkpoint containing `model_state_dict`, optimizer states, epoch, and global step. |
| `config.json` | Training and architecture config converted from the Hydra run config. |
| `{architecture_file}` | Architecture diagram generated from the saved model config, with block shapes and dimensions. |
| `tokenizer.json` | MeetingBank transcript tokenizer alias for source inputs. |
| `transcript_tokenizer.json` | Explicit MeetingBank transcript tokenizer. |
| `summary_tokenizer.json` | MeetingBank summary tokenizer for target text. |
| `loss_curve.csv` | TensorBoard `train/loss` scalar export. |
| `loss_curve.svg` | Static training-loss plot generated from `loss_curve.csv`. |

## Usage

These checkpoints are from a custom PyTorch codebase, not a `transformers.AutoModel` checkpoint. Use the repo-native builder to instantiate the architecture, then load the checkpoint state dict.

```python
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

import src  # registers components
from src.model.builder import {builder_function}

repo_id = "{repo_id}"

config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
checkpoint_path = hf_hub_download(repo_id=repo_id, filename="{checkpoint_file}")

cfg = OmegaConf.load(config_path)
model = {builder_function}(cfg)

state = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(state["model_state_dict"])
model.eval()

print(f"Loaded {{repo_id}} from {{Path(checkpoint_path).name}}")
```

## Notes

- This is a research checkpoint for comparing attention variants under the same MeetingBank setup.
- The config and tokenizers are included so future runs can reproduce the architecture and preprocessing assumptions.
- Use `config.json` as the source of truth for architecture parameters.
