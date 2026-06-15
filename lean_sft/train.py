from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_dataset

from .config import load_config, require_env
from .schema import format_for_sft


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Lean StarCoder SFT adapters with TRL.")
    parser.add_argument("--phase", choices=["syntax", "proof"], required=True)
    parser.add_argument("--dataset-repo", default=None)
    parser.add_argument("--base-or-adapter", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--push-repo", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    token = require_env("HF_TOKEN")
    train_cfg = cfg["training"]
    dataset_repo = args.dataset_repo or cfg["dataset_repo"]
    ds = load_dataset(dataset_repo, token=token)
    train_ds = _phase_dataset(ds["train"], args.phase)
    eval_ds = _phase_dataset(ds["validation"], args.phase)
    model_name = args.base_or_adapter or cfg["base_model"]

    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], token=token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = None
    precision = str(train_cfg.get("precision", "fp16"))
    compute_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    if bool(train_cfg["use_4bit"]):
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        token=token,
        device_map="auto",
        torch_dtype="auto",
        quantization_config=quant_cfg,
        trust_remote_code=True,
    )
    if args.base_or_adapter:
        model = PeftModel.from_pretrained(model, args.base_or_adapter, token=token)

    peft_cfg = LoraConfig(
        r=int(train_cfg["lora_r"]),
        lora_alpha=int(train_cfg["lora_alpha"]),
        lora_dropout=float(train_cfg["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        logging_dir=str(Path(cfg.get("logging", {}).get("tensorboard_dir", "runs/lean-starcoder-sft")) / args.phase),
        max_length=int(train_cfg["max_length"]),
        completion_only_loss=True,
        num_train_epochs=float(train_cfg[f"phase{1 if args.phase == 'syntax' else 2}_epochs"]),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=250,
        save_total_limit=2,
        bf16=precision == "bf16",
        fp16=precision == "fp16",
        report_to=["tensorboard"],
        push_to_hub=True,
        hub_model_id=args.push_repo,
        hub_token=token,
        seed=int(train_cfg["seed"]),
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_cfg,
        processing_class=tokenizer,
        formatting_func=lambda rows: [format_for_sft(row) for row in _rows(rows)],
    )
    trainer.train()
    trainer.push_to_hub(commit_message=f"train {args.phase} Lean SFT adapter")

    if args.phase == "proof" and bool(train_cfg["merge_final"]):
        _merge_and_push(cfg, args.output_dir, args.push_repo, token)


def _phase_dataset(dataset, phase: str):
    return dataset.filter(lambda row: row["phase"] == phase)


def _rows(batch):
    if isinstance(batch, dict) and batch and all(isinstance(v, list) for v in batch.values()):
        return [dict(zip(batch, values)) for values in zip(*batch.values())]
    return [batch]


def _merge_and_push(cfg: dict, output_dir: str, adapter_repo: str, token: str) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_dir = Path(output_dir) / "merged"
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        token=token,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_repo, token=token)
    merged = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], token=token, trust_remote_code=True)
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    merged.push_to_hub(cfg["merged_repo"], token=token, commit_message="merge final Lean SFT model")
    tokenizer.push_to_hub(cfg["merged_repo"], token=token, commit_message="add tokenizer")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
