from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "train.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT for the public reproduction.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sft-adapter"))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:
        raise SystemExit("Install public_experiment/requirements-colab.txt first.") from error

    records = _load_jsonl(args.data)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        max_length=args.max_length,
        completion_only_loss=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=use_bf16,
        fp16=use_fp16,
        seed=args.seed,
        model_init_kwargs={
            "dtype": torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32
        },
    )
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=Dataset.from_list(records),
        peft_config=lora,
    )
    result = trainer.train()
    trainer.save_model(str(args.output_dir))
    metadata = {
        "stage": "sft",
        "base_model": args.model,
        "train_examples": len(records),
        "epochs": args.epochs,
        "train_loss": result.metrics.get("train_loss"),
        "seed": args.seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


if __name__ == "__main__":
    main()
