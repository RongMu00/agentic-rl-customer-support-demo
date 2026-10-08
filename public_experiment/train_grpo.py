from __future__ import annotations

import argparse
import json
from pathlib import Path

from .reward import verifier_reward


DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "train.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small-scale GRPO with verifier rewards.")
    parser.add_argument("--sft-adapter", default="artifacts/sft-adapter")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/grpo-adapter"))
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from datasets import Dataset
        from peft import AutoPeftModelForCausalLM, PeftConfig
        from transformers import AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as error:
        raise SystemExit("Install public_experiment/requirements-colab.txt first.") from error

    records = _load_jsonl(args.data)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32

    adapter_config = PeftConfig.from_pretrained(args.sft_adapter)
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.sft_adapter,
        is_trainable=True,
        dtype=dtype,
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_config.base_model_name_or_path, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = GRPOConfig(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.num_generations,
        learning_rate=1e-5,
        beta=0.04,
        temperature=0.8,
        max_completion_length=512,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=use_bf16,
        fp16=use_fp16,
        seed=args.seed,
    )
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=Dataset.from_list(
            [{"prompt": row["prompt"], "scenario_json": row["scenario_json"]} for row in records]
        ),
        reward_funcs=verifier_reward,
        processing_class=tokenizer,
    )
    result = trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    metadata = {
        "stage": "grpo",
        "sft_adapter": args.sft_adapter,
        "train_prompts": len(records),
        "max_steps": args.max_steps,
        "num_generations": args.num_generations,
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
