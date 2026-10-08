from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from .reward import score_completion


DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "heldout.jsonl"
BOOLEAN_METRICS = (
    "valid_tool_calls",
    "task_completed",
    "policy_compliant",
    "unauthorized_action",
    "false_promise",
    "grounded_response",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a base model or LoRA adapter.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter")
    parser.add_argument("--stage", choices=("base", "sft", "grpo"), required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from peft import AutoPeftModelForCausalLM, PeftConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise SystemExit("Install public_experiment/requirements-colab.txt first.") from error

    base_model = (
        PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
        if args.adapter
        else args.model
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16 if torch.cuda.is_available() else torch.float32
    )
    if args.adapter:
        model = AutoPeftModelForCausalLM.from_pretrained(args.adapter, dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    records = _load_jsonl(args.data)
    examples = []
    for record in records:
        rendered = tokenizer.apply_chat_template(
            record["prompt"], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        metrics = score_completion(completion, record["scenario_json"])
        examples.append({"completion": completion, **metrics})

    summary = {
        "scope": "public_reproducible_experiment",
        "stage": args.stage,
        "model": args.model,
        "adapter": args.adapter,
        "heldout_cases": len(examples),
        "metrics": {
            "mean_reward": round(mean(item["reward"] for item in examples), 4),
            **{
                name: round(mean(float(item[name]) for item in examples), 4)
                for name in BOOLEAN_METRICS
            },
        },
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "examples"}, indent=2))


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


if __name__ == "__main__":
    main()
