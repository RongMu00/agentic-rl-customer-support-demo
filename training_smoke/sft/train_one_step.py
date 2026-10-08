from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training_smoke.common import load_jsonl, mask_prompt_tokens, validate_sft_record


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "sft_examples.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a one-step assistant-only SFT smoke test.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise SystemExit(
            'Training dependencies are missing. Install them with: pip install -e ".[training-smoke]"'
        ) from error

    device = _resolve_device(args.device, torch)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.config.use_cache = False
    model.train()

    records = load_jsonl(args.data)
    if not records:
        raise ValueError("the SFT dataset is empty")
    for record in records:
        validate_sft_record(record)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    losses: list[float] = []
    gradient_norms: list[float] = []

    for step in range(args.steps):
        record = records[step % len(records)]
        batch = _tokenize_record(record, tokenizer, args.max_length, torch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite SFT loss: {loss.item()}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        gradient_norms.append(float(gradient_norm.detach().cpu()))

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    print(
        json.dumps(
            {
                "status": "ok",
                "mode": "assistant-only-sft",
                "model": args.model,
                "device": device,
                "steps": args.steps,
                "losses": [round(value, 6) for value in losses],
                "gradient_norms": [round(value, 6) for value in gradient_norms],
                "checkpoint_saved": str(args.output_dir) if args.output_dir else None,
            },
            indent=2,
        )
    )


def _tokenize_record(
    record: dict[str, Any], tokenizer: Any, max_length: int, torch: Any, device: str
) -> dict[str, Any]:
    messages = record["messages"]
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    prompt_length = min(len(prompt_ids), len(input_ids))
    labels = mask_prompt_tokens(input_ids, prompt_length)
    if all(label == -100 for label in labels):
        raise ValueError("--max-length removed the complete assistant target")

    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long, device=device),
        "labels": torch.tensor([labels], dtype=torch.long, device=device),
    }


def _resolve_device(requested: str, torch: Any) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    main()
