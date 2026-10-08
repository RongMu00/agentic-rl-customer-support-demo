from __future__ import annotations

import argparse
import json
from typing import Any

from training_smoke.common import group_relative_advantages
from training_smoke.reward_adapter import build_candidate_group


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one offline, sequence-level GRPO-style policy update."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    prompt, candidates = build_candidate_group()
    rewards = [candidate.reward for candidate in candidates]
    advantages = torch.tensor(
        group_relative_advantages(rewards), dtype=torch.float32, device=device
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    mean_logprobs = torch.stack(
        [
            _completion_mean_logprob(
                model,
                tokenizer,
                prompt,
                candidate.completion,
                args.max_length,
                torch,
                device,
            )
            for candidate in candidates
        ]
    ).float()

    # The replayed candidates stand in for a rollout group. At smoke-test step zero,
    # the current policy is also the frozen old policy. This still exercises the
    # clipped group-relative objective and its gradient path without online sampling.
    old_logprobs = mean_logprobs.detach()
    ratios = torch.exp(mean_logprobs - old_logprobs)
    unclipped = ratios * advantages
    clipped = torch.clamp(ratios, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages
    loss = -torch.minimum(unclipped, clipped).mean()
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite policy loss: {loss.item()}")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    print(
        json.dumps(
            {
                "status": "ok",
                "mode": "offline-sequence-grpo-style",
                "model": args.model,
                "device": device,
                "policy_loss": round(float(loss.detach().cpu()), 6),
                "gradient_norm": round(float(gradient_norm.detach().cpu()), 6),
                "candidates": [
                    {
                        "name": candidate.name,
                        "reward": candidate.reward,
                        "advantage": round(float(advantage.detach().cpu()), 6),
                        "mean_logprob": round(float(logprob.detach().cpu()), 6),
                        "active_caps": list(candidate.active_caps),
                    }
                    for candidate, advantage, logprob in zip(
                        candidates, advantages, mean_logprobs, strict=True
                    )
                ],
            },
            indent=2,
        )
    )


def _completion_mean_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    completion: str,
    max_length: int,
    torch: Any,
    device: str,
) -> Any:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(
        prompt + completion,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("--max-length removed the complete candidate completion")

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shifted_logits = logits[:, :-1, :]
    shifted_targets = input_ids[:, 1:]
    token_logprobs = torch.log_softmax(shifted_logits.float(), dim=-1).gather(
        -1, shifted_targets.unsqueeze(-1)
    ).squeeze(-1)
    completion_start = max(len(prompt_ids) - 1, 0)
    completion_logprobs = token_logprobs[0, completion_start:]
    if completion_logprobs.numel() == 0:
        raise ValueError("candidate completion contains no trainable tokens")
    return completion_logprobs.mean()


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
