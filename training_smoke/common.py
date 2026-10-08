from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(value)
    return records


def validate_sft_record(record: dict[str, Any]) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("an SFT record needs at least two messages")

    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("message role must be system, user, or assistant")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError("message content must be a non-empty string")

    if messages[-1]["role"] != "assistant":
        raise ValueError("the final SFT message must be the assistant target")
    if any(message["role"] == "assistant" for message in messages[:-1]):
        raise ValueError("this smoke test expects one final assistant target")


def mask_prompt_tokens(input_ids: Iterable[int], prompt_length: int) -> list[int]:
    ids = list(input_ids)
    if prompt_length < 0 or prompt_length > len(ids):
        raise ValueError("prompt_length must be within the input sequence")
    return [-100] * prompt_length + ids[prompt_length:]


def group_relative_advantages(rewards: Iterable[float], epsilon: float = 1e-8) -> list[float]:
    values = [float(reward) for reward in rewards]
    if len(values) < 2:
        raise ValueError("group-relative advantages require at least two rewards")

    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    if variance <= epsilon:
        return [0.0 for _ in values]
    scale = math.sqrt(variance + epsilon)
    return [(value - mean) / scale for value in values]

