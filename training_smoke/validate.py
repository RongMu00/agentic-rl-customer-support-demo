from __future__ import annotations

import json
from pathlib import Path

from .common import group_relative_advantages, load_jsonl, validate_sft_record
from .reward_adapter import build_candidate_group


DATA_PATH = Path(__file__).resolve().parent / "data" / "sft_examples.jsonl"


def main() -> None:
    records = load_jsonl(DATA_PATH)
    for record in records:
        validate_sft_record(record)

    _, candidates = build_candidate_group()
    rewards = [candidate.reward for candidate in candidates]
    advantages = group_relative_advantages(rewards)
    print(
        json.dumps(
            {
                "sft_records": len(records),
                "candidates": [
                    {
                        "name": candidate.name,
                        "reward": candidate.reward,
                        "advantage": round(advantage, 4),
                        "active_caps": list(candidate.active_caps),
                    }
                    for candidate, advantage in zip(candidates, advantages, strict=True)
                ],
                "status": "ok",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

