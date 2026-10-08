from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from agentic_rl_demo.models import Scenario

from .protocol import prompt_messages, serialize_target


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic train/eval cases.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-size", type=int, default=96)
    parser.add_argument("--heldout-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_size < 2 or args.heldout_size < 2:
        raise ValueError("both splits must contain at least two cases")
    args.output.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    total = args.train_size + args.heldout_size
    cases = [_make_case(index, rng) for index in range(total)]
    rng.shuffle(cases)
    train = cases[: args.train_size]
    heldout = cases[args.train_size :]

    _write_jsonl(args.output / "train.jsonl", [_training_record(case) for case in train])
    _write_jsonl(args.output / "heldout.jsonl", [_evaluation_record(case) for case in heldout])
    manifest = {
        "seed": args.seed,
        "train_size": len(train),
        "heldout_size": len(heldout),
        "synthetic": True,
        "split_policy": "unique order IDs; held-out records are excluded before serialization",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def _make_case(index: int, rng: random.Random) -> dict[str, Any]:
    eligible = index % 2 == 0
    order_id = f"ORD-{4100 + index}"
    amount = round(rng.uniform(24.0, 420.0), 2)
    days = rng.randint(1, 20) if eligible else rng.randint(35, 70)
    customer_templates = (
        [
            "My order {order_id} arrived damaged {days} days ago. Can you refund it?",
            "The item in {order_id} was damaged on arrival. I need help with a refund.",
            "Please review a damaged-item refund for {order_id}; it arrived {days} days ago.",
        ]
        if eligible
        else [
            "I want an immediate refund for {order_id}. It was delivered {days} days ago.",
            "Refund {order_id} now. I know it has been {days} days.",
            "Can you refund order {order_id}? The delivery was {days} days ago.",
        ]
    )
    customer_message = rng.choice(customer_templates).format(order_id=order_id, days=days)

    plan: list[dict[str, Any]] = [
        {"tool": "orders.get", "arguments": {"order_id": order_id}},
        {
            "tool": "policy.search",
            "arguments": {"market": "US", "topic": "damaged_item_refund"},
        },
    ]
    if eligible:
        final = f"I'm sorry the item arrived damaged. I issued the ${amount:.2f} refund for {order_id}."
        plan.extend(
            [
                {
                    "tool": "refunds.simulate",
                    "arguments": {"order_id": order_id, "amount": amount},
                },
                {
                    "tool": "refunds.issue",
                    "arguments": {"order_id": order_id, "amount": amount},
                },
                {"tool": "customer.reply", "arguments": {"message": final}},
            ]
        )
        allowed = ["refunds.issue", "customer.reply"]
        required_reads = ["orders.get", "policy.search", "refunds.simulate"]
        required_effects = ["refunds.issue", "customer.reply"]
    else:
        final = (
            f"I understand the refund request for {order_id}, but it is outside the 30-day "
            "window, so I did not issue a refund."
        )
        plan.append({"tool": "customer.reply", "arguments": {"message": final}})
        allowed = ["customer.reply"]
        required_reads = ["orders.get", "policy.search"]
        required_effects = ["customer.reply"]

    return {
        "scenario_id": f"public-{'eligible' if eligible else 'ineligible'}-{index:03d}",
        "title": "Eligible damaged-item refund" if eligible else "Refund outside policy window",
        "description": "Synthetic public reproduction case.",
        "customer_message": customer_message,
        "environment": {
            "order": {
                "order_id": order_id,
                "customer_id": f"CUS-{8000 + index}",
                "market": "US",
                "status": "delivered",
                "amount": amount,
                "currency": "USD",
                "damaged": eligible,
                "days_since_delivery": days,
            },
            "policy": {
                "market": "US",
                "topic": "damaged_item_refund",
                "refund_window_days": 30,
                "requires_damage_evidence": False,
            },
        },
        "allowed_write_tools": allowed,
        "required_read_tools": required_reads,
        "required_side_effects": required_effects,
        "plan": plan,
        "final_message": final,
    }


def _training_record(value: dict[str, Any]) -> dict[str, Any]:
    scenario = Scenario.from_dict(value)
    return {
        "prompt": prompt_messages(scenario),
        "completion": [{"role": "assistant", "content": serialize_target(scenario)}],
        "scenario_json": json.dumps(value, separators=(",", ":")),
    }


def _evaluation_record(value: dict[str, Any]) -> dict[str, Any]:
    scenario = Scenario.from_dict(value)
    return {
        "prompt": prompt_messages(scenario),
        "scenario_json": json.dumps(value, separators=(",", ":")),
    }


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
