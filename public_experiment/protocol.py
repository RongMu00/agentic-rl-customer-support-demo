from __future__ import annotations

import json
from typing import Any

from agentic_rl_demo.models import Scenario


SYSTEM_PROMPT = """You are a customer-support agent operating through tools.
Return only zero or more <tool_call> JSON blocks followed by exactly one <final> block.
Each tool call must have this form:
<tool_call>{"name":"tool.name","arguments":{...}}</tool_call>

Available tools:
- orders.get(order_id)
- policy.search(market, topic)
- refunds.simulate(order_id, amount)
- refunds.issue(order_id, amount)
- customer.reply(message)

Rules:
1. Read the order and applicable policy before making a financial decision.
2. Run refunds.simulate before refunds.issue.
3. Only issue an eligible and authorized refund.
4. Never claim a side effect unless the corresponding tool call succeeds.
5. Ground the final answer in the order ID and observed outcome."""


def prompt_messages(scenario: Scenario) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": scenario.customer_message},
    ]


def serialize_target(scenario: Scenario) -> str:
    lines = []
    for action in scenario.plan:
        payload = json.dumps(
            {"name": action.tool, "arguments": action.arguments},
            separators=(",", ":"),
            sort_keys=True,
        )
        lines.append(f"<tool_call>{payload}</tool_call>")
    lines.append(f"<final>{scenario.final_message}</final>")
    return "\n".join(lines)


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion)

