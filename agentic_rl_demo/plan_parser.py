from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace

from .models import Action, Scenario


TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
FINAL_PATTERN = re.compile(r"<final>\s*(.*?)\s*</final>", re.DOTALL)


@dataclass(frozen=True)
class ParsedPlan:
    actions: tuple[Action, ...]
    final_message: str
    valid: bool
    errors: tuple[str, ...]


def parse_completion(completion: str) -> ParsedPlan:
    errors: list[str] = []
    actions: list[Action] = []

    for index, match in enumerate(TOOL_CALL_PATTERN.finditer(completion), start=1):
        try:
            payload = json.loads(match.group(1))
            if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
                raise ValueError("expected name and arguments")
            arguments = payload.get("arguments", {})
            if not isinstance(arguments, dict):
                raise ValueError("arguments must be an object")
            actions.append(Action(tool=payload["name"], arguments=arguments))
        except (json.JSONDecodeError, ValueError) as error:
            errors.append(f"tool_call_{index}: {error}")

    final_match = FINAL_PATTERN.search(completion)
    final_message = final_match.group(1).strip() if final_match else ""
    if not actions:
        errors.append("no_valid_tool_calls")
    if not final_message:
        errors.append("missing_final_message")

    outside = TOOL_CALL_PATTERN.sub("", completion)
    outside = FINAL_PATTERN.sub("", outside).strip()
    if outside:
        errors.append("text_outside_protocol")

    return ParsedPlan(
        actions=tuple(actions),
        final_message=final_message,
        valid=not errors,
        errors=tuple(errors),
    )


def scenario_with_plan(base: Scenario, parsed: ParsedPlan) -> Scenario:
    return replace(base, plan=parsed.actions, final_message=parsed.final_message)

