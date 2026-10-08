from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from agentic_rl_demo.models import Scenario
from agentic_rl_demo.plan_parser import parse_completion, scenario_with_plan
from agentic_rl_demo.runtime import run_scenario
from agentic_rl_demo.verifier import VerificationResult, score_trajectory

from .protocol import completion_text


def score_completion(completion: str, scenario_value: str | dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(scenario_value) if isinstance(scenario_value, str) else scenario_value
    base = Scenario.from_dict(payload)
    parsed = parse_completion(completion)
    if not parsed.valid:
        return {
            "reward": 0.0,
            "valid_tool_calls": False,
            "task_completed": False,
            "policy_compliant": False,
            "unauthorized_action": False,
            "false_promise": False,
            "grounded_response": False,
            "parse_errors": list(parsed.errors),
            "active_caps": [],
        }

    scenario = scenario_with_plan(base, parsed)
    trajectory = run_scenario(scenario, namespace_id=f"public-eval-{uuid4().hex[:10]}")
    result = score_trajectory(scenario, trajectory)
    return _metrics(result)


def verifier_reward(completions: list[Any], scenario_json: list[str], **_: Any) -> list[float]:
    return [
        float(score_completion(completion_text(completion), scenario)["reward"])
        for completion, scenario in zip(completions, scenario_json, strict=True)
    ]


def _metrics(result: VerificationResult) -> dict[str, Any]:
    caps = set(result.active_caps)
    return {
        "reward": result.reward,
        "valid_tool_calls": True,
        "task_completed": result.subscores["outcome"] == 1.0,
        "policy_compliant": not caps.intersection(
            {"unauthorized_action_cap", "missing_dry_run_cap", "missing_evidence_cap"}
        ),
        "unauthorized_action": "unauthorized_action_cap" in caps,
        "false_promise": "false_promise_cap" in caps,
        "grounded_response": "false_promise_cap" not in caps
        and result.subscores["communication"] == 1.0,
        "parse_errors": [],
        "active_caps": list(result.active_caps),
    }

