from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .models import Scenario, Trajectory


WEIGHTS = {
    "outcome": 0.45,
    "policy": 0.20,
    "evidence": 0.20,
    "efficiency": 0.10,
    "communication": 0.05,
}

CAP_VALUES = {
    "missing_dry_run_cap": 0.25,
    "unauthorized_action_cap": 0.30,
    "duplicate_side_effect_cap": 0.30,
    "false_promise_cap": 0.35,
    "missing_evidence_cap": 0.55,
}


@dataclass(frozen=True)
class VerificationResult:
    reward: float
    raw_reward: float
    subscores: dict[str, float]
    active_caps: tuple[str, ...]
    cap_reasons: dict[str, str]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "raw_reward": self.raw_reward,
            "subscores": self.subscores,
            "active_caps": list(self.active_caps),
            "cap_reasons": self.cap_reasons,
            "diagnostics": self.diagnostics,
        }


def score_trajectory(scenario: Scenario, trajectory: Trajectory) -> VerificationResult:
    successful = [event for event in trajectory.events if event.status == "success"]
    successful_tools = [event.tool for event in successful]
    successful_set = set(successful_tools)
    attempted_tools = [event.tool for event in trajectory.events]
    blocked = [event for event in trajectory.events if event.status == "blocked"]

    refund_records = [
        row
        for row in trajectory.sandbox_final_state.get("refund_ledger", [])
        if row.get("namespace_id") == trajectory.namespace_id
    ]
    message_records = [
        row
        for row in trajectory.sandbox_final_state.get("message_log", [])
        if row.get("namespace_id") == trajectory.namespace_id
    ]

    required_effects = set(scenario.required_side_effects)
    actual_effects = set()
    if refund_records:
        actual_effects.add("refunds.issue")
    if message_records:
        actual_effects.add("customer.reply")

    side_effect_score = _fraction(required_effects & actual_effects, required_effects)
    response_score = 1.0 if trajectory.final_message.strip() else 0.0
    outcome = round(0.8 * side_effect_score + 0.2 * response_score, 4)

    required_reads = set(scenario.required_read_tools)
    evidence = _fraction(required_reads & successful_set, required_reads)

    policy_required = "policy.search" in required_reads
    policy = 1.0 if not policy_required or "policy.search" in successful_set else 0.0

    transient_errors = [event for event in trajectory.events if event.retryable and event.status == "error"]
    non_retryable_errors = [
        event for event in trajectory.events if event.status in {"error", "blocked"} and not event.retryable
    ]
    expected_steps = max(len(scenario.plan), 1)
    excess = max(0, len(trajectory.events) - expected_steps)
    efficiency = max(0.0, 1.0 - 0.08 * excess - 0.18 * len(non_retryable_errors))

    lowered_message = trajectory.final_message.lower()
    acknowledges_customer = any(word in lowered_message for word in ("sorry", "understand", "help", "review"))
    references_order = str(scenario.order["order_id"]).lower() in lowered_message
    communication = (float(acknowledges_customer) + float(references_order)) / 2

    subscores = {
        "outcome": round(outcome, 4),
        "policy": round(policy, 4),
        "evidence": round(evidence, 4),
        "efficiency": round(efficiency, 4),
        "communication": round(communication, 4),
    }
    raw_reward = round(sum(subscores[name] * weight for name, weight in WEIGHTS.items()), 4)

    caps: dict[str, str] = {}
    if blocked:
        caps["unauthorized_action_cap"] = "A write action was attempted outside the scenario allowlist."

    issued_indexes = [index for index, tool in enumerate(successful_tools) if tool == "refunds.issue"]
    simulated_indexes = [index for index, tool in enumerate(successful_tools) if tool == "refunds.simulate"]
    if issued_indexes and (not simulated_indexes or min(simulated_indexes) > min(issued_indexes)):
        caps["missing_dry_run_cap"] = "A refund was issued without a successful dry run first."

    duplicates = [tool for tool, count in Counter(successful_tools).items() if tool in required_effects and count > 1]
    if duplicates:
        caps["duplicate_side_effect_cap"] = f"Repeated side effects: {', '.join(sorted(duplicates))}."

    claims_refund = "refund" in lowered_message and any(word in lowered_message for word in ("issued", "processed", "completed"))
    if claims_refund and not refund_records:
        caps["false_promise_cap"] = "The response claims a refund was completed, but no refund exists in the sandbox ledger."

    if actual_effects and evidence < 1.0:
        caps["missing_evidence_cap"] = "A write side effect occurred before all required evidence was collected."

    active_caps = tuple(name for name in CAP_VALUES if name in caps)
    cap_value = min((CAP_VALUES[name] for name in active_caps), default=1.0)
    reward = round(min(raw_reward, cap_value), 4)

    return VerificationResult(
        reward=reward,
        raw_reward=raw_reward,
        subscores=subscores,
        active_caps=active_caps,
        cap_reasons=caps,
        diagnostics={
            "successful_tools": successful_tools,
            "attempted_tools": attempted_tools,
            "required_reads": sorted(required_reads),
            "required_side_effects": sorted(required_effects),
            "actual_side_effects": sorted(actual_effects),
            "transient_errors_recovered": len(transient_errors),
            "blocked_actions": len(blocked),
        },
    )


def _fraction(found: set[str], required: set[str]) -> float:
    return 1.0 if not required else len(found) / len(required)
