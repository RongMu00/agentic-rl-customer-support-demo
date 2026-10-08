from __future__ import annotations

from collections import Counter
from uuid import uuid4

from .models import Scenario, ToolEvent, Trajectory
from .sandbox import Sandbox
from .tools import TOOLS, ToolError


def run_scenario(scenario: Scenario, *, namespace_id: str | None = None) -> Trajectory:
    """Execute a deterministic plan through the same boundaries as a model rollout.

    A model-backed version would produce each action dynamically. The public demo
    uses a deterministic plan so the safety and verification path runs on any laptop.
    """

    namespace = namespace_id or f"rollout-{uuid4().hex[:10]}"
    sandbox = Sandbox(namespace)
    trajectory = Trajectory(
        scenario_id=scenario.scenario_id,
        namespace_id=namespace,
        customer_message=scenario.customer_message,
    )
    failure_budget = Counter(scenario.failure_injections)

    for step, action in enumerate(scenario.plan, start=1):
        definition = TOOLS.get(action.tool)
        if definition is None:
            trajectory.events.append(
                ToolEvent(step, action.tool, action.arguments, "error", error="unknown_tool", attempt=1)
            )
            continue

        if definition.is_write and action.tool not in scenario.allowed_write_tools:
            trajectory.events.append(
                ToolEvent(
                    step=step,
                    tool=action.tool,
                    arguments=action.arguments,
                    status="blocked",
                    error="authorization_denied",
                    retryable=False,
                )
            )
            continue

        max_attempts = scenario.max_retries + 1
        for attempt in range(1, max_attempts + 1):
            if failure_budget[action.tool] > 0:
                failure_budget[action.tool] -= 1
                trajectory.events.append(
                    ToolEvent(
                        step=step,
                        tool=action.tool,
                        arguments=action.arguments,
                        status="error",
                        error="transient_upstream_timeout",
                        retryable=True,
                        attempt=attempt,
                    )
                )
                if attempt < max_attempts:
                    continue
                break
            try:
                result = definition.handler(action.arguments, scenario, sandbox)
                trajectory.events.append(
                    ToolEvent(step, action.tool, action.arguments, "success", result=result, attempt=attempt)
                )
            except ToolError as error:
                trajectory.events.append(
                    ToolEvent(
                        step=step,
                        tool=action.tool,
                        arguments=action.arguments,
                        status="error",
                        error=error.code,
                        retryable=error.retryable,
                        attempt=attempt,
                    )
                )
                if error.retryable and attempt < max_attempts:
                    continue
            break

    trajectory.final_message = scenario.final_message
    trajectory.sandbox_final_state = sandbox.export()
    return trajectory
