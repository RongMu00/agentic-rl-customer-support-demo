from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Action:
    tool: str
    arguments: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Action":
        return cls(tool=str(value["tool"]), arguments=dict(value.get("arguments", {})))


@dataclass
class ToolEvent:
    step: int
    tool: str
    arguments: dict[str, Any]
    status: str
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    retryable: bool = False
    attempt: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Trajectory:
    scenario_id: str
    namespace_id: str
    customer_message: str
    events: list[ToolEvent] = field(default_factory=list)
    final_message: str = ""
    sandbox_final_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "namespace_id": self.namespace_id,
            "customer_message": self.customer_message,
            "events": [event.to_dict() for event in self.events],
            "final_message": self.final_message,
            "sandbox_final_state": self.sandbox_final_state,
        }


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    title: str
    description: str
    customer_message: str
    order: dict[str, Any]
    policy: dict[str, Any]
    plan: tuple[Action, ...]
    allowed_write_tools: frozenset[str]
    required_read_tools: tuple[str, ...]
    required_side_effects: tuple[str, ...]
    final_message: str
    failure_injections: dict[str, int] = field(default_factory=dict)
    max_retries: int = 2

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Scenario":
        return cls(
            scenario_id=str(value["scenario_id"]),
            title=str(value["title"]),
            description=str(value["description"]),
            customer_message=str(value["customer_message"]),
            order=dict(value["environment"]["order"]),
            policy=dict(value["environment"]["policy"]),
            plan=tuple(Action.from_dict(item) for item in value["plan"]),
            allowed_write_tools=frozenset(value.get("allowed_write_tools", [])),
            required_read_tools=tuple(value.get("required_read_tools", [])),
            required_side_effects=tuple(value.get("required_side_effects", [])),
            final_message=str(value["final_message"]),
            failure_injections={str(k): int(v) for k, v in value.get("failure_injections", {}).items()},
            max_retries=int(value.get("max_retries", 2)),
        )
