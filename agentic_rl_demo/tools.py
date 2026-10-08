from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable

from .models import Scenario
from .sandbox import Sandbox


class ToolError(RuntimeError):
    def __init__(self, message: str, *, code: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    is_write: bool
    handler: Callable[[dict[str, Any], Scenario, Sandbox], dict[str, Any]]


def _require(arguments: dict[str, Any], key: str) -> Any:
    value = arguments.get(key)
    if value in (None, ""):
        raise ToolError(f"missing required argument: {key}", code="invalid_arguments")
    return value


def get_order(arguments: dict[str, Any], scenario: Scenario, sandbox: Sandbox) -> dict[str, Any]:
    order_id = str(_require(arguments, "order_id"))
    if order_id != str(scenario.order["order_id"]):
        raise ToolError("order not found", code="not_found")
    return dict(scenario.order)


def search_policy(arguments: dict[str, Any], scenario: Scenario, sandbox: Sandbox) -> dict[str, Any]:
    market = str(_require(arguments, "market"))
    topic = str(_require(arguments, "topic"))
    if market != scenario.policy["market"] or topic != scenario.policy["topic"]:
        raise ToolError("no matching policy", code="not_found")
    return dict(scenario.policy)


def simulate_refund(arguments: dict[str, Any], scenario: Scenario, sandbox: Sandbox) -> dict[str, Any]:
    order_id = str(_require(arguments, "order_id"))
    amount = Decimal(str(_require(arguments, "amount")))
    expected = Decimal(str(scenario.order["amount"]))
    if order_id != scenario.order["order_id"]:
        raise ToolError("order not found", code="not_found")
    if amount != expected:
        raise ToolError("refund amount does not match order", code="amount_mismatch")
    eligible = scenario.order["damaged"] and scenario.order["days_since_delivery"] <= scenario.policy["refund_window_days"]
    return {"eligible": eligible, "order_id": order_id, "amount": float(amount), "currency": scenario.order["currency"]}


def issue_refund(arguments: dict[str, Any], scenario: Scenario, sandbox: Sandbox) -> dict[str, Any]:
    order_id = str(_require(arguments, "order_id"))
    amount = Decimal(str(_require(arguments, "amount")))
    if order_id != scenario.order["order_id"]:
        raise ToolError("order not found", code="not_found")
    if sandbox.records("refund_ledger"):
        raise ToolError("refund already exists", code="duplicate_write")
    record = sandbox.append(
        "refund_ledger",
        {"order_id": order_id, "amount": float(amount), "currency": scenario.order["currency"], "status": "issued"},
        tool="refunds.issue",
    )
    return {"refund_id": f"RF-{order_id.split('-')[-1]}", **record}


def reply_to_customer(arguments: dict[str, Any], scenario: Scenario, sandbox: Sandbox) -> dict[str, Any]:
    message = str(_require(arguments, "message"))
    record = sandbox.append("message_log", {"message": message}, tool="customer.reply")
    return {"delivered": True, **record}


TOOLS: dict[str, ToolDefinition] = {
    "orders.get": ToolDefinition("orders.get", False, get_order),
    "policy.search": ToolDefinition("policy.search", False, search_policy),
    "refunds.simulate": ToolDefinition("refunds.simulate", False, simulate_refund),
    "refunds.issue": ToolDefinition("refunds.issue", True, issue_refund),
    "customer.reply": ToolDefinition("customer.reply", True, reply_to_customer),
}
