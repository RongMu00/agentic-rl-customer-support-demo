from __future__ import annotations

from copy import deepcopy
from typing import Any


class Sandbox:
    """Rollout-scoped mutable state.

    Read-only business data stays outside this object. Every write is namespaced,
    audited, and isolated from concurrent rollouts.
    """

    def __init__(self, namespace_id: str):
        self.namespace_id = namespace_id
        self._state: dict[str, Any] = {
            "refund_ledger": [],
            "message_log": [],
            "audit_log": [],
        }

    def append(self, ledger: str, record: dict[str, Any], *, tool: str) -> dict[str, Any]:
        enriched = {
            **deepcopy(record),
            "tool": tool,
            "namespace_id": self.namespace_id,
        }
        self._state.setdefault(ledger, []).append(enriched)
        self._state["audit_log"].append(
            {
                "tool": tool,
                "namespace_id": self.namespace_id,
                "result": "ok",
                "record": deepcopy(record),
            }
        )
        return deepcopy(enriched)

    def records(self, ledger: str) -> list[dict[str, Any]]:
        return [
            deepcopy(row)
            for row in self._state.get(ledger, [])
            if row.get("namespace_id") == self.namespace_id
        ]

    def export(self) -> dict[str, Any]:
        return deepcopy(self._state)
