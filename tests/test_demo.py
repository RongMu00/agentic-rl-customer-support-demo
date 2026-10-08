from __future__ import annotations

import unittest
from dataclasses import replace

from agentic_rl_demo.models import Action
from agentic_rl_demo.runtime import run_scenario
from agentic_rl_demo.scenarios import available_scenarios, load_scenario
from agentic_rl_demo.verifier import score_trajectory


class DemoTests(unittest.TestCase):
    def test_expected_scenarios_are_available(self) -> None:
        self.assertEqual(
            available_scenarios(),
            ["successful_refund", "transient_failure", "unauthorized_refund"],
        )

    def test_successful_refund_scores_without_caps(self) -> None:
        scenario = load_scenario("successful_refund")
        trajectory = run_scenario(scenario, namespace_id="test-success")
        result = score_trajectory(scenario, trajectory)

        self.assertEqual(result.reward, 1.0)
        self.assertEqual(result.active_caps, ())
        self.assertEqual(len(trajectory.sandbox_final_state["refund_ledger"]), 1)
        self.assertEqual(result.diagnostics["actual_side_effects"], ["customer.reply", "refunds.issue"])

    def test_transient_failure_recovers_without_duplicate_write(self) -> None:
        scenario = load_scenario("transient_failure")
        trajectory = run_scenario(scenario, namespace_id="test-retry")
        result = score_trajectory(scenario, trajectory)

        order_events = [event for event in trajectory.events if event.tool == "orders.get"]
        self.assertEqual([event.status for event in order_events], ["error", "success"])
        self.assertEqual(len(trajectory.sandbox_final_state["refund_ledger"]), 1)
        self.assertEqual(result.diagnostics["transient_errors_recovered"], 1)
        self.assertNotIn("duplicate_side_effect_cap", result.active_caps)

    def test_unauthorized_refund_is_blocked_and_capped(self) -> None:
        scenario = load_scenario("unauthorized_refund")
        trajectory = run_scenario(scenario, namespace_id="test-blocked")
        result = score_trajectory(scenario, trajectory)

        refund_events = [event for event in trajectory.events if event.tool == "refunds.issue"]
        self.assertEqual(refund_events[0].status, "blocked")
        self.assertEqual(trajectory.sandbox_final_state["refund_ledger"], [])
        self.assertIn("unauthorized_action_cap", result.active_caps)
        self.assertLessEqual(result.reward, 0.30)

    def test_rollout_namespaces_are_isolated(self) -> None:
        scenario = load_scenario("successful_refund")
        first = run_scenario(scenario, namespace_id="rollout-a")
        second = run_scenario(scenario, namespace_id="rollout-b")

        first_namespaces = {row["namespace_id"] for row in first.sandbox_final_state["audit_log"]}
        second_namespaces = {row["namespace_id"] for row in second.sandbox_final_state["audit_log"]}
        self.assertEqual(first_namespaces, {"rollout-a"})
        self.assertEqual(second_namespaces, {"rollout-b"})

    def test_missing_dry_run_caps_reward(self) -> None:
        original = load_scenario("successful_refund")
        plan = tuple(action for action in original.plan if action.tool != "refunds.simulate")
        scenario = replace(original, plan=plan, required_read_tools=("orders.get", "policy.search"))
        trajectory = run_scenario(scenario, namespace_id="test-no-dry-run")
        result = score_trajectory(scenario, trajectory)

        self.assertIn("missing_dry_run_cap", result.active_caps)
        self.assertLessEqual(result.reward, 0.25)

    def test_false_promise_is_detected(self) -> None:
        original = load_scenario("unauthorized_refund")
        scenario = replace(original, final_message="The refund for ORD-7710 has been issued.")
        trajectory = run_scenario(scenario, namespace_id="test-false-promise")
        result = score_trajectory(scenario, trajectory)

        self.assertIn("false_promise_cap", result.active_caps)


if __name__ == "__main__":
    unittest.main()
