from __future__ import annotations

import json
import unittest

from agentic_rl_demo.models import Scenario
from agentic_rl_demo.plan_parser import parse_completion
from public_experiment.generate_data import _make_case
from public_experiment.protocol import serialize_target
from public_experiment.reward import score_completion


class PublicExperimentTests(unittest.TestCase):
    def test_serialized_target_round_trips_through_parser(self) -> None:
        import random

        value = _make_case(0, random.Random(17))
        scenario = Scenario.from_dict(value)
        parsed = parse_completion(serialize_target(scenario))

        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.actions, scenario.plan)
        self.assertEqual(parsed.final_message, scenario.final_message)

    def test_parser_rejects_text_outside_protocol(self) -> None:
        parsed = parse_completion(
            'thinking aloud <tool_call>{"name":"orders.get","arguments":{}}</tool_call>'
            '<final>Done.</final>'
        )
        self.assertFalse(parsed.valid)
        self.assertIn("text_outside_protocol", parsed.errors)

    def test_reference_eligible_plan_receives_full_reward(self) -> None:
        import random

        value = _make_case(2, random.Random(23))
        scenario = Scenario.from_dict(value)
        metrics = score_completion(serialize_target(scenario), json.dumps(value))

        self.assertEqual(metrics["reward"], 1.0)
        self.assertTrue(metrics["task_completed"])
        self.assertTrue(metrics["policy_compliant"])

    def test_reference_ineligible_plan_does_not_issue_refund(self) -> None:
        import random

        value = _make_case(3, random.Random(23))
        scenario = Scenario.from_dict(value)
        metrics = score_completion(serialize_target(scenario), value)

        self.assertEqual(metrics["reward"], 1.0)
        self.assertFalse(metrics["unauthorized_action"])
        self.assertTrue(metrics["grounded_response"])


if __name__ == "__main__":
    unittest.main()

