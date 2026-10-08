from __future__ import annotations

import unittest
from pathlib import Path

from training_smoke.common import (
    group_relative_advantages,
    load_jsonl,
    mask_prompt_tokens,
    validate_sft_record,
)
from training_smoke.reward_adapter import build_candidate_group


class TrainingSmokeContractTests(unittest.TestCase):
    def test_assistant_only_labels_mask_the_prompt(self) -> None:
        self.assertEqual(mask_prompt_tokens([10, 20, 30, 40], 2), [-100, -100, 30, 40])

    def test_group_relative_advantages_prefer_the_higher_reward(self) -> None:
        advantages = group_relative_advantages([1.0, 0.3])
        self.assertAlmostEqual(sum(advantages), 0.0)
        self.assertGreater(advantages[0], 0.0)
        self.assertLess(advantages[1], 0.0)

    def test_equal_rewards_have_zero_advantage(self) -> None:
        self.assertEqual(group_relative_advantages([0.5, 0.5]), [0.0, 0.0])

    def test_sft_examples_are_well_formed(self) -> None:
        data_path = (
            Path(__file__).resolve().parent.parent
            / "training_smoke"
            / "data"
            / "sft_examples.jsonl"
        )
        records = load_jsonl(data_path)
        self.assertGreaterEqual(len(records), 2)
        for record in records:
            validate_sft_record(record)

    def test_reward_adapter_scores_one_prompt_group(self) -> None:
        prompt, candidates = build_candidate_group()
        self.assertIn("ORD-1042", prompt)
        self.assertEqual([candidate.name for candidate in candidates], ["safe", "unsafe"])
        self.assertEqual(candidates[0].reward, 1.0)
        self.assertLess(candidates[1].reward, candidates[0].reward)
        self.assertIn("unauthorized_action_cap", candidates[1].active_caps)


if __name__ == "__main__":
    unittest.main()

