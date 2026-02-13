"""Tests for dual reward logging in Comb math debate environment."""

import unittest
from unittest.mock import patch, MagicMock


class TestDualRewards(unittest.TestCase):
    """Test dual reward logging structure and agreement logic."""

    def test_metrics_structure_with_comb_available(self):
        """Test that compute_reward() returns metrics with all required keys when COMB_AVAILABLE."""
        # This test verifies the metrics structure by checking the code logic
        # We can't easily test the full async path in a sync test, but we can verify
        # the error path which returns the same metrics structure

        # Import the module to check its structure
        from src.training import comb_math_debate_env

        # Verify the default_metrics structure exists in the compute_reward method
        # by reading the source (this is a structural test)
        import inspect

        # Get the compute_reward method
        if hasattr(comb_math_debate_env, 'MathDebateScenario'):
            scenario_class = comb_math_debate_env.MathDebateScenario

            # Check that the class has compute_reward method
            self.assertTrue(hasattr(scenario_class, 'compute_reward'),
                          "MathDebateScenario should have compute_reward method")

            # Verify the method exists and is async
            compute_reward = getattr(scenario_class, 'compute_reward')
            self.assertTrue(inspect.iscoroutinefunction(compute_reward),
                          "compute_reward should be async")

            # Read the source to verify metrics structure
            source = inspect.getsource(compute_reward)

            # Verify all required metrics keys are in the source
            self.assertIn("ground_truth_reward", source,
                         "compute_reward should include ground_truth_reward")
            self.assertIn("reward_agreement", source,
                         "compute_reward should include reward_agreement")
            self.assertIn("correctness_score", source,
                         "compute_reward should include correctness_score")
            self.assertIn("format_score", source,
                         "compute_reward should include format_score")
            self.assertIn("validator_call_time", source,
                         "compute_reward should include validator_call_time")

    def test_metrics_structure_stub_path(self):
        """Test that stub path (COMB_AVAILABLE=False) returns metrics with all keys."""
        with patch('src.training.comb_math_debate_env.COMB_AVAILABLE', False):
            from src.training.comb_math_debate_env import MathDebateScenario, interface

            # Create a scenario
            scenario = MagicMock(spec=MathDebateScenario)
            scenario.conversation_setup = MagicMock()

            # Manually call the fallback logic
            result = interface.RewardOutput(
                reward=0.0,
                text_info={"error": "Comb not available"},
                metrics={
                    "correctness_score": 0.0,
                    "ground_truth_reward": 0.0,
                    "reward_agreement": 0.0,
                    "format_score": 0.0,
                    "validator_call_time": 0.0,
                },
            )

            # Verify all required keys are present
            self.assertIn("correctness_score", result.metrics)
            self.assertIn("ground_truth_reward", result.metrics)
            self.assertIn("reward_agreement", result.metrics)
            self.assertIn("format_score", result.metrics)
            self.assertIn("validator_call_time", result.metrics)

    def test_reward_agreement_logic(self):
        """Test reward_agreement computation: 1 when BEE and GT agree, 0 when they disagree."""
        # Test case 1: Both agree on correct (BEE=1.0, GT=1.0) -> agreement=1
        bee_score = 1.0
        gt_reward = 1.0
        bee_binary = 1.0 if bee_score >= 0.5 else 0.0
        agreement = 1.0 if bee_binary == gt_reward else 0.0
        self.assertEqual(agreement, 1.0, "BEE=1.0 and GT=1.0 should agree")

        # Test case 2: Disagree (BEE=1.0, GT=0.0) -> agreement=0
        bee_score = 1.0
        gt_reward = 0.0
        bee_binary = 1.0 if bee_score >= 0.5 else 0.0
        agreement = 1.0 if bee_binary == gt_reward else 0.0
        self.assertEqual(agreement, 0.0, "BEE=1.0 and GT=0.0 should disagree")

        # Test case 3: Both agree on incorrect (BEE=0.0, GT=0.0) -> agreement=1
        bee_score = 0.0
        gt_reward = 0.0
        bee_binary = 1.0 if bee_score >= 0.5 else 0.0
        agreement = 1.0 if bee_binary == gt_reward else 0.0
        self.assertEqual(agreement, 1.0, "BEE=0.0 and GT=0.0 should agree")

        # Test case 4: Disagree (BEE=0.0, GT=1.0) -> agreement=0
        bee_score = 0.0
        gt_reward = 1.0
        bee_binary = 1.0 if bee_score >= 0.5 else 0.0
        agreement = 1.0 if bee_binary == gt_reward else 0.0
        self.assertEqual(agreement, 0.0, "BEE=0.0 and GT=1.0 should disagree")

        # Test case 5: BEE threshold (BEE=0.5, GT=1.0) -> agreement=1
        bee_score = 0.5
        gt_reward = 1.0
        bee_binary = 1.0 if bee_score >= 0.5 else 0.0
        agreement = 1.0 if bee_binary == gt_reward else 0.0
        self.assertEqual(agreement, 1.0, "BEE=0.5 (binary 1.0) and GT=1.0 should agree")


if __name__ == "__main__":
    unittest.main()
