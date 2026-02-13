"""Unit tests for debate metric computation functions."""

from __future__ import annotations

import numpy as np
import pytest

from src.training.wandb_enrichment.debate_metrics import (
    compute_all_scalar_metrics,
    compute_per_role_kl,
    compute_per_role_rewards,
    compute_zero_advantage_metrics,
)
from src.training.wandb_enrichment.metric_schema import (
    METRIC_FRAC_ZERO_STD,
    METRIC_FRAC_ZERO_STD_CORRECT,
    METRIC_FRAC_ZERO_STD_INCORRECT,
    METRIC_KL_JUDGE,
    METRIC_KL_SOLVER,
    METRIC_KL_VERIFIER,
    METRIC_MEAN_REWARD_STD,
    METRIC_REWARD_JUDGE,
    METRIC_REWARD_SOLVER,
    METRIC_REWARD_VERIFIER,
)


class TestComputePerRoleRewards:
    """Tests for compute_per_role_rewards function."""

    def test_compute_per_role_rewards_basic(self):
        """Test basic per-role reward computation with 2 samples per role."""
        # 6 rewards: 2 per role
        rewards = np.array([0.5, 1.0, 0.0, 0.8, 1.0, 0.2])
        role_labels = ["solver", "solver", "verifier", "verifier", "judge", "judge"]

        result = compute_per_role_rewards(rewards, role_labels)

        # Verify correct means
        assert result[METRIC_REWARD_SOLVER] == pytest.approx(0.75)  # (0.5 + 1.0) / 2
        assert result[METRIC_REWARD_VERIFIER] == pytest.approx(0.4)  # (0.0 + 0.8) / 2
        assert result[METRIC_REWARD_JUDGE] == pytest.approx(0.6)  # (1.0 + 0.2) / 2

    def test_compute_per_role_rewards_missing_role(self):
        """Test that missing roles are omitted from result dict."""
        # Only solver and verifier, no judge
        rewards = np.array([0.5, 1.0, 0.0, 0.8])
        role_labels = ["solver", "solver", "verifier", "verifier"]

        result = compute_per_role_rewards(rewards, role_labels)

        # Verify solver and verifier present, judge absent
        assert METRIC_REWARD_SOLVER in result
        assert METRIC_REWARD_VERIFIER in result
        assert METRIC_REWARD_JUDGE not in result

    def test_compute_per_role_rewards_empty(self):
        """Test that empty arrays return empty dict."""
        rewards = np.array([])
        role_labels = []

        result = compute_per_role_rewards(rewards, role_labels)

        assert result == {}

    def test_compute_per_role_rewards_length_mismatch(self):
        """Test handling of mismatched lengths."""
        rewards = np.array([0.5, 1.0, 0.8])
        role_labels = ["solver", "verifier"]  # Too short

        result = compute_per_role_rewards(rewards, role_labels)

        # Should return empty dict and log warning
        assert result == {}


class TestComputeZeroAdvantage:
    """Tests for compute_zero_advantage_metrics function."""

    def test_compute_zero_advantage_all_different(self):
        """Test with all different rewards (no zero-std prompts)."""
        # 8 different rewards (1 prompt)
        rewards = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        result = compute_zero_advantage_metrics(rewards, n_rollouts_per_prompt=8)

        # No zero-std prompts
        assert result[METRIC_FRAC_ZERO_STD] == pytest.approx(0.0)
        assert result[METRIC_FRAC_ZERO_STD_CORRECT] == pytest.approx(0.0)
        assert result[METRIC_FRAC_ZERO_STD_INCORRECT] == pytest.approx(0.0)
        # Mean std should be positive
        assert result[METRIC_MEAN_REWARD_STD] > 0.0

    def test_compute_zero_advantage_all_same(self):
        """Test with all identical rewards (100% zero-std)."""
        # 8 identical rewards (1 prompt, all same)
        rewards = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        result = compute_zero_advantage_metrics(rewards, n_rollouts_per_prompt=8)

        # 100% zero-std
        assert result[METRIC_FRAC_ZERO_STD] == pytest.approx(1.0)
        # All correct
        assert result[METRIC_FRAC_ZERO_STD_CORRECT] == pytest.approx(1.0)
        assert result[METRIC_FRAC_ZERO_STD_INCORRECT] == pytest.approx(0.0)
        # Mean std should be ~0
        assert result[METRIC_MEAN_REWARD_STD] == pytest.approx(0.0, abs=1e-7)

    def test_compute_zero_advantage_mixed(self):
        """Test with mix of zero-std and varied prompts."""
        # 2 prompts: first all-same, second all-different
        rewards = np.array([
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # Prompt 1: all same
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,  # Prompt 2: all different
        ])

        result = compute_zero_advantage_metrics(rewards, n_rollouts_per_prompt=8)

        # 50% zero-std (1 out of 2 prompts)
        assert result[METRIC_FRAC_ZERO_STD] == pytest.approx(0.5)
        assert result[METRIC_FRAC_ZERO_STD_CORRECT] == pytest.approx(0.5)
        assert result[METRIC_FRAC_ZERO_STD_INCORRECT] == pytest.approx(0.0)

    def test_compute_zero_advantage_correct_vs_incorrect(self):
        """Test breakdown of all-correct vs all-incorrect zero-std prompts."""
        # 3 prompts: all correct, all incorrect, mixed
        rewards = np.array([
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # Prompt 1: all correct
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Prompt 2: all incorrect
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,  # Prompt 3: varied
        ])

        result = compute_zero_advantage_metrics(rewards, n_rollouts_per_prompt=8)

        # 2 out of 3 prompts have zero-std
        assert result[METRIC_FRAC_ZERO_STD] == pytest.approx(2/3)
        # 1 correct, 1 incorrect
        assert result[METRIC_FRAC_ZERO_STD_CORRECT] == pytest.approx(1/3)
        assert result[METRIC_FRAC_ZERO_STD_INCORRECT] == pytest.approx(1/3)

    def test_compute_zero_advantage_truncation(self):
        """Test that partial groups are truncated with warning."""
        # 10 rewards but n_rollouts_per_prompt=8 -> should truncate to 8
        rewards = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5])

        result = compute_zero_advantage_metrics(rewards, n_rollouts_per_prompt=8)

        # Should only use first 8 rewards (1 prompt)
        assert result[METRIC_FRAC_ZERO_STD] == pytest.approx(1.0)

    def test_compute_zero_advantage_insufficient_data(self):
        """Test with fewer rewards than n_rollouts_per_prompt."""
        # Only 4 rewards but need 8
        rewards = np.array([0.5, 0.8, 1.0, 0.2])

        result = compute_zero_advantage_metrics(rewards, n_rollouts_per_prompt=8)

        # Should return all zeros
        assert result[METRIC_FRAC_ZERO_STD] == pytest.approx(0.0)
        assert result[METRIC_FRAC_ZERO_STD_CORRECT] == pytest.approx(0.0)
        assert result[METRIC_FRAC_ZERO_STD_INCORRECT] == pytest.approx(0.0)
        assert result[METRIC_MEAN_REWARD_STD] == pytest.approx(0.0)


class TestComputePerRoleKL:
    """Tests for compute_per_role_kl function."""

    def test_compute_per_role_kl_basic(self):
        """Test basic per-role KL computation with simple masks."""
        # 2 sequences, 4 tokens each
        kl_per_token = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ])

        # Solver gets first 2 tokens, verifier gets last 2 tokens
        role_masks = {
            "solver": np.array([
                [True, True, False, False],
                [True, False, False, False],
            ]),
            "verifier": np.array([
                [False, False, True, True],
                [False, True, True, False],
            ]),
        }

        result = compute_per_role_kl(kl_per_token, role_masks)

        # Solver: (0.1 + 0.2 + 0.5) / 3 = 0.8 / 3
        assert result[METRIC_KL_SOLVER] == pytest.approx(0.8 / 3)
        # Verifier: (0.3 + 0.4 + 0.6 + 0.7) / 4 = 2.0 / 4
        assert result[METRIC_KL_VERIFIER] == pytest.approx(0.5)
        # Judge not present
        assert METRIC_KL_JUDGE not in result

    def test_compute_per_role_kl_empty_masks(self):
        """Test that empty role_masks returns empty dict."""
        kl_per_token = np.array([[0.1, 0.2], [0.3, 0.4]])
        role_masks = {}

        result = compute_per_role_kl(kl_per_token, role_masks)

        assert result == {}

    def test_compute_per_role_kl_none_masks(self):
        """Test that None role_masks returns empty dict."""
        kl_per_token = np.array([[0.1, 0.2], [0.3, 0.4]])
        role_masks = None

        result = compute_per_role_kl(kl_per_token, role_masks)

        assert result == {}

    def test_compute_per_role_kl_zero_mask(self):
        """Test that role with no True values is omitted."""
        kl_per_token = np.array([[0.1, 0.2], [0.3, 0.4]])

        role_masks = {
            "solver": np.array([[True, True], [True, True]]),
            "verifier": np.array([[False, False], [False, False]]),  # No tokens
        }

        result = compute_per_role_kl(kl_per_token, role_masks)

        # Solver should be present
        assert METRIC_KL_SOLVER in result
        # Verifier should be omitted (no tokens)
        assert METRIC_KL_VERIFIER not in result

    def test_compute_per_role_kl_all_roles(self):
        """Test with all three roles present."""
        kl_per_token = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])

        role_masks = {
            "solver": np.array([[True, True, False, False, False, False]]),
            "verifier": np.array([[False, False, True, True, False, False]]),
            "judge": np.array([[False, False, False, False, True, True]]),
        }

        result = compute_per_role_kl(kl_per_token, role_masks)

        # All three roles should be present
        assert METRIC_KL_SOLVER in result
        assert METRIC_KL_VERIFIER in result
        assert METRIC_KL_JUDGE in result


class TestComputeAllScalarMetrics:
    """Tests for compute_all_scalar_metrics convenience function."""

    def test_compute_all_scalar_metrics_integration(self):
        """Test that all metrics are computed and merged correctly."""
        rewards = np.array([0.5, 1.0, 0.0, 0.8, 1.0, 0.2, 0.7, 0.9])
        role_labels = ["solver", "solver", "verifier", "verifier", "judge", "judge", "solver", "verifier"]

        kl_per_token = np.array([[0.1, 0.2]] * 8)
        role_masks = {
            "solver": np.array([[True, False]] * 8),
            "verifier": np.array([[False, True]] * 8),
        }

        result = compute_all_scalar_metrics(
            rewards=rewards,
            role_labels=role_labels,
            n_rollouts_per_prompt=8,
            kl_per_token=kl_per_token,
            role_masks=role_masks,
        )

        # Should have per-role rewards
        assert METRIC_REWARD_SOLVER in result
        assert METRIC_REWARD_VERIFIER in result
        assert METRIC_REWARD_JUDGE in result

        # Should have zero-advantage metrics
        assert METRIC_FRAC_ZERO_STD in result
        assert METRIC_MEAN_REWARD_STD in result

        # Should have per-role KL
        assert METRIC_KL_SOLVER in result
        assert METRIC_KL_VERIFIER in result

    def test_compute_all_scalar_metrics_without_kl(self):
        """Test that KL metrics are skipped when kl_per_token is None."""
        rewards = np.array([0.5, 1.0, 0.0, 0.8])
        role_labels = ["solver", "solver", "verifier", "verifier"]

        result = compute_all_scalar_metrics(
            rewards=rewards,
            role_labels=role_labels,
            n_rollouts_per_prompt=4,
            kl_per_token=None,
            role_masks=None,
        )

        # Should have per-role rewards and zero-advantage
        assert METRIC_REWARD_SOLVER in result
        assert METRIC_FRAC_ZERO_STD in result

        # Should NOT have KL metrics
        assert METRIC_KL_SOLVER not in result
        assert METRIC_KL_VERIFIER not in result


class TestMetricKeyPrefixes:
    """Test that all returned metric keys use the debate/ prefix."""

    def test_metric_keys_use_debate_prefix(self):
        """Verify all returned dict keys start with 'debate/'."""
        # Test per-role rewards
        rewards = np.array([0.5, 1.0])
        role_labels = ["solver", "verifier"]
        result1 = compute_per_role_rewards(rewards, role_labels)
        for key in result1:
            assert key.startswith("debate/"), f"Key {key} missing debate/ prefix"

        # Test zero-advantage
        rewards = np.array([1.0] * 8)
        result2 = compute_zero_advantage_metrics(rewards)
        for key in result2:
            assert key.startswith("debate/"), f"Key {key} missing debate/ prefix"

        # Test per-role KL
        kl = np.array([[0.1, 0.2]])
        masks = {"solver": np.array([[True, False]])}
        result3 = compute_per_role_kl(kl, masks)
        for key in result3:
            assert key.startswith("debate/"), f"Key {key} missing debate/ prefix"

        # Test all metrics
        result4 = compute_all_scalar_metrics(
            rewards=np.array([0.5, 1.0, 0.8, 0.9]),
            role_labels=["solver", "verifier", "judge", "solver"],
            n_rollouts_per_prompt=4,
        )
        for key in result4:
            assert key.startswith("debate/"), f"Key {key} missing debate/ prefix"
