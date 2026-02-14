# Testing Patterns

**Analysis Date:** 2026-02-14

## Test Framework

**Runner:**
- pytest 7.2.1 (from `pyproject.toml` dependency-groups)
- pytest-asyncio >=0.25 for async test support
- Config: No separate pytest.ini - configuration in `pyproject.toml` if present

**Assertion Library:**
- pytest built-in assertions
- numpy.testing for array comparisons
- pytest.approx for floating-point comparisons

**Run Commands:**
```bash
pytest                           # Run all tests
pytest -v                        # Verbose mode
pytest -k "test_name"            # Run specific test
pytest tests/test_pipeline.py    # Run single file
python -m pytest                 # Alternative invocation
```

## Test File Organization

**Location:**
- All tests in `tests/` directory at project root
- Co-located with source in `src/`, not nested within it

**Naming:**
- Pattern: `test_<module>.py`
- Examples: `test_pipeline.py`, `test_reward_shaping.py`, `test_cli.py`, `test_difference_rewards.py`

**Structure:**
```
tests/
├── conftest.py              # Shared fixtures
├── test_pipeline.py         # Pipeline orchestration tests
├── test_cli.py              # CLI integration tests
├── test_reward_shaping.py   # Reward shaping unit tests
├── test_difference_rewards.py
├── test_reward_mixing.py
└── test_*.py                # More test files
```

## Test Structure

**Suite Organization:**
```python
"""Unit tests for difference rewards shaping strategy.

TDD RED phase: Tests define expected behavior for DifferenceRewardShaper
which computes per-agent marginal contribution D_i = G(z) - G(z_{-i}).
"""

from __future__ import annotations

import numpy as np
import pytest


class TestDifferenceRewardsAllCorrect:
    """When G(z) is high and removing any agent drops reward to 0."""

    def test_difference_rewards_all_correct(self):
        """When G(z) = 5.0 and all counterfactuals G(z_{-i}) = 0.0,
        each agent gets D_i = 5.0 - 0.0 = 5.0 (all agents essential)."""
        from src.training.reward_shaping.difference_rewards import (
            DifferenceRewardShaper,
        )

        shaper = DifferenceRewardShaper()
        rewards = np.array([5.0])
        trajectory_metadata = [...]

        result = shaper.shape_rewards(rewards, None, trajectory_metadata)

        assert isinstance(result, dict)
        np.testing.assert_array_almost_equal(result["solver"], np.array([5.0]))
```

**Patterns:**
- Module docstring describes what's being tested and TDD phase
- Test classes group related scenarios
- Test class docstrings describe the scenario context
- Test function docstrings explain expected behavior with example values
- Imports inside test functions for isolation (optional pattern)

## Mocking

**Framework:** unittest.mock (standard library)

**Patterns:**
```python
from unittest.mock import AsyncMock, MagicMock, patch

# Patching LLM calls to avoid API costs
with patch("src.orchestration.pipeline.LLMClient") as mock_llm_client_cls:
    mock_client = MagicMock()
    mock_client.generate = AsyncMock()
    mock_llm_client_cls.return_value = mock_client

    # Set up return values
    mock_client.generate.side_effect = [
        (mock_solver_response, mock_token_usage),
        (mock_verif_response, mock_token_usage),
        (mock_judgment, mock_token_usage),
    ]

    # Run test
    result = await pipeline.run(problem_description="Test")
```

**What to Mock:**
- LLM API calls (LLMClient) - always mocked to avoid costs
- External API calls
- File I/O when testing logic (use tmp_path fixture otherwise)

**What NOT to Mock:**
- Pydantic models and data structures
- Pure functions without side effects
- Internal business logic being tested

## Fixtures and Factories

**Test Data:**
```python
# In conftest.py
@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    """Create a minimal pipeline.yaml in tmp_path.

    Args:
        tmp_path: pytest tmp_path fixture

    Returns:
        Path to the config file
    """
    config_data = {
        "config_version": "test-v1.0",
        "max_iterations": 3,
        "solver": {...},
        "verifier": {...},
        "judge": {...},
    }

    config_path = tmp_path / "test_pipeline.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    return config_path

@pytest.fixture
def mock_litellm():
    """Patch litellm.acompletion to return canned responses."""
    with patch("src.orchestration.pipeline.LLMClient") as mock_llm_client:
        # Setup mock...
        yield mock_llm_client
```

**Location:**
- Shared fixtures in `tests/conftest.py`
- Test-specific fixtures in individual test files
- Use pytest's `tmp_path` fixture for temporary directories

## Coverage

**Requirements:** Not enforced (no coverage config detected in pyproject.toml)

**View Coverage:**
```bash
pytest --cov=src --cov-report=html
pytest --cov=src --cov-report=term
```

## Test Types

**Unit Tests:**
- Pure function tests without mocks
- Example: `tests/test_reward_shaping.py` - tests mathematical formulas
- Pattern: Test single function with various inputs, no dependencies

**Integration Tests:**
- Test multiple components together with mocks for external dependencies
- Example: `tests/test_pipeline.py` - tests full pipeline with mocked LLM
- Example: `tests/test_cli.py` - tests CLI runner with mocked LLM
- Pattern: Mock external boundaries (LLM API), test real internal interactions

**E2E Tests:**
- Not detected in current test suite
- Would require real API credentials and would be slow/expensive

## Common Patterns

**Async Testing:**
```python
@pytest.mark.asyncio
async def test_pipeline_completes_successfully(mock_config, mock_token_usage, tmp_path):
    """Test that pipeline completes with mocked agents."""
    mock_config.trajectory_output_dir = str(tmp_path / "trajectories")

    # Setup mocks...

    pipeline = SolverVerifierJudgePipeline(mock_config)
    result = await pipeline.run(problem_description="What is the meaning of life?")

    assert isinstance(result, PipelineResult)
    assert result.iterations == 1
```

**Error Testing:**
```python
def test_invalid_format():
    """Test that invalid format raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_data("invalid.xyz")
```

**Parametrized Testing:**
```python
@pytest.mark.parametrize("mode", ["quality", "margin"])
def test_turn_0_uses_raw_reward(mode):
    """Test that turn 0 always uses raw reward (no shaping)."""
    shaped = apply_reward_shaping(
        raw_rewards=raw_rewards,
        mode=mode,
        alpha=1.0,
        beta=0.0,
    )
    assert shaped[0, 0] == 0.8
```

**Fixture Usage:**
```python
@pytest.fixture
def mock_config():
    """Create a test pipeline configuration."""
    return PipelineConfig(
        solver=AgentConfig(...),
        verifier=AgentConfig(...),
        judge=JudgeConfig(...),
        max_iterations=5,
    )

def test_example(mock_config, tmp_path):
    """Test uses fixtures injected by pytest."""
    pipeline = Pipeline(mock_config)
    # Test logic...
```

**File I/O Testing:**
```python
async def test_trajectory_output(tmp_config, mock_litellm, tmp_path):
    """Test that trajectory JSONL file is created with correct entries."""
    result = await run_single(
        config_path=str(tmp_config),
        problem="Test problem",
    )

    # Verify file exists
    trajectory_path = Path(result.trajectory_path)
    assert trajectory_path.exists()

    # Read and validate
    entries = []
    with open(trajectory_path) as f:
        for line in f:
            entries.append(json.loads(line))

    assert len(entries) >= 3
```

**NumPy Array Testing:**
```python
def test_reward_mixing_alpha_1_returns_global():
    """Test that alpha=1.0 returns global reward."""
    shaper = RewardMixingShaper(alpha=1.0)
    rewards = np.array([5.0, 0.0, 5.0])

    result = shaper.shape_rewards(rewards, None, trajectory_metadata)

    assert isinstance(result, dict)
    np.testing.assert_array_almost_equal(result["solver"], rewards)
    np.testing.assert_array_almost_equal(result["verifier"], rewards)
```

**Mock Side Effects:**
```python
# Multiple return values from same mock
mock_client.generate.side_effect = [
    (solver_response_1, token_usage),  # First call
    (verif_response_1, token_usage),   # Second call
    (solver_response_2, token_usage),  # Third call
    (verif_response_2, token_usage),   # Fourth call
]
```

## Test Documentation

**Docstring Pattern:**
- Every test has a docstring
- Format: One-line summary, then optional details with concrete values
- Example:
```python
def test_quality_mode_formula(self):
    """Test quality mode formula: Q*R - (1-Q)*(1-R)."""
    # ... test code with inline calculation comments
```

**TDD Markers:**
- Some test files include TDD phase in module docstring
- Example: `"""TDD RED phase: Tests define expected behavior for RewardMixingShaper"""`

---

*Testing analysis: 2026-02-14*
