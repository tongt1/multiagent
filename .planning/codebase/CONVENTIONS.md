# Coding Conventions

**Analysis Date:** 2026-02-14

## Naming Patterns

**Files:**
- Snake_case for Python modules: `solver.py`, `llm_client.py`, `dataset_loader.py`
- Test files prefixed with `test_`: `test_pipeline.py`, `test_reward_shaping.py`
- Configuration files use descriptive names: `config.py`, `training_config.py`

**Functions:**
- Snake_case: `generate()`, `shape_rewards()`, `config_hash()`, `apply_reward_shaping()`
- Async functions prefixed with `async def`: `async def run()`, `async def generate()`

**Variables:**
- Snake_case: `problem_description`, `solver_response`, `token_usage`, `trajectory_metadata`
- Constants use UPPER_CASE: Not widely used in this codebase

**Types:**
- PascalCase for classes: `SolverAgent`, `PipelineConfig`, `RewardShaper`, `TokenUsage`
- Response models follow pattern: `SolverResponse`, `VerificationResult`, `PipelineResult`
- Config classes suffixed with `Config`: `AgentConfig`, `JudgeConfig`, `RewardShapingConfig`

## Code Style

**Formatting:**
- Tool: Ruff (version >=0.8.5)
- Line length: 100 characters
- Target Python version: 3.11
- Configuration: `pyproject.toml` lines 45-50

**Linting:**
- Tool: Ruff with mypy for type checking
- Mypy strict mode enabled (`strict = true` in `pyproject.toml`)
- Ruff rules enabled: E (pycodestyle errors), F (pyflakes), I (isort), N (pep8-naming), W (pycodestyle warnings), UP (pyupgrade)
- Config file: `pyproject.toml`

## Import Organization

**Order:**
1. Future imports: `from __future__ import annotations`
2. Standard library: `import json`, `from pathlib import Path`, `from typing import Any, Optional`
3. Third-party packages: `import numpy as np`, `from pydantic import BaseModel`, `from loguru import logger`
4. Local imports: `from src.agents import SolverAgent`, `from src.models.config import PipelineConfig`

**Pattern observed in `src/orchestration/pipeline.py`:**
```python
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel

from src.agents import JudgeAgent, SolverAgent, VerifierAgent
from src.infrastructure.cost_tracker import CostTracker
```

**Path Aliases:**
- Use `src.` prefix for all internal imports
- No path aliases configured

## Error Handling

**Patterns:**
- Retry logic via `tenacity` library for LLM calls (see `src/infrastructure/llm_client.py`)
- Retry decorator: `@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))`
- ValueError for invalid configuration: `raise ValueError(f"SolverAgent requires role='solver', got '{config.role}'")`
- No custom exception classes detected - uses standard Python exceptions

## Logging

**Framework:** loguru

**Patterns:**
- Import: `from loguru import logger`
- Info level for progress: `logger.info(f"Starting pipeline run: {run_id}")`
- Info level for iterations: `logger.info(f"=== Iteration {iteration} ===")`
- Used throughout orchestration and infrastructure modules

## Comments

**When to Comment:**
- Module-level docstrings at top of every file
- Inline comments for non-obvious logic or phase markers: `# DEBATE MODE EXECUTION BELOW`, `# SOLVER STEP`
- TDD phase annotations in test docstrings: `"""TDD RED phase: Tests define expected behavior..."""`

**Docstrings:**
- Triple-quoted strings: `"""`
- Google-style format with Args/Returns sections
- All public functions and classes have docstrings
- Example from `src/agents/solver.py`:
```python
async def generate(
    self,
    problem_description: str,
    feedback: str | None = None,
) -> tuple[SolverResponse, TokenUsage]:
    """Generate a solution to the problem.

    Args:
        problem_description: The problem to solve
        feedback: Optional feedback from previous iteration

    Returns:
        Tuple of (solver response, token usage)
    """
```

## Function Design

**Size:** Functions are focused and single-purpose, typically 20-60 lines

**Parameters:**
- Use type hints for all parameters: `def __init__(self, config: AgentConfig, llm_client: LLMClient) -> None:`
- Optional parameters use `| None` union syntax (Python 3.10+): `feedback: str | None = None`
- Dict types use `dict[str, Any]` notation

**Return Values:**
- Always type-annotated
- Tuple returns for multiple values: `tuple[SolverResponse, TokenUsage]`
- Async functions return coroutines: `async def run() -> PipelineResult:`

## Module Design

**Exports:**
- Explicit `__init__.py` files that re-export key classes
- Example from `src/agents/__init__.py`:
```python
from src.agents.judge import JudgeAgent
from src.agents.solver import SolverAgent
from src.agents.verifier import VerifierAgent
```

**Barrel Files:**
- Used in `src/agents/__init__.py`, `src/models/__init__.py`, `src/data/__init__.py`
- Pattern: Import from submodules and re-export via `__all__`

## Type Annotations

**Pattern:**
- All functions have complete type annotations
- Pydantic models for structured data: `class SolverResponse(BaseModel):`
- Use `from __future__ import annotations` for forward references
- Literal types for enums: `Literal["solver", "verifier"]`, `Literal["debate", "baseline"]`
- Field validators from Pydantic: `Field(ge=0.0, le=1.0)` for confidence bounds

## Async/Await Usage

**Pattern:**
- Async functions for all LLM calls and pipeline operations
- Use `async with` for context managers: `async with TrajectoryLogger(...) as traj:`
- Await pattern: `solver_response, solver_usage = await self.solver.generate(...)`

## Testing Conventions

**Test Organization:**
- Tests organized into classes by scenario: `class TestRewardMixingAlphaOne:`, `class TestDifferenceRewardsAllCorrect:`
- Test class docstrings describe the scenario: `"""Alpha=1.0: pure global reward."""`
- Test function names describe behavior: `test_reward_mixing_alpha_1_returns_global()`

**Assertion Style:**
- Direct assertions: `assert result.iterations == 1`
- pytest.approx for floats: `assert shaped[0, 1] == pytest.approx(1.5)`
- numpy testing for arrays: `np.testing.assert_array_almost_equal(result["solver"], rewards)`
- isinstance checks: `assert isinstance(result, PipelineResult)`

---

*Convention analysis: 2026-02-14*
