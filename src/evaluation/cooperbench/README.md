# CooperBench Integration

Integration of the [CooperBench](https://github.com/cooperbench/CooperBench) cooperative coding benchmark into our multiagent debate RL codebase.

## Overview

CooperBench (arXiv:2601.13295) is the first benchmark for evaluating AI agent cooperation on collaborative coding tasks. It contains 652 tasks across 12 repositories in 4 languages (Python, TypeScript, Go, Rust). In the cooperative setting, two agents each implement a separate feature in a shared codebase, then their patches are merged and tested together.

We map CooperBench's cooperative evaluation to our solver-verifier debate paradigm:
- **Solver agent** = Agent 1 (implements feature 1)
- **Verifier agent** = Agent 2 (implements feature 2)
- **Judge** = Patch merge + test execution in Docker sandbox
- **Debate rounds** = Approach summary exchange between agents

## Module Structure

```
src/evaluation/cooperbench/
  __init__.py      # Package exports + load_config() YAML adapter
  models.py        # Pydantic v2 data models (config, problem, result, etc.)
  loader.py        # Dataset discovery and loading from filesystem
  evaluator.py     # Docker sandbox patch merging and test execution
  reward.py        # RLVR-compatible reward functions (binary, partial, shaped)
  pipeline.py      # Solver-verifier debate cooperation pipeline
  runner.py        # Batch experiment runner with parallelism
  README.md        # This file

configs/
  cooperbench_default.yaml  # Default configuration

tests/
  test_cooperbench_models.py   # 37 model tests
  test_cooperbench_loader.py   # 25 loader tests
```

## Quick Start

### 1. Clone CooperBench dataset

```bash
git clone https://github.com/cooperbench/CooperBench.git repos/CooperBench
```

### 2. Load and inspect the dataset

```python
from src.evaluation.cooperbench.loader import load_cooperbench_dataset

# Load lite subset (100 feature pairs across 26 tasks)
problems = load_cooperbench_dataset(
    dataset_path="repos/CooperBench/dataset",
    subset="lite",
)
print(f"Loaded {len(problems)} problems")
for p in problems[:3]:
    print(f"  {p.repo}/{p.task_id} features={p.features}")
```

### 3. Load config from YAML

```python
from src.evaluation.cooperbench import load_config

config = load_config("configs/cooperbench_default.yaml")
print(config.mode, config.solver_model, config.max_rounds)
```

### 4. Run an experiment

```python
import asyncio
from src.evaluation.cooperbench import load_config
from src.evaluation.cooperbench.runner import CooperBenchExperimentRunner

config = load_config("configs/cooperbench_default.yaml")
config.dataset_path = "repos/CooperBench/dataset"
runner = CooperBenchExperimentRunner(config)
results = asyncio.run(runner.run_experiment())
print(f"Pass rate: {results.pass_rate:.1%}")
```

## Dataset Structure

CooperBench expects this filesystem layout:

```
dataset/
  subsets/
    lite.json          # 100 feature pairs from 26 tasks
    flash.json         # Larger subset
  <repo_name>/         # e.g., pillow_task, llama_index_task
    task<id>/          # e.g., task25, task17244
      Dockerfile       # Docker sandbox build file
      setup.sh         # Environment setup
      runner.sh        # Standardized test runner
      run_tests.sh     # Test execution script
      combined.patch   # Gold combined patch
      feature<N>/      # e.g., feature1, feature2
        feature.md     # Natural language feature specification
        feature.patch  # Gold implementation patch
        tests.patch    # Test cases patch
```

## Reward Functions

| Type | Function | Range | Description |
|------|----------|-------|-------------|
| Binary | `compute_cooperbench_reward` | {0, 1} | 1.0 iff both features pass. RLVR-compatible. |
| Partial | `compute_cooperbench_partial_reward` | {0, 0.5, 1} | 0.5 per passing feature. Denser signal. |
| Shaped | `compute_cooperbench_shaped_reward` | [0, 1] | Partial + merge bonus for clean merges. |

## Cooperation Pipeline (Coop Mode)

1. Solver gets feature 1 spec, generates patch + approach summary
2. Verifier gets feature 2 spec, generates patch + approach summary
3. Agents exchange approach summaries (rounds 2+)
4. Both refine patches with partner context
5. Patches merged (naive if non-overlapping files, union fallback)
6. Tests run in Docker sandbox per feature
7. Binary reward: 1.0 if both pass, 0.0 otherwise

## Configuration

See `configs/cooperbench_default.yaml` for all options. Key settings:
- `mode`: `coop` (2-agent) or `solo` (1-agent baseline)
- `dataset.subset`: `lite` (100 pairs), `flash`, or `null` (all)
- `cooperation.max_rounds`: Number of approach exchange iterations
- `evaluation.backend`: `docker` (implemented), `modal`/`gcp` (planned)

## Relation to Existing Debate RL

This module is **additive** — it does not modify existing code. It reuses:
- `src.infrastructure.llm_client.LLMClient` for LLM API calls
- Pydantic v2 patterns consistent with `src.models.*`
- loguru logging consistent with the rest of the codebase

The CooperBench reward functions can be integrated into the training loop as an alternative evaluation metric alongside MATH 500.
