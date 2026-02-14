# Phase 2: Experiment Configuration - Research

**Researched:** 2026-02-14
**Domain:** SWEEP config generation for reward shaping comparison experiment (SmolLM-135M)
**Confidence:** HIGH

## Summary

Phase 2 requires creating 5 SWEEP config files that are identical except for the `reward_shaping_strategy` and `reward_shaping_params` fields in `DebateMetricStreamerConfig`. The existing codebase has all the building blocks: a working debate SWEEP config template (`configs/sweep_math_debate_grpo.py`), a SmolLM-135M model profile (`configs/model_profiles.py`), a SmolLM-135M run config (`run_configs/smollm_135m_rloo_math.run`), and the reward shaping strategies fully wired end-to-end (Phase 1 complete).

The primary design decision is whether to create 5 separate Python files (one per strategy) or a single Python file that generates 5 sweep trials via `get_search_space()`. Analysis of the codebase shows the existing configs each return `[{}]` (single trial) from `get_search_space()`. The requirements demand that "diffing any two configs shows differences only in reward_shaping_strategy and reward_shaping_params fields," which strongly favors a single base config with a shared constants block, generating 5 separate files via a scripted or templated approach. However, creating 5 separate files from a common base provides the clearest audit trail and allows each config to be submitted independently.

The most important deviations from the existing 7B debate config are: (1) SmolLM-135M needs `num_training_gpus=1`, `num_sampling_gpus=1`, `max_sequence_length=2048`, and its own GCS checkpoint path; (2) the queue should use `post_training_cohere_labs_queue` with `dev-low` priority (existing configs use `dev-high`); (3) the `run_config` path must point to the SmolLM run config; (4) `DebateMetricStreamerConfig` must have its `reward_shaping_strategy` and `reward_shaping_params` uncommented and set per-strategy.

**Primary recommendation:** Create a single base config Python file with shared constants, then generate 5 strategy-specific SWEEP config files that import the shared base and differ only in `DebateMetricStreamerConfig` parameters. Use a validation test to confirm all 5 configs are identical except for reward shaping fields.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| sweep | (internal) | SWEEP config framework for job submission | Used by all existing SWEEP configs in configs/ |
| post_training.canonicals.sweep_base | (internal) | Base sweep class and PostTraining config | Standard base class for all SWEEP configs |
| post_training.flink | (internal) | Flink training pipeline components | All actors, learners, samplers, streamers |
| pydantic | >=2.12 | Config validation via ComponentBase/FlinkActorOutputStreamerConfig | Used by DebateMetricStreamerConfig |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | >=8.2 | Validation tests for config generation | Config correctness tests |
| difflib | stdlib | Programmatic diffing of config dicts | Verification that configs differ only in reward shaping |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| 5 separate .py files | Single file with get_search_space returning 5 trials | Single file is DRY-er but harder to audit and submit individually; separate files match existing repo convention |
| Python-based config generation | YAML/JSON configs | SWEEP framework requires Python class definitions; YAML would need a wrapper |
| Script to generate 5 files from template | Manual copy-paste | Script ensures consistency; manual risks divergence |

**Installation:** No new dependencies required. All needed libraries are already available.

## Architecture Patterns

### Recommended Approach: Base Constants Module + 5 Strategy Config Files

The pattern that best satisfies the requirements (ECFG-06: identical hyperparameters except reward shaping) is a shared constants module imported by all 5 configs.

```
configs/
  reward_shaping_sweep/
    __init__.py
    _base.py                           # Shared constants: model, GPU, batch, LR, etc.
    sweep_identity.py                  # Strategy: identity (no params)
    sweep_difference_rewards.py        # Strategy: difference_rewards (no params)
    sweep_potential_based.py           # Strategy: potential_based (gamma=0.99, debate_length)
    sweep_coma_advantage.py            # Strategy: coma_advantage (n_rollouts_per_prompt=4)
    sweep_reward_mixing.py             # Strategy: reward_mixing (alpha=0.5)
tests/
  test_reward_shaping_sweep_configs.py # Validation test
```

### Pattern 1: Shared Base Constants

**What:** All 5 configs import shared constants from `_base.py`. The constants block covers all hyperparameters, model paths, queue settings, and infrastructure config.

**When to use:** Always -- this is the mechanism that guarantees ECFG-06 (identical hyperparams except reward shaping).

**Example:**

```python
# configs/reward_shaping_sweep/_base.py

from configs.model_profiles import SMOLLM_135M

# Model profile
_PROFILE = SMOLLM_135M
NUM_TRAINING_GPUS = _PROFILE.num_training_gpus     # 1
NUM_SAMPLING_GPUS = _PROFILE.num_sampling_gpus      # 1
MAX_SEQUENCE_LENGTH = _PROFILE.max_sequence_length  # 2048
CKPT_PATH = _PROFILE.ckpt_path  # gs://cohere-dev-central-2/users/roman_cohere_com/smollm-135M/megazord_weights/ckpt-0

# Training hyperparameters (shared across all 5 configs)
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
TOTAL_TRAIN_STEPS = 15         # SmolLM-135M: very fast iteration
LEARNING_RATE = 3e-6
KL_BETA = 0.03
GENERATIONS_PER_PROMPT = 4     # GRPO group size
EXPORT_EVERY_STEPS = 5
HARD_UPDATE_REF_EVERY_STEPS = 5
SEED = 42
N_GRADIENT_ACCUMULATION_STEPS = 1

# Queue and priority
QUEUE = "post_training_cohere_labs_queue"
PRIORITY_CLASS = "dev-low"

# Data path
MATH_500_DATA_PATH = "gs://cohere-dev-central-2/comb/data/math_500/2025_05_15/scenarios_train.jsonl"

# Run config
RUN_CONFIG_PATH = "smollm_135m_rloo_math.run"
```

### Pattern 2: Per-Strategy Config File (Minimal Diff)

**What:** Each strategy file imports base constants and only specifies the `DebateMetricStreamerConfig` with strategy-specific params.

**When to use:** Each of the 5 config files follows this pattern.

**Example:**

```python
# configs/reward_shaping_sweep/sweep_coma_advantage.py
"""SWEEP config for COMA advantage reward shaping (SmolLM-135M, MATH-500)."""

from configs.reward_shaping_sweep._base import *  # All shared constants
# ... standard imports ...

class RewardShapingSweep(sweep_base.Sweep):
    # ... identical to other configs except:
    actor_outputs_streamers=[
        FilteringStreamerConfig(...),  # identical
        DebateMetricStreamerConfig(
            n_rollouts_per_prompt=GENERATIONS_PER_PROMPT,
            reward_shaping_strategy="coma_advantage",
            reward_shaping_params={"n_rollouts_per_prompt": GENERATIONS_PER_PROMPT},
        ),
    ],
```

### Pattern 3: Config Validation Test

**What:** A test that loads all 5 configs, serializes them to dicts, and verifies they differ only in `reward_shaping_strategy` and `reward_shaping_params`.

**When to use:** Required for ECFG-06 verification.

**Example:**

```python
def test_configs_differ_only_in_reward_shaping():
    """All 5 configs are identical except reward_shaping_strategy and reward_shaping_params."""
    configs = load_all_five_configs()
    for i, config_a in enumerate(configs):
        for j, config_b in enumerate(configs):
            if i >= j:
                continue
            diff = deep_diff(config_a, config_b)
            # Only reward_shaping_strategy and reward_shaping_params should differ
            for key in diff:
                assert "reward_shaping" in key, f"Unexpected diff: {key}"
```

### Anti-Patterns to Avoid
- **Copy-pasting entire config files:** Leads to hyperparameter divergence when one file is updated but not others. Use shared constants instead.
- **Using the 7B debate config as base:** The 7B config uses 8+16 GPUs, 8192 seq length, `c3_7B_12-2024_command_release` checkpoint. SmolLM-135M uses 1+1 GPUs, 2048 seq length, GCS checkpoint. Mixing these up wastes GPU resources and may fail.
- **Hardcoding strategy params in the base:** The base should have NO reward shaping config. Each strategy file adds its own.
- **Using `dev-high` priority:** Requirements specify `dev-low` for shared cluster resources. The existing 7B configs use `dev-high` -- do not copy this.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Config generation | Manual 5-file copy-paste | Shared base module + per-strategy files | Guarantees ECFG-06 (identical hyperparams) |
| Strategy validation | Custom strategy name checking | Registry's `create_strategy_from_config()` | Already validates at load time |
| SmolLM-135M GPU/seq settings | Manual constants | `configs/model_profiles.SMOLLM_135M` | Already defined with correct values |
| MATH-500 data path | Hardcoded string | Reference from `run_configs/smollm_135m_rloo_math.run` | Single source of truth for data location |
| Queue/priority specification | Custom string | `sweep.Queue.post_training_cohere_labs_queue` with `priority_class="dev-low"` | Existing enum ensures valid queue name |

**Key insight:** The infrastructure for SWEEP configs is mature. The task is primarily about correctly composing existing components with SmolLM-135M settings and per-strategy `DebateMetricStreamerConfig` variations.

## Common Pitfalls

### Pitfall 1: Hyperparameter Divergence Between Configs
**What goes wrong:** Changing a hyperparameter in one config file but not the others invalidates the apples-to-apples comparison (violates ECFG-06).
**Why it happens:** Without a shared constants module, each file has its own copy of constants that can drift independently.
**How to avoid:** All non-reward-shaping hyperparameters must come from a single shared source (`_base.py`). A validation test should enforce this.
**Warning signs:** Diff between any two configs shows changes outside reward_shaping fields.

### Pitfall 2: Wrong Model Profile (7B Instead of SmolLM-135M)
**What goes wrong:** Config uses 8 training GPUs and 16 sampling GPUs for a 135M model that fits on 1 GPU. Job either wastes resources or fails due to incorrect mesh configuration.
**Why it happens:** Existing debate configs (`sweep_math_debate_grpo.py`) target the 7B model. Copy-pasting without adjusting GPU counts and checkpoint path is a common error.
**How to avoid:** Import `SMOLLM_135M` from `configs/model_profiles.py` and use its `num_training_gpus`, `num_sampling_gpus`, `max_sequence_length`, and `ckpt_path`. The profile has: 1 training GPU, 1 sampling GPU, 2048 seq length, GCS checkpoint.
**Warning signs:** `partition=gpu_8` or `gpu_16` in a SmolLM config. Checkpoint path containing "c3_7B" or "command_release".

### Pitfall 3: n_rollouts_per_prompt Mismatch
**What goes wrong:** `DebateMetricStreamerConfig.n_rollouts_per_prompt` does not match `GENERATIONS_PER_PROMPT` in the loss config, causing incorrect zero-advantage metrics and incorrect COMA advantage computation.
**Why it happens:** `GENERATIONS_PER_PROMPT` is used in two places: the GRPO loss config (`preference.generations_per_prompt`) and `DebateMetricStreamerConfig.n_rollouts_per_prompt`. If they differ, the streamer computes metrics over wrong group boundaries.
**How to avoid:** Both must use the same `GENERATIONS_PER_PROMPT` constant from the shared base.
**Warning signs:** Unexpected zero-advantage fraction or COMA advantages that don't sum to zero within groups.

### Pitfall 4: coma_advantage n_rollouts_per_prompt Param vs Config Field
**What goes wrong:** COMA advantage strategy takes `n_rollouts_per_prompt` as both a `DebateMetricStreamerConfig` field AND a `reward_shaping_params` dict entry. These serve different purposes but must be consistent.
**Why it happens:** `DebateMetricStreamerConfig.n_rollouts_per_prompt` controls group-level metric computation (zero-advantage). `reward_shaping_params={"n_rollouts_per_prompt": N}` controls COMA's counterfactual baseline computation. If N differs between them, advantage computation uses wrong group boundaries.
**How to avoid:** Set both to `GENERATIONS_PER_PROMPT`. Document this dual-use.
**Warning signs:** COMA advantages that look plausible per-group but wrong overall.

### Pitfall 5: Queue Name Does Not Match Enum
**What goes wrong:** Using a string like `"flex_queue"` instead of `sweep.Queue.post_training_cohere_labs_queue` causes job submission failure.
**Why it happens:** The requirements mention "post-training flex queue" but the actual sweep enum is `sweep.Queue.post_training_cohere_labs_queue`. The term "flex queue" is informal.
**How to avoid:** Use `sweep.Queue.post_training_cohere_labs_queue` exactly as in existing configs. Change `priority_class` to `"dev-low"` to control resource allocation.
**Warning signs:** Import error or submission rejection mentioning unknown queue.

### Pitfall 6: Missing env_name_remap for Debate Environment
**What goes wrong:** Without `env_name_remap={"math": "math_debate"}`, the training pipeline uses the single-speaker "math" environment instead of the multi-turn debate environment. Reward shaping becomes meaningless.
**Why it happens:** The SmolLM run config (`smollm_135m_rloo_math.run`) specifies data paths for MATH-500 but does not configure the Comb environment remap. This must be done in the SWEEP config's `CombItemsPreprocessorConfig`.
**How to avoid:** Always include `env_name_remap={"math": "math_debate"}` in the `CombItemsPreprocessorConfig`, exactly as in `sweep_math_debate_grpo.py`.
**Warning signs:** No role labels in WandB metrics, no debate transcript structure in generation logs.

### Pitfall 7: SmolLM-135M Needs Different vLLM Worker Config
**What goes wrong:** SmolLM-135M fits on 1 GPU but the sidecar config uses `--gpus-per-vllm-worker=1` with `partition=gpu_16`. The 135M model only needs 1 GPU total for sampling.
**Why it happens:** Existing debate config allocates 16 GPUs for vLLM workers to serve the 7B model across multiple workers for throughput.
**How to avoid:** Set `NUM_SAMPLING_GPUS = 1` (from model profile), `partition=gpu_1`, and keep `--gpus-per-vllm-worker=1`.
**Warning signs:** Job pending for a long time waiting for 16 GPUs when only 1 is needed.

## Code Examples

### SmolLM-135M DebateMetricStreamerConfig Variations

The 5 configurations of `DebateMetricStreamerConfig` for each strategy:

```python
# Source: DebateMetricStreamerConfig fields from src/training/wandb_enrichment/debate_streamer.py

# 1. Identity (baseline)
DebateMetricStreamerConfig(
    n_rollouts_per_prompt=GENERATIONS_PER_PROMPT,
    # No reward_shaping_strategy -> defaults to identity
)

# 2. Difference rewards
DebateMetricStreamerConfig(
    n_rollouts_per_prompt=GENERATIONS_PER_PROMPT,
    reward_shaping_strategy="difference_rewards",
    reward_shaping_params={},
)

# 3. Potential-based (debate_length potential, gamma=0.99)
DebateMetricStreamerConfig(
    n_rollouts_per_prompt=GENERATIONS_PER_PROMPT,
    reward_shaping_strategy="potential_based",
    reward_shaping_params={"gamma": 0.99, "potential_type": "debate_length"},
)

# 4. COMA advantage (n_rollouts_per_prompt=4)
DebateMetricStreamerConfig(
    n_rollouts_per_prompt=GENERATIONS_PER_PROMPT,
    reward_shaping_strategy="coma_advantage",
    reward_shaping_params={"n_rollouts_per_prompt": GENERATIONS_PER_PROMPT},
)

# 5. Reward mixing (alpha=0.5)
DebateMetricStreamerConfig(
    n_rollouts_per_prompt=GENERATIONS_PER_PROMPT,
    reward_shaping_strategy="reward_mixing",
    reward_shaping_params={"alpha": 0.5},
)
```

### SmolLM-135M Key Config Differences vs 7B

```python
# Source: configs/model_profiles.py SMOLLM_135M vs sweep_math_debate_grpo.py constants

# SmolLM-135M (this phase)         | 7B (existing configs)
# -------------------------------- | --------------------------------
# NUM_TRAINING_GPUS = 1            | NUM_TRAINING_GPUS = 8
# NUM_SAMPLING_GPUS = 1            | NUM_SAMPLING_GPUS = 16
# MAX_SEQUENCE_LENGTH = 2048       | MAX_SEQUENCE_LENGTH = 4096
# CKPT_PATH = "gs://...smollm..."  | CKPT_PATH = "c3_7B_12-2024_command_release"
# TRAIN_BATCH_SIZE = 2             | TRAIN_BATCH_SIZE = 8
# TOTAL_TRAIN_STEPS = 15           | TOTAL_TRAIN_STEPS = 300
# run_config = smollm_135m_...run  | run_config = rloo_7B_math.run
# partition = gpu_1                | partition = gpu_8 / gpu_16
# priority_class = "dev-low"       | priority_class = "dev-high"
```

### SWEEP Config Class Pattern (Per-Strategy File)

```python
# Source: Pattern derived from configs/sweep_math_debate_grpo.py adapted for SmolLM-135M

"""SWEEP config for [strategy] reward shaping (SmolLM-135M, MATH-500)."""
from __future__ import annotations
from collections.abc import Iterable

import configs._image_reuse  # noqa: F401
import sweep

from post_training.canonicals import sweep_base
from post_training.flink import flink_zord
from post_training.flink.components import flink_reward_model
from post_training.flink.components.flink_comb_actor import FlinkCombActorConfig
from post_training.flink.components.flink_eval import FlinkEvalConfig
from post_training.flink.components.flink_input_data_preprocessors import CombItemsPreprocessorConfig
from post_training.flink.components.flink_learner_rloo import FlinkRlooLearnerConfig
from post_training.flink.components.flink_learning_filter import FilterMode, FilterMultiplexerConfig
from post_training.flink.components.flink_learning_filter.filter_on_identical_reward import FilterOnIdenticalRewardConfig
from post_training.flink.components.flink_learning_filter.filter_on_truncated import FilterOnTruncatedConfig
from post_training.flink.components.flink_learning_filter.filtering_streamer import FilteringStreamerConfig
from post_training.flink.components.flink_sampler_vllm_sidecar import FlinkVllmSidecarSamplerConfig
from post_training.flink.utils.endpoint_resolver import EndpointResolverConfig
from post_training.flink.components.debate_enrichment import DebateMetricStreamerConfig

# Import shared base constants
from configs.reward_shaping_sweep._base import (
    NUM_TRAINING_GPUS, NUM_SAMPLING_GPUS, MAX_SEQUENCE_LENGTH, CKPT_PATH,
    TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, TOTAL_TRAIN_STEPS, LEARNING_RATE,
    KL_BETA, GENERATIONS_PER_PROMPT, EXPORT_EVERY_STEPS,
    HARD_UPDATE_REF_EVERY_STEPS, SEED, N_GRADIENT_ACCUMULATION_STEPS,
    PRIORITY_CLASS, RUN_CONFIG_PATH,
)

class RewardShapingSweep(sweep_base.Sweep):
    settings = sweep.SweepSettings(
        sweep_output_path="${HOME}/sweep_jobs/reward_shaping_comparison/",
        cluster=sweep.Cluster.cw_us_east_04_prod,
    )
    fax = sweep_base.PostTraining(
        partition=f"gpu_{NUM_TRAINING_GPUS}",
        queue=sweep.Queue.post_training_cohere_labs_queue,
        jobs_max_fanout=1,
        wandb_project="multiagent-debate-rl",
        priority_class=PRIORITY_CLASS,
        run_config=RUN_CONFIG_PATH,
        ckpt_path=CKPT_PATH,
        # ... patch_run_config with shared hyperparams ...
        # ... DebateMetricStreamerConfig with strategy-specific params ...
    )

    def get_search_space(self) -> Iterable[sweep.ParamOverrides]:
        return [{}]
```

### Validation Test Pattern

```python
# Source: test pattern for ECFG-06 verification

def test_all_configs_share_hyperparams():
    """ECFG-06: All 5 configs share identical hyperparameters except reward shaping."""
    from configs.reward_shaping_sweep._base import (
        NUM_TRAINING_GPUS, NUM_SAMPLING_GPUS, TRAIN_BATCH_SIZE,
        TOTAL_TRAIN_STEPS, LEARNING_RATE, KL_BETA, GENERATIONS_PER_PROMPT,
        SEED, CKPT_PATH, PRIORITY_CLASS,
    )

    # Verify all configs reference the same base constants
    assert NUM_TRAINING_GPUS == 1  # SmolLM-135M
    assert NUM_SAMPLING_GPUS == 1
    assert PRIORITY_CLASS == "dev-low"


def test_strategy_params_correct():
    """ECFG-03/04/05: Strategy-specific params set correctly."""
    # potential_based: debate_length potential with gamma=0.99
    # coma_advantage: n_rollouts_per_prompt=4
    # reward_mixing: alpha=0.5
```

## Key Architectural Findings

### Finding 1: SmolLM-135M Model Profile Exists (HIGH confidence)
`configs/model_profiles.py` defines `SMOLLM_135M` with all needed settings:
- `ckpt_path = "gs://cohere-dev-central-2/users/roman_cohere_com/smollm-135M/megazord_weights/ckpt-0"`
- `num_training_gpus = 1`
- `num_sampling_gpus = 1`
- `max_sequence_length = 2048`
- `needs_mesh_override = False`

### Finding 2: SmolLM-135M Run Config Exists (HIGH confidence)
`run_configs/smollm_135m_rloo_math.run` is a complete run config with MATH-500 data paths:
- `data_dir_dict` points to `gs://cohere-dev-central-2/comb/data/math_500/2025_05_15/scenarios_train.jsonl`
- `train_batch_size = 2`, `eval_batch_size = 2`
- `total_train_steps = 15` (very fast iteration)
- `generations_per_prompt = 4` (GRPO group size)
- `reward_model.name = "dummyrewardmodel"` (reward from Comb environment)
- `sharding.n_tensor_parallel = 1` (SmolLM fits on 1 GPU)

### Finding 3: DebateMetricStreamerConfig Accepts Strategy Fields (HIGH confidence)
From Phase 1, `DebateMetricStreamerConfig` has:
- `reward_shaping_strategy: str = ""` -- empty defaults to identity
- `reward_shaping_params: dict = {}` -- strategy-specific params
- `n_rollouts_per_prompt: int = 8` -- default is 8 but should match GENERATIONS_PER_PROMPT

These fields are currently commented out in `sweep_math_debate_grpo.py` (lines 213-214). Phase 2 uncomments and sets them per-strategy.

### Finding 4: Queue and Priority Configuration (HIGH confidence)
All existing SWEEP configs use `sweep.Queue.post_training_cohere_labs_queue`. The term "flex queue" in requirements refers to this queue. Priority is controlled by `priority_class` parameter:
- Existing 7B configs: `priority_class="dev-high"`
- Phase 2 requirement: `priority_class="dev-low"` (shared cluster resources, cost-conscious)

### Finding 5: COMA n_rollouts_per_prompt Dual-Use (HIGH confidence)
The COMA advantage strategy requires `n_rollouts_per_prompt` in its `reward_shaping_params` dict. This MUST match `GENERATIONS_PER_PROMPT` (used in both loss config and `DebateMetricStreamerConfig.n_rollouts_per_prompt`). The SmolLM run config uses `generations_per_prompt = 4`, so all three must be 4.

### Finding 6: existing sweep_math_debate_grpo.py Sidecar Config (HIGH confidence)
The sidecar configuration pattern in existing configs includes vLLM worker setup with `--gpus-per-vllm-worker=1`. For SmolLM-135M with 1 sampling GPU, the sidecar partition should be `gpu_1` with a single worker. The model fits on 1 GPU so no multi-worker parallelism is needed.

### Finding 7: Debate Environment Remap Required (HIGH confidence)
The `CombItemsPreprocessorConfig` must include `env_name_remap={"math": "math_debate"}` to route MATH-500 data items through the debate environment. The SmolLM run config does NOT include this remap (it is a generic RLOO math config). The SWEEP config's `patch_run_config` must add this remap, exactly as done in `sweep_math_debate_grpo.py`.

### Finding 8: Run Config Reference Path (MEDIUM confidence)
The existing SWEEP configs reference the run_config as `"${HOME}/repos/post_training/post_training/experimental/comb_flink/configs/rloo_7B_math.run"` -- a production Flink run config. For SmolLM-135M, we need to either: (a) use the local `run_configs/smollm_135m_rloo_math.run` which already has correct data paths, batch sizes, and model metadata, or (b) use the SmolLM run config path appropriate for the cluster environment. The run_config path must be resolvable on the training cluster, not just locally. If the SmolLM run config is committed to the same repo that gets checked out on the cluster, a relative path or repo-relative path should work.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Reward shaping params commented out in SWEEP config | Uncommented and set per-strategy | This phase | Enables actual reward shaping comparison |
| 7B model for all experiments | SmolLM-135M for fast iteration | This project | Minutes instead of hours per run |
| dev-high priority on flex queue | dev-low priority for comparison experiments | This phase | Reduced cluster cost, longer queue times acceptable |

**Current state (before this phase):**
- Reward shaping is wired end-to-end (Phase 1 complete)
- No strategy-specific SWEEP configs exist for SmolLM-135M
- Existing 7B debate config has reward shaping fields commented out
- SmolLM-135M model profile and run config exist but no SWEEP config uses them

**After this phase:**
- 5 SWEEP config files ready for submission
- All share identical hyperparameters except reward shaping
- Each targets SmolLM-135M on flex queue with dev-low priority and MATH-500 data
- Validation test confirms config consistency

## Open Questions

1. **SmolLM-135M Run Config Path on Cluster**
   - What we know: `run_configs/smollm_135m_rloo_math.run` exists locally with correct MATH-500 data paths.
   - What's unclear: Whether the SWEEP framework expects an absolute cluster path (like `${HOME}/repos/post_training/...`) or can use a repo-relative path. Existing configs use absolute paths with `${HOME}/repos/post_training/` prefix.
   - Recommendation: Use the same `${HOME}/repos/` pattern as existing configs, pointing to the SmolLM run config. If the run config is not available at that cluster path, it may need to be placed in the post_training repo or inlined as `patch_run_config` overrides.

2. **GENERATIONS_PER_PROMPT Value for SmolLM-135M**
   - What we know: SmolLM run config uses `generations_per_prompt = 4`. Existing 7B debate config uses `GENERATIONS_PER_PROMPT = 4`. Requirements say `coma_advantage` uses `n_rollouts_per_prompt=4`.
   - What's unclear: Whether 4 is optimal for SmolLM-135M (smaller model may benefit from more rollouts for variance reduction).
   - Recommendation: Use 4 to match requirements exactly. Tuning is out of scope per PROJECT.md ("Default strategy hyperparams -- no per-strategy tuning in v1").

3. **Batch Size and Train Steps for Meaningful Comparison**
   - What we know: SmolLM run config uses `train_batch_size=2` and `total_train_steps=15` (designed for quick validation). The requirements just say "ready to submit" without specifying training length.
   - What's unclear: Whether 15 steps is sufficient to observe meaningful reward shaping differences across strategies.
   - Recommendation: Use the SmolLM run config defaults (2 batch size, 15 steps) for initial validation. These can be bumped later without changing the config structure. The primary goal of Phase 2 is config correctness, not training convergence.

## Sources

### Primary (HIGH confidence)
- `configs/sweep_math_debate_grpo.py` -- Reference SWEEP config showing full structure, imports, sidecar config, and commented reward shaping params
- `configs/model_profiles.py` -- SmolLM-135M model profile with GPU/seq/ckpt settings
- `run_configs/smollm_135m_rloo_math.run` -- SmolLM run config with MATH-500 data paths and training parameters
- `src/training/wandb_enrichment/debate_streamer.py` -- DebateMetricStreamerConfig field definitions (reward_shaping_strategy, reward_shaping_params)
- `src/training/reward_shaping/__init__.py` -- Registry of 5 available strategies
- `tests/test_reward_shaping_integration.py` -- Test patterns for all 5 strategies with correct params

### Secondary (MEDIUM confidence)
- `configs/sweep_math_baseline_grpo.py` -- Shows how baseline (non-debate) config differs
- `configs/sweep_math_debate_multimodel_grpo.py` -- Shows multi-model metadata pattern
- `configs/experiments/experiment_config.py` -- ExperimentConfig pattern (not used directly but shows config generation pattern)

### Tertiary (LOW confidence)
- `sweep.Queue` enum values -- inferred from existing usage, cannot import `sweep` locally to enumerate
- `sweep.Cluster` values -- only `cw_us_east_04_prod` observed in existing configs

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All components already exist in the codebase; no new dependencies
- Architecture: HIGH - Pattern directly derived from existing SWEEP configs and model profile
- Pitfalls: HIGH - Based on direct code reading of existing configs, model profiles, and streamer config
- Queue/priority: HIGH - Existing configs demonstrate the exact pattern; only priority_class changes

**Research date:** 2026-02-14
**Valid until:** 2026-03-14 (stable -- code changes are internal to this repo)
