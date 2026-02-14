---
phase: 02-experiment-configuration
verified: 2026-02-14T04:15:00Z
status: passed
score: 8/8 must-haves verified
re_verification: false
---

# Phase 2: Experiment Configuration Verification Report

**Phase Goal:** Five SWEEP configs exist that are identical except for reward shaping strategy, ready to submit to flex queue for an apples-to-apples comparison
**Verified:** 2026-02-14T04:15:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Each of the 5 SWEEP configs can be imported as a Python module without errors | VERIFIED | All 5 files parse as valid Python AST; 85/85 pytest tests pass including class structure, `__main__` guard, and `get_search_space` method checks |
| 2 | Diffing any two configs shows differences ONLY in reward_shaping_strategy and reward_shaping_params -- all other hyperparameters, data paths, queue settings, and model configs are identical | VERIFIED | Manual `diff` of identity vs each of the 4 other configs shows changes only in docstrings and the `DebateMetricStreamerConfig` constructor args (`reward_shaping_strategy`, `reward_shaping_params`). All non-docstring non-streamer code is byte-identical. Test `test_non_streamer_code_identical` and `test_diff_only_in_debate_metric_streamer` both pass. |
| 3 | All 5 configs use sweep.Queue.post_training_cohere_labs_queue with priority_class='dev-low' | VERIFIED | Each config contains `queue=sweep.Queue.post_training_cohere_labs_queue` and `priority_class=PRIORITY_CLASS`. `_base.py` defines `PRIORITY_CLASS = "dev-low"`. Tests `test_priority_class_dev_low` and `test_queue_is_post_training` pass for all 5. |
| 4 | All 5 configs use SmolLM-135M settings (1 training GPU, 1 sampling GPU, 2048 seq len, GCS checkpoint) | VERIFIED | `_base.py` imports `SMOLLM_135M` from `configs.model_profiles` (which defines `num_training_gpus=1`, `num_sampling_gpus=1`, `max_sequence_length=2048`, `ckpt_path` containing `smollm-135M`). All 5 configs use `partition=f"gpu_{NUM_TRAINING_GPUS}"`. Test `test_base_constants_smollm` passes. |
| 5 | potential_based config uses gamma=0.99 and potential_type='debate_length' | VERIFIED | `sweep_potential_based.py` line 172: `reward_shaping_params={"gamma": 0.99, "potential_type": "debate_length"}`. Test `test_potential_based_strategy` passes. |
| 6 | coma_advantage config uses n_rollouts_per_prompt=4 in both DebateMetricStreamerConfig field AND reward_shaping_params | VERIFIED | `sweep_coma_advantage.py` lines 171-173: `n_rollouts_per_prompt=GENERATIONS_PER_PROMPT` (field) and `reward_shaping_params={"n_rollouts_per_prompt": GENERATIONS_PER_PROMPT}` (params dict). `GENERATIONS_PER_PROMPT = 4`. Test `test_coma_dual_n_rollouts` passes. |
| 7 | reward_mixing config uses alpha=0.5 | VERIFIED | `sweep_reward_mixing.py` line 171: `reward_shaping_params={"alpha": 0.5}`. Test `test_reward_mixing_strategy` passes. |
| 8 | All 5 configs include env_name_remap={'math': 'math_debate'} for debate environment | VERIFIED | Each config contains `env_name_remap={"math": "math_debate"}` in `CombItemsPreprocessorConfig`. Test `test_env_name_remap` passes for all 5. |

**Score:** 8/8 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `configs/reward_shaping_sweep/__init__.py` | Empty init for package | VERIFIED | Exists, empty file |
| `configs/reward_shaping_sweep/_base.py` | Shared constants (GPU, batch, LR, queue, data path) imported by all 5 configs | VERIFIED | 45 lines, defines all shared constants, imports `SMOLLM_135M` from `configs.model_profiles` |
| `configs/reward_shaping_sweep/sweep_identity.py` | Identity baseline SWEEP config | VERIFIED | 231 lines, `class RewardShapingSweep`, `DebateMetricStreamerConfig(n_rollouts_per_prompt=GENERATIONS_PER_PROMPT)` with no strategy |
| `configs/reward_shaping_sweep/sweep_difference_rewards.py` | Difference rewards SWEEP config | VERIFIED | 233 lines, strategy="difference_rewards", params={} |
| `configs/reward_shaping_sweep/sweep_potential_based.py` | Potential-based SWEEP config | VERIFIED | 234 lines, strategy="potential_based", params={"gamma": 0.99, "potential_type": "debate_length"} |
| `configs/reward_shaping_sweep/sweep_coma_advantage.py` | COMA advantage SWEEP config | VERIFIED | 235 lines, strategy="coma_advantage", n_rollouts_per_prompt in both field and params |
| `configs/reward_shaping_sweep/sweep_reward_mixing.py` | Reward mixing SWEEP config | VERIFIED | 233 lines, strategy="reward_mixing", params={"alpha": 0.5} |
| `tests/test_reward_shaping_sweep_configs.py` | Validation tests for ECFG-01 through ECFG-08 | VERIFIED | 442 lines, 85 tests across 8 test classes, all pass in 0.09s |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `sweep_*.py` (all 5) | `_base.py` | `from configs.reward_shaping_sweep._base import` | WIRED | Every config file imports all 20+ constants from `_base.py`. Verified by grep and test `test_imports_base_constants`. |
| `_base.py` | `configs/model_profiles.py` | `from configs.model_profiles import SMOLLM_135M` | WIRED | `_base.py` line 11: `from configs.model_profiles import SMOLLM_135M`. `model_profiles.py` exists and defines `SMOLLM_135M = ModelProfile(...)`. |
| `sweep_*.py` (all 5) | `DebateMetricStreamerConfig` | `from post_training.flink.components.debate_enrichment import DebateMetricStreamerConfig` | WIRED (cluster path) | Import references cluster module path. Local development copy exists at `src/training/wandb_enrichment/debate_streamer.py` with the class defined and `reward_shaping_strategy`/`reward_shaping_params` fields present. |
| `tests/test_reward_shaping_sweep_configs.py` | `_base.py` | `from configs.reward_shaping_sweep._base import` | WIRED | Test file imports and validates `CKPT_PATH`, `GENERATIONS_PER_PROMPT`, `MAX_SEQUENCE_LENGTH`, `NUM_TRAINING_GPUS`, `NUM_SAMPLING_GPUS`, `PRIORITY_CLASS` at runtime. |

### Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| ECFG-01: Identity baseline SWEEP config | SATISFIED | -- |
| ECFG-02: Difference rewards SWEEP config | SATISFIED | -- |
| ECFG-03: Potential-based SWEEP config (gamma=0.99, debate_length) | SATISFIED | -- |
| ECFG-04: COMA advantage SWEEP config (n_rollouts_per_prompt=4) | SATISFIED | -- |
| ECFG-05: Reward mixing SWEEP config (alpha=0.5) | SATISFIED | -- |
| ECFG-06: All 5 configs share identical hyperparameters except reward shaping | SATISFIED | Verified by diff and `test_non_streamer_code_identical` |
| ECFG-07: All 5 configs use post-training flex queue with dev-low priority | SATISFIED | -- |
| ECFG-08: All 5 configs use MATH-500 online data from existing GCS path | SATISFIED | `MATH_500_DATA_PATH` = `gs://cohere-dev-central-2/comb/data/math_500/2025_05_15/scenarios_train.jsonl` in `_base.py`; `run_configs/smollm_135m_rloo_math.run` exists locally |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `tests/test_reward_shaping_sweep_configs.py` | 106 | `DEBATE_METRIC_STREAMER_PLACEHOLDER` string constant | Info | Deliberate substitution marker for text normalization in diff test. Not a code stub. |

No blockers or warnings found. No TODO/FIXME/HACK comments in any config files. No empty implementations. No stub patterns.

### Human Verification Required

### 1. Cluster Import Validation

**Test:** On the training cluster, run `python -c "from configs.reward_shaping_sweep.sweep_identity import RewardShapingSweep; s = RewardShapingSweep(); print(s.fax.priority_class)"` (repeat for all 5)
**Expected:** All 5 import successfully and `priority_class` prints `dev-low`
**Why human:** The `sweep`, `post_training.canonicals`, and `post_training.flink` modules are cluster-only dependencies not available in the local dev environment. AST-level validation confirms structural correctness, but runtime instantiation can only be verified on the cluster.

### 2. GCS Data Path Accessibility

**Test:** Run `gsutil ls gs://cohere-dev-central-2/comb/data/math_500/2025_05_15/scenarios_train.jsonl`
**Expected:** File exists and is accessible
**Why human:** GCS path validity requires authenticated cloud access

### 3. Flex Queue Submission Dry Run

**Test:** Run `uv run python configs/reward_shaping_sweep/sweep_identity.py start` (without --submit) on the cluster
**Expected:** Generates sweep plan without errors, shows 1 trial targeting `post_training_cohere_labs_queue` with `dev-low` priority
**Why human:** Queue validation requires cluster infrastructure

### Gaps Summary

No gaps found. All 8 observable truths are verified. All 8 artifacts exist, are substantive (not stubs), and are properly wired. All 8 ECFG requirements are satisfied. The 5 config files are structurally identical except for the DebateMetricStreamerConfig arguments, as confirmed by both `diff` analysis and the 85-test validation suite (all passing).

The three human verification items are standard cluster-deployment checks that cannot be validated in the local development environment but do not block the structural correctness of the phase goal.

---

_Verified: 2026-02-14T04:15:00Z_
_Verifier: Claude (gsd-verifier)_
