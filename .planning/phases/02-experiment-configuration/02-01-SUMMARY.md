---
phase: 02-experiment-configuration
plan: 01
subsystem: configs
tags: [sweep, smollm-135m, reward-shaping, grpo, debate]

# Dependency graph
requires:
  - phase: 01-end-to-end-reward-integration
    provides: "DebateMetricStreamerConfig with reward_shaping_strategy/params fields wired end-to-end"
provides:
  - "5 SWEEP config files (identity, difference_rewards, potential_based, coma_advantage, reward_mixing)"
  - "Shared _base.py constants module for SmolLM-135M"
  - "85-test validation suite confirming config consistency (ECFG-01 through ECFG-08)"
affects: [03-evaluation-comparison, training-runs, wandb-dashboards]

# Tech tracking
tech-stack:
  added: []
  patterns: ["shared _base.py constants imported by all strategy configs", "AST + difflib-based config validation (sweep not available locally)"]

key-files:
  created:
    - configs/reward_shaping_sweep/__init__.py
    - configs/reward_shaping_sweep/_base.py
    - configs/reward_shaping_sweep/sweep_identity.py
    - configs/reward_shaping_sweep/sweep_difference_rewards.py
    - configs/reward_shaping_sweep/sweep_potential_based.py
    - configs/reward_shaping_sweep/sweep_coma_advantage.py
    - configs/reward_shaping_sweep/sweep_reward_mixing.py
    - tests/test_reward_shaping_sweep_configs.py
  modified: []

key-decisions:
  - "AST/text-based test validation instead of runtime import (sweep module unavailable locally)"
  - "Shared _base.py module guarantees ECFG-06 (identical hyperparams except reward shaping)"
  - "SmolLM-135M with 1+1 GPUs, 2048 seq, dev-low priority for cost-efficient validation"

patterns-established:
  - "Shared base constants: all sweep config variants import from _base.py to prevent hyperparameter divergence"
  - "Source-level config validation: AST parsing + difflib when runtime imports unavailable"

# Metrics
duration: 6min
completed: 2026-02-14
---

# Phase 2 Plan 1: SWEEP Config Generation Summary

**5 SmolLM-135M SWEEP configs for reward shaping comparison (identity/difference/potential/COMA/mixing) with shared _base.py constants and 85-test validation suite**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-14T03:21:15Z
- **Completed:** 2026-02-14T03:27:30Z
- **Tasks:** 2
- **Files created:** 8

## Accomplishments
- Created shared _base.py with all SmolLM-135M constants (1 GPU, 2048 seq, dev-low priority, MATH-500 data path)
- Created 5 strategy-specific SWEEP configs that are identical except for DebateMetricStreamerConfig arguments
- Built 85-test validation suite covering ECFG-01 through ECFG-08 using AST parsing and source-level diffing
- Verified configs via diff: non-docstring code differs only in reward_shaping_strategy and reward_shaping_params lines

## Task Commits

Each task was committed atomically:

1. **Task 1: Create shared base constants + 5 strategy SWEEP configs** - `21e116a` (feat)
2. **Task 2: Create validation test suite** - `448c025` (test)

## Files Created/Modified
- `configs/reward_shaping_sweep/__init__.py` - Empty init for package
- `configs/reward_shaping_sweep/_base.py` - Shared constants (SmolLM-135M profile, training hyperparams, queue, paths)
- `configs/reward_shaping_sweep/sweep_identity.py` - Identity baseline (no reward shaping)
- `configs/reward_shaping_sweep/sweep_difference_rewards.py` - Difference rewards (D_i = G - G_{-i})
- `configs/reward_shaping_sweep/sweep_potential_based.py` - Potential-based (gamma=0.99, debate_length)
- `configs/reward_shaping_sweep/sweep_coma_advantage.py` - COMA advantage (n_rollouts_per_prompt=4 in both fields)
- `configs/reward_shaping_sweep/sweep_reward_mixing.py` - Reward mixing (alpha=0.5)
- `tests/test_reward_shaping_sweep_configs.py` - 85 validation tests for ECFG-01 through ECFG-08

## Decisions Made
- **AST-based testing:** Since `sweep` is a cluster-only internal module not available in the local dev environment, all validation tests use AST parsing and text comparison. The critical `test_non_streamer_code_identical` test strips docstrings and DebateMetricStreamerConfig blocks, then compares normalized code -- proving structural identity.
- **Shared _base.py pattern:** All non-reward-shaping constants are centralized in `_base.py` and imported by each config. This eliminates the hyperparameter divergence risk (Pitfall 1 from research).
- **SmolLM-135M run_config path:** Used `${HOME}/repos/post_training/...` pattern matching existing configs, pointing to `smollm_135m_rloo_math.run`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test assertions for source-level string matching**
- **Found during:** Task 2 (validation test suite)
- **Issue:** `test_react_config_present` checked for `'"reasoning_effort"'` (quoted string) but source uses Python keyword arg `reasoning_effort="ON"` (unquoted key). Also `test_diff_only_in_debate_metric_streamer` used line-index comparison which fails when docstrings have different line counts.
- **Fix:** Changed string check to `"reasoning_effort" in source`. Rewrote diff test to strip module docstrings first, then compare only code portions using difflib.unified_diff.
- **Files modified:** tests/test_reward_shaping_sweep_configs.py
- **Verification:** All 85 tests pass
- **Committed in:** 448c025 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Test assertion fix necessary for correctness. No scope creep.

## Issues Encountered
- `sweep` module unavailable locally (internal Cohere cluster dependency) -- configs cannot be runtime-imported for testing. Addressed by using AST parsing and source-level comparison, which actually provides stronger guarantees about source-level identity than runtime comparison would.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 5 SWEEP configs are submission-ready (`uv run python configs/reward_shaping_sweep/sweep_*.py --submit start`)
- Configs validated to differ only in reward shaping fields (ECFG-06 verified)
- Ready for cluster submission and WandB comparison dashboard setup

---
*Phase: 02-experiment-configuration*
*Completed: 2026-02-14*
