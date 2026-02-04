---
phase: 05-wandb-logging-enrichment
plan: 02
subsystem: training-monitoring
tags: [wandb, metrics, grpo, gradient-norms, numpy, pytest]

# Dependency graph
requires:
  - phase: 05-01
    provides: Metric schema with debate/ namespace constants
provides:
  - Per-role reward computation (solver, verifier, judge)
  - Zero-advantage detection for GRPO training collapse
  - Per-role KL divergence aggregation
  - Gradient norm logging enabled in SWEEP configs
affects: [05-03-debug-data-writer, 06-hook-integration, 07-analysis]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Pure computation functions using numpy (no JAX dependencies)
    - Epsilon handling for numerical stability in division
    - Import metric constants from schema for all dict keys

key-files:
  created:
    - src/training/wandb_enrichment/debate_metrics.py
    - tests/test_debate_metrics.py
  modified:
    - configs/sweep_math_debate_grpo.py
    - configs/sweep_math_baseline_grpo.py

key-decisions:
  - "Zero-advantage detection reshapes to (n_prompts, n_rollouts_per_prompt) and truncates incomplete groups with warning"
  - "Per-role KL uses epsilon 1e-8 for numerical stability when normalizing by token count"
  - "Gradient norm logging via advanced_logging config (Flink built-in) rather than custom metric computation"
  - "All metric computation functions return dicts with debate/-prefixed keys from metric_schema constants"

patterns-established:
  - "Metric computation functions operate on numpy arrays for testability"
  - "Functions omit roles/metrics when data is absent rather than returning zeros"
  - "Comprehensive unit tests cover basic cases, edge cases, and empty inputs"

# Metrics
duration: 3min
completed: 2026-02-04
---

# Phase 5 Plan 2: Debate Metrics Computation Summary

**Per-role rewards, zero-advantage GRPO collapse detection, per-role KL divergence, and gradient norm logging via Flink advanced_logging config**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-04T18:29:02Z
- **Completed:** 2026-02-04T18:32:26Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Pure numpy computation functions for per-role rewards, zero-advantage detection, and per-role KL divergence
- 18 comprehensive unit tests covering basic cases, edge cases, and prefix validation
- Gradient norm logging enabled in both debate and baseline SWEEP configs via advanced_logging
- All metrics use debate/ namespace prefix from metric_schema constants

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement debate metric computation functions** - `1082023` (feat)
2. **Task 2: Enable gradient norm logging in SWEEP configs** - `b57708a` (feat)

**Plan metadata:** (will be created after STATE.md update)

## Files Created/Modified

- `src/training/wandb_enrichment/debate_metrics.py` - Pure numpy computation functions for per-role rewards, zero-advantage detection, and per-role KL
- `tests/test_debate_metrics.py` - 18 unit tests covering all computation functions and edge cases
- `configs/sweep_math_debate_grpo.py` - Added advanced_logging config for gradient norm tracking
- `configs/sweep_math_baseline_grpo.py` - Added advanced_logging config for gradient norm tracking

## Decisions Made

1. **Zero-advantage truncation strategy**: When batch size is not divisible by n_rollouts_per_prompt (GRPO group size), truncate to last complete group and log warning via loguru rather than padding or erroring.

2. **Per-role KL epsilon**: Use 1e-8 epsilon when normalizing by token count to avoid division by zero when a role has no tokens in the batch.

3. **Gradient norm via Flink config**: Gradient norm logging is handled by Flink's built-in advanced_logging config rather than custom computation. This logs train/grad_norm and train/update_norm to W&B, which will map to debate/grad/global_norm in the dashboard (mapping handled in plan 05-03).

4. **Role omission vs zeros**: Computation functions omit metrics for absent roles rather than returning 0.0 values. This makes it clear which roles were present in the batch.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all tests passed on first run, configs parsed successfully.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for plan 05-03 (debug data writer):**
- Metric computation functions available for integration
- All functions return debate/-prefixed dicts matching schema
- Unit tests verify correctness of all edge cases

**Ready for plan 05-04 (hook integration):**
- Gradient norm logging already enabled in SWEEP configs
- Will automatically log to W&B when training runs

**No blockers or concerns.**

---
*Phase: 05-wandb-logging-enrichment*
*Completed: 2026-02-04*
