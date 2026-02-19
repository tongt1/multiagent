---
phase: 01-end-to-end-reward-integration
plan: 01
subsystem: training
tags: [reward-shaping, grpo, debate-streamer, numpy, mutation]

# Dependency graph
requires:
  - phase: 08-reward-shaping (prior project)
    provides: "5 reward shaping strategies (identity, difference_rewards, reward_mixing, coma_advantage, potential_based) and DebateMetricStreamer with shaped reward metrics logging"
provides:
  - "In-place mutation of item.data['rewards'] with shaped values in DebateMetricStreamer.get()"
  - "Per-role dict strategy mapping: shaped_dict[role][idx] -> item.data['rewards']"
  - "Global ndarray strategy direct indexing with judge override to 0.0"
  - "8 unit/integration tests verifying write-back correctness"
affects: [01-02, 02-training-infrastructure, 03-evaluation-comparison]

# Tech tracking
tech-stack:
  added: []
  patterns: ["shaped reward write-back after unshaped metrics computation", "per-item dtype preservation on mutation"]

key-files:
  created:
    - ".planning/phases/01-end-to-end-reward-integration/01-01-SUMMARY.md"
  modified:
    - "src/training/wandb_enrichment/debate_streamer.py"
    - "tests/test_debate_streamer.py"
    - "tests/test_reward_shaping_integration.py"

key-decisions:
  - "Refactored _compute_shaped_reward_metrics to _compute_and_apply_shaped_rewards returning (metrics, shaped_per_item) tuple"
  - "Judge items always receive 0.0 reward regardless of strategy type"
  - "Missing role in per-role dict falls back to raw reward (per locked decision from context)"
  - "Original dtype preserved via np.array(value, dtype=original_dtype) to prevent float64 promotion"

patterns-established:
  - "Write-back pattern: unshaped metrics computed first from raw rewards, then shaped values mutate item.data['rewards'] before return"
  - "One-time INFO log pattern: instance flag _logged_mutation prevents repeated log messages"

# Metrics
duration: 5min
completed: 2026-02-14
---

# Phase 1 Plan 1: Shaped Reward Write-Back Summary

**In-place mutation of item.data['rewards'] with shaped values in DebateMetricStreamer.get() so GRPO/RLOO learner receives shaped rewards for gradient computation**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-14T02:50:38Z
- **Completed:** 2026-02-14T02:55:38Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Shaped rewards now flow to the downstream FlinkRlooLearner via item.data["rewards"] mutation, closing the critical gap where shaping was cosmetic (metrics-only)
- Per-role dict strategies (difference_rewards, coma_advantage, reward_mixing) correctly map shaped_dict[role][idx] to each item by role_label
- Global ndarray strategies (identity, potential_based) index directly with judge override to 0.0
- 8 new tests verify all mutation paths: identity no-op, global write-back, per-role indexing, judge zero, unshaped metric preservation, missing role fallback, and 2 integration tests

## Task Commits

Each task was committed atomically:

1. **Task 1: Add shaped reward write-back to DebateMetricStreamer.get()** - `a29e50a` (feat)
2. **Task 2: Add unit and integration tests for reward mutation** - `a4ba888` (test)

## Files Created/Modified
- `src/training/wandb_enrichment/debate_streamer.py` - Refactored _compute_and_apply_shaped_rewards to return (metrics, shaped_per_item); added write-back loop in get() with dtype preservation and one-time INFO log
- `tests/test_debate_streamer.py` - 6 new tests: identity write-back, global strategy, per-role strategy, judge zero, unshaped metrics preserved, missing role fallback
- `tests/test_reward_shaping_integration.py` - 2 new tests: reward_mixing and difference_rewards item mutation integration

## Decisions Made
- Refactored _compute_shaped_reward_metrics to _compute_and_apply_shaped_rewards (renamed + new return type) rather than adding a separate method, keeping the shaping logic cohesive
- Used np.copy(rewards) as starting point for per-role dict case to ensure raw fallback for unknown roles
- Preserved original dtype via np.array(value, dtype=original_dtype) to prevent unintended float64 promotion in the training pipeline

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Pre-existing import error in tests/test_reward_shaping.py (references removed RewardShapingConfig) -- not caused by this plan, excluded from verification suite
- Python dependencies (pydantic, litellm, loguru, etc.) needed manual installation for tests -- worktree did not have a virtual environment configured

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Shaped rewards now flow through to the learner via item.data["rewards"] mutation
- Ready for Plan 01-02: gradient verification tests proving shaped rewards affect training loss
- All 56 tests pass across test_debate_streamer, test_reward_shaping_integration, and test_reward_shaping_registry

---
*Phase: 01-end-to-end-reward-integration*
*Completed: 2026-02-14*
