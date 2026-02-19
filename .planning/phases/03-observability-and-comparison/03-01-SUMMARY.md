---
phase: 03-observability-and-comparison
plan: 01
subsystem: observability
tags: [wandb, metrics, workspace-template, reward-shaping, dashboard]

# Dependency graph
requires:
  - phase: 01-end-to-end-reward-integration
    provides: DebateMetricStreamer with _compute_and_apply_shaped_rewards() and hardcoded metric strings
  - phase: 02-experiment-configuration
    provides: 5 SWEEP configs with reward_shaping_strategy in DebateMetricStreamerConfig
provides:
  - METRIC_SHAPED_REWARD_* constants in metric_schema.py (centralized, importable)
  - wandb.config.update() surfacing reward_shaping_strategy as top-level WandB run config key
  - Workspace template with shaped reward comparison panels (mean + per-role)
  - 3 new tests validating wandb config update and workspace template content
affects: [03-02-submit-and-validate, future-dashboard-extensions]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "sys.modules mock pattern for wandb in test environment without wandb installed"
    - "Role-to-metric-constant lookup dict for per-role metric name resolution"

key-files:
  created: []
  modified:
    - src/training/wandb_enrichment/metric_schema.py
    - src/training/wandb_enrichment/debate_streamer.py
    - src/training/wandb_enrichment/workspace_template.py
    - tests/test_debate_streamer.py

key-decisions:
  - "Used sys.modules mock pattern for wandb tests since wandb is not installed locally"
  - "Placed _update_wandb_config() call after workspace init in first get() to ensure wandb.run exists"

patterns-established:
  - "Shaped reward metrics use METRIC_SHAPED_REWARD_* constants, never hardcoded strings"
  - "wandb.config.update() with allow_val_change=True for post-init config augmentation"

# Metrics
duration: 3min
completed: 2026-02-14
---

# Phase 3 Plan 01: Observability and Comparison Summary

**WandB run config enrichment with reward_shaping_strategy key and workspace template with 2 shaped reward comparison panels (mean + per-role)**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-14T03:55:09Z
- **Completed:** 2026-02-14T03:58:21Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- 5 METRIC_SHAPED_REWARD_* constants centralized in metric_schema.py, replacing all hardcoded "debate/shaped_reward/*" strings
- wandb.config.update() surfaces reward_shaping_strategy as a top-level filterable/groupable WandB config key on first get()
- Workspace template extended with Section 4 "Reward Shaping" containing mean shaped reward and per-role shaped reward LinePlot panels
- 3 new tests validate: wandb config update called once with strategy name, skipped when wandb.run is None, workspace template source contains shaped reward panels

## Task Commits

Each task was committed atomically:

1. **Task 1: Add shaped reward metric constants and wandb.config.update() to streamer** - `7fc1e13` (feat)
2. **Task 2: Extend workspace template with shaped reward panels and add tests** - `58843ce` (feat)

## Files Created/Modified
- `src/training/wandb_enrichment/metric_schema.py` - Added 5 METRIC_SHAPED_REWARD_* constants and registered them in ALL_DEBATE_METRICS
- `src/training/wandb_enrichment/debate_streamer.py` - Added _update_wandb_config() method, called on first get(), refactored hardcoded metric strings to use imported constants
- `src/training/wandb_enrichment/workspace_template.py` - Added shaped reward metric imports, Section 4 with 2 LinePlot panels, renumbered Rollout Samples to Section 5
- `tests/test_debate_streamer.py` - Added 3 new tests for wandb config update behavior and workspace template content

## Decisions Made
- Used `sys.modules` mock pattern for wandb tests since wandb is not installed in the local dev environment -- `patch.dict(sys.modules, {"wandb": mock_wandb})` instead of `patch("wandb.run")`
- Placed `_update_wandb_config()` call after workspace init but still inside the `if self._get_count == 1:` guard -- ensures wandb.run exists and config update happens exactly once

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed wandb mock approach in tests for environment without wandb**
- **Found during:** Task 2 (adding wandb config update tests)
- **Issue:** Plan specified `patch("wandb.run", new=MagicMock())` but wandb module is not installed locally, causing ModuleNotFoundError
- **Fix:** Used `patch.dict(sys.modules, {"wandb": mock_wandb})` to inject a complete mock wandb module instead of patching attributes on the real module
- **Files modified:** tests/test_debate_streamer.py
- **Verification:** All 22 tests pass including the 3 new wandb-related tests
- **Committed in:** 58843ce (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Necessary adaptation for test environment. No scope creep.

## Issues Encountered
None beyond the wandb mock fix documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All shaped reward metrics now centralized as importable constants
- WandB run config will surface reward_shaping_strategy for run filtering/grouping
- Workspace template will auto-create shaped reward comparison panels on first training run
- Ready for Plan 03-02: submit all 5 SWEEP configs and validate comparison dashboard

## Self-Check: PASSED

- All 5 files verified present on disk
- Both commit hashes (7fc1e13, 58843ce) verified in git log

---
*Phase: 03-observability-and-comparison*
*Completed: 2026-02-14*
