# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-14)

**Core value:** All 5 reward strategies must be fully wired end-to-end so shaped rewards affect gradient updates and produce meaningful WandB comparison curves.
**Current focus:** Phase 3 - Observability and Comparison

## Current Position

Phase: 3 of 3 (Observability and Comparison) -- Plan 01 complete
Plan: 1 of 2 in current phase
Status: Plan 03-01 complete, ready for Plan 03-02
Last activity: 2026-02-14 -- Completed 03-01 (WandB config enrichment + shaped reward workspace panels)

Progress: [############░░] 80%

## Performance Metrics

**Velocity:**
- Total plans completed: 4
- Average duration: 5.0min
- Total execution time: 0.33 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-end-to-end-reward-integration | 2 | 11min | 5.5min |
| 02-experiment-configuration | 1 | 6min | 6min |
| 03-observability-and-comparison | 1 | 3min | 3min |

**Recent Trend:**
- Last 5 plans: 5min, 6min, 6min, 3min
- Trend: Accelerating, 3min for Plan 03-01

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- SmolLM-135M chosen for fast iteration (minutes not hours)
- All 5 strategies in single comparison with same hyperparams
- Default strategy hyperparams (no per-strategy tuning in v1)
- Refactored _compute_shaped_reward_metrics to _compute_and_apply_shaped_rewards returning (metrics, shaped_per_item) tuple
- Judge items always receive 0.0 reward regardless of strategy type
- Missing role in per-role dict falls back to raw reward
- Original dtype preserved via np.array(value, dtype=original_dtype)
- Used torch.backward() for gradient-norm comparison per locked decisions R12/R13
- atol=1e-6, rtol=1e-5 tolerances for identity regression per user decision
- Invalid strategy name errors at __init__() time (fail fast), not at runtime
- AST/text-based test validation for SWEEP configs (sweep module unavailable locally)
- Shared _base.py module guarantees ECFG-06 (identical hyperparams except reward shaping)
- Used sys.modules mock pattern for wandb tests since wandb is not installed locally
- Placed _update_wandb_config() call after workspace init in first get() to ensure wandb.run exists

### Pending Todos

None.

### Blockers/Concerns

- RESOLVED: Reward shaping strategies now wired into DebateMetricStreamer with write-back to item.data["rewards"]
- RESOLVED: Identity regression and gradient-path liveness verified via 13 integration tests
- RESOLVED: DebateMetricStreamerConfig reward_shaping_strategy/params fields now uncommented in 5 strategy-specific SWEEP configs
- RESOLVED: All 5 configs verified identical except for reward shaping fields (85 tests, ECFG-01 through ECFG-08)
- RESOLVED: Shaped reward metrics centralized as METRIC_SHAPED_REWARD_* constants, hardcoded strings replaced
- RESOLVED: wandb.config.update() surfaces reward_shaping_strategy for WandB run filtering
- Next: Plan 03-02 -- submit all 5 SWEEP configs and validate comparison dashboard

## Session Continuity

Last session: 2026-02-14
Stopped at: Completed 03-01-PLAN.md (WandB config enrichment + workspace panels). Plan 03-01 complete.
Resume file: .planning/phases/03-observability-and-comparison/03-01-SUMMARY.md
