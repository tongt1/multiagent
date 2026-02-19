---
phase: 03-analysis-modules
plan: 01
subsystem: analysis
tags: [wilson-ci, trapezoidal-auc, statistics, figure4, difficulty-curves]

# Dependency graph
requires:
  - phase: 02-scale-and-verify
    provides: "data/results.json unified data store with 500 records"
provides:
  - "scripts/analyze_fig4.py -- rerunnable Figure 4 analysis script"
  - "data/fig4_metrics.json -- per-bucket rates, Wilson CIs, AUC, retention metrics"
affects: [04-figure-generation, figure4-plotting]

# Tech tracking
tech-stack:
  added: []
  patterns: ["stdlib-only analysis script (json, math, collections, pathlib, argparse)", "Wilson score CI for binomial proportions", "trapezoidal AUC over sparse bucket data"]

key-files:
  created: ["scripts/analyze_fig4.py", "data/fig4_metrics.json"]
  modified: []

key-decisions:
  - "Solo rates use seed=0 only (97 valid records), matching coop denominator for fair comparison"
  - "Bucket 3 solo is 1/1 passed (seed=0 only), not 2/3 (all seeds) -- correct per plan design"
  - "Empty buckets excluded from AUC -- integrate only over 3 populated bucket centers"
  - "Wilson CI returns (0.0, 1.0) for n=0 edge case -- complete uncertainty"

patterns-established:
  - "Analysis script pattern: load results.json, filter errors, compute metrics, write structured JSON"
  - "Wilson CI edge case handling: n=0 -> (0.0, 1.0), p=0 -> ci_lower=0.0"
  - "AUC computation: trapezoidal integration over populated buckets only, with sparsity warning"

requirements-completed: [FIG4-03, FIG4-04, FIG4-05, FIG4-06]

# Metrics
duration: 3min
completed: 2026-02-18
---

# Phase 3 Plan 1: Figure 4 Analysis Summary

**Per-bucket success rates with Wilson 95% CIs, trapezoidal AUC over 3 populated buckets, and retention ratio -- all pure stdlib Python, validated against 9 data property checks**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-18T23:06:02Z
- **Completed:** 2026-02-18T23:09:00Z
- **Tasks:** 2
- **Files created:** 2

## Accomplishments
- Computed per-bucket solo and coop success rates for all 3 populated buckets (3, 6, 9) with correct seed filtering
- Wilson 95% confidence intervals for every rate, correctly handling p=0, p=1, n=0, and small-n edge cases
- AUC via trapezoidal integration: solo=0.15 (from 100% in bucket 3, 0% in buckets 6 and 9), coop=0.0 (all rates zero)
- Retention metric: 0.0 for both coop settings (AUC_coop/AUC_solo = 0/0.15)
- All 9 validation checks passed: bucket counts, seed filtering, CI bounds, AUC consistency, no inflation, metadata completeness

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement Figure 4 analysis script** - `0d6ba14` (feat)
2. **Task 2: Validate Figure 4 metrics** - no file changes (inline validation, all 9 checks PASS)

## Files Created/Modified
- `scripts/analyze_fig4.py` - Figure 4 analysis: per-bucket rates, Wilson CIs, trapezoidal AUC, retention
- `data/fig4_metrics.json` - Structured JSON output with 3 buckets, AUC values, retention, metadata

## Key Metrics from Output

| Bucket | Center | Solo Rate | Solo CI | Coop-Comm Rate | Coop-NoComm Rate |
|--------|--------|-----------|---------|----------------|------------------|
| 3 | 0.35 | 100.0% (1/1) | [0.2065, 1.0] | 0.0% (0/1) | 0.0% (0/1) |
| 6 | 0.65 | 0.0% (0/1) | [0.0, 0.7935] | 0.0% (0/1) | 0.0% (0/1) |
| 9 | 0.95 | 0.0% (0/95) | [0.0, 0.0389] | 0.0% (0/94) | 0.0% (0/86) |

- **Solo AUC:** 0.15 (3 points, x_range [0.35, 0.95])
- **Coop AUC:** 0.0 (both settings)
- **Retention:** 0.0 (both settings)
- **Records used:** 473 of 500 (27 excluded for eval_error)
- **Solo seed=0 records:** 97 (3 excluded for eval_error)

## Decisions Made
- Solo rates use seed=0 only (97 valid records after excluding eval errors), giving fair comparison with coop settings (~96 and ~88 valid records respectively)
- Bucket 3 solo shows 1/1 passed with seed=0 (not 2/3 from all seeds) -- this is correct per the plan's design to use single-seed comparison
- Empty buckets are excluded from output entirely, not treated as rate=0 -- AUC integrates only over the 3 populated bucket centers
- Wilson CI for n=0 returns (0.0, 1.0) representing complete uncertainty, used for consistency in the data structure

## Deviations from Plan

None - plan executed exactly as written.

Note: The plan's "Expected values" section mentioned "Bucket 3 solo: 2/3 passed = 66.7%" which was based on research using all 3 seeds. With seed=0 only (as the plan correctly specifies), bucket 3 solo is 1/1 = 100%. This is consistent behavior, not a deviation.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `data/fig4_metrics.json` is ready for Phase 4 figure generation (Figure 4 difficulty-stratified success curves)
- Structured JSON output matches the schema specified in the plan
- Script is idempotent and rerunnable with `--input` and `--output` CLI arguments

## Self-Check: PASSED

- FOUND: scripts/analyze_fig4.py
- FOUND: data/fig4_metrics.json
- FOUND: .planning/phases/03-analysis-modules/03-01-SUMMARY.md
- FOUND: commit 0d6ba14 (Task 1)

---
*Phase: 03-analysis-modules*
*Completed: 2026-02-18*
