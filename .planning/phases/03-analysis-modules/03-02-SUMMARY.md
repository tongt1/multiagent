---
phase: 03-analysis-modules
plan: 02
subsystem: analysis
tags: [wilson-ci, speech-acts, regex-classifier, merge-conflicts, communication-overhead]

# Dependency graph
requires:
  - phase: 02-results-collection
    provides: "Unified data/results.json with 500 records, merge outcomes, messages, difficulty scores"
provides:
  - "data/fig5_metrics.json: success rates, merge conflict rates, speech acts, communication overhead"
  - "scripts/analyze_fig5.py: rerunnable Figure 5 analysis script (429 lines, stdlib only)"
affects: [04-figure-generation, 05-qualitative-analysis]

# Tech tracking
tech-stack:
  added: []
  patterns: [largest-remainder-rounding, regex-speech-act-classification, wilson-ci-binomial]

key-files:
  created:
    - scripts/analyze_fig5.py
    - data/fig5_metrics.json
  modified: []

key-decisions:
  - "Merge conflict rates use ALL coop records (including eval_error) because merge outcomes are valid even when eval subsequently fails"
  - "Speech act percentages use largest-remainder rounding to guarantee sum == 100.0%"
  - "Speech acts collected from ALL coop-comm records (428 messages) regardless of eval_error"

patterns-established:
  - "Largest-remainder rounding for percentage distributions that must sum to 100%"
  - "Wilson CI function with n=0 handling returns (0.0, 1.0) for complete uncertainty"

requirements-completed: [FIG5-01, FIG5-02, FIG5-03, FIG5-04]

# Metrics
duration: 3min
completed: 2026-02-18
---

# Phase 3 Plan 02: Figure 5 Analysis Summary

**Comm vs no-comm analysis: merge conflicts (41% vs 55%), speech act classification via regex heuristics, 22.8% mean communication overhead**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-18T23:06:43Z
- **Completed:** 2026-02-18T23:09:45Z
- **Tasks:** 2
- **Files created:** 2

## Accomplishments
- Merge conflict rates computed: coop-comm 41% vs coop-nocomm 55% (strongest signal in data, communication reduces conflicts by 14%)
- Speech act classification of 428 messages: 46.7% plan, 26.0% question, 10.5% update, 16.8% other
- Communication overhead: mean 22.8%, median 21.1%, range 2.9%--53.8% across 100 tasks
- Success rates confirmed at 0% for both coop settings (0/96 comm, 0/88 nocomm after excluding eval errors)
- All Wilson 95% CIs valid, speech act categories sum to exactly 100.0%, no double-counting

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement Figure 5 analysis script with speech act classifier** - `8f5f1e8` (feat)
2. **Task 2: Validate Figure 5 metrics against known data properties** - no file changes (inline validation, 10/10 checks passed)

## Files Created/Modified
- `scripts/analyze_fig5.py` - Figure 5 analysis: success rates, merge conflict rates, speech acts, overhead (429 lines, stdlib only)
- `data/fig5_metrics.json` - Structured JSON metrics output consumed by Phase 4 figure generation

## Decisions Made

1. **Merge conflict rates include eval_error records:** All eval_error records have `merge_failed` outcomes because the merge failure causes the eval error. Excluding them would undercount conflicts (37/96 = 38.5% instead of 41/100 = 41%). The plan's expected values (41%, 55%) match the unfiltered totals.

2. **Speech acts collected from ALL coop-comm records:** Messages exist even for records with eval_error (428 total from all 100 records, vs 411 from 96 non-error records). Including all messages gives more complete speech act distribution.

3. **Largest-remainder rounding for percentages:** Individual percentages rounded to 1 decimal place can drift from 100% sum due to floating point. Largest-remainder method distributes the deficit to categories with the largest fractional parts, guaranteeing exact 100.0% sum.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed speech act percentage rounding to guarantee 100% sum**
- **Found during:** Task 1 (initial implementation)
- **Issue:** Naive `round(pct, 1)` for each category produced 99.9% sum (floating point: 46.7 + 25.9 + 10.5 + 16.8 = 99.9), failing the verification check `abs(sum - 100.0) < 0.1`
- **Fix:** Implemented largest-remainder rounding algorithm that distributes fractional deficit to categories with largest remainders
- **Files modified:** scripts/analyze_fig5.py
- **Verification:** Sum now equals exactly 100.0
- **Committed in:** 8f5f1e8 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential for correctness -- plan requires percentages sum to 100%.

## Issues Encountered
None -- all verification checks passed on first run after the rounding fix.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `data/fig5_metrics.json` ready for Phase 4 Figure 5 generation (3-panel plot)
- Phase 5 qualitative analysis can use speech act classifications from this output
- Merge conflict rate (41% vs 55%) is the strongest signal in the dataset for paper comparison

## Self-Check: PASSED

- FOUND: scripts/analyze_fig5.py
- FOUND: data/fig5_metrics.json
- FOUND: .planning/phases/03-analysis-modules/03-02-SUMMARY.md
- FOUND: commit 8f5f1e8

---
*Phase: 03-analysis-modules*
*Completed: 2026-02-18*
