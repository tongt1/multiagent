---
phase: 05-qualitative-transcript-analysis
plan: 01
subsystem: analysis
tags: [scipy, fisher-exact, mann-whitney-u, speech-acts, qualitative, statistics]

# Dependency graph
requires:
  - phase: 03-analysis-modules
    provides: "classify_speech_act function in analyze_fig5.py"
  - phase: 02-results-collection-and-data-foundation
    provides: "data/results.json with 100 coop-comm transcripts"
provides:
  - "Qualitative transcript metrics (Plan:Question ratio, first-turn planning, specificity)"
  - "Statistical correlation of communication patterns with merge conflict outcomes"
  - "Summary comparison table with Mann-Whitney U and Fisher's exact p-values"
affects: []

# Tech tracking
tech-stack:
  added: [scipy.stats.fisher_exact, scipy.stats.mannwhitneyu]
  patterns: [import-reuse-classifier, per-trajectory-metrics, group-comparison-with-statistical-tests]

key-files:
  created:
    - scripts/analyze_qualitative.py
    - data/qualitative_metrics.json
  modified: []

key-decisions:
  - "Reuse exact classify_speech_act from analyze_fig5.py for Phase 3 consistency"
  - "Store infinite Plan:Question ratios as None (JSON null), run stats on finite only"
  - "Report counter-intuitive findings accurately: conflict trajectories have HIGHER planning"
  - "Include small-sample caveat for n=11 no-plan-first group"
  - "Note line mention sparsity (1/428 messages) rather than omitting the metric"

patterns-established:
  - "Import classifier from prior phase script for cross-phase consistency"
  - "Per-trajectory metric computation with group comparison and statistical testing"

requirements-completed: [QUAL-01, QUAL-02, QUAL-03, QUAL-04]

# Metrics
duration: 2min
completed: 2026-02-19
---

# Phase 5 Plan 1: Qualitative Transcript Analysis Summary

**Qualitative metrics (P:Q ratio, first-turn planning, file/line specificity) computed for 100 coop-comm trajectories with Mann-Whitney U and Fisher's exact tests correlating patterns to merge conflict outcomes**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-19T00:28:40Z
- **Completed:** 2026-02-19T00:31:30Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Plan:Question ratio computed per trajectory, Mann-Whitney U test shows conflict group has significantly higher ratios (mean 1.96 vs 1.15, p=0.0016)
- First-turn planning detected in 89/100 trajectories, Fisher's exact test on 2x2 table (p=0.0249) with small-sample caveat for n=11 no-plan group
- Specificity metrics: 361/428 messages mention file paths (no group difference, p=0.917), line mentions sparse (1/428)
- Summary comparison table generated in JSON and console formats with all four QUAL metrics, p-values, and direction notes

## Task Commits

Each task was committed atomically:

1. **Task 1: Create qualitative transcript analysis script** - `0700f4a` (feat)
2. **Task 2: Run analysis and validate output metrics** - `2d17467` (feat)

## Files Created/Modified
- `scripts/analyze_qualitative.py` - Qualitative transcript analysis script computing QUAL-01 through QUAL-04 with imported classifier and scipy statistical tests
- `data/qualitative_metrics.json` - Per-trajectory metrics (100 records), group comparisons with p-values, and summary comparison table

## Decisions Made
- Reused exact `classify_speech_act` from `analyze_fig5.py` via import rather than duplicating or improving -- ensures Plan:Question ratios are on the same classification basis as Phase 3 speech act percentages
- Stored `plan_question_ratio` as `None` (JSON null) for 31 infinite-ratio trajectories (plans but no questions) rather than using float('inf') or large numbers
- Reported counter-intuitive correlations accurately: conflict trajectories have HIGHER planning rates (97.6% vs 83.1%) and HIGHER P:Q ratios (1.96 vs 1.15) -- do not imply causation
- Included small-sample caveat for no-plan-first group (n=11) in Fisher's exact test results
- Computed and reported line mention metric despite sparsity (1/428) with explicit note rather than omitting

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Key Results

| Metric | No Conflict (n=59) | Conflict (n=41) | p-value | Direction |
|--------|-------------------|-----------------|---------|-----------|
| P:Q ratio (mean, finite) | 1.15 | 1.96 | 0.0016 | Higher ratio -> more conflicts |
| First-turn planning rate | 83.1% | 97.6% | 0.0249 | Plan-first -> more conflicts |
| File mentions/trajectory | 3.61 | 3.61 | 0.9173 | No difference |
| Line mentions/trajectory | 0.017 | 0.000 | sparse | Insufficient data |

**Key insight:** Counter-intuitively, conflict trajectories show MORE planning behavior. This likely reflects that tasks where agents plan to modify overlapping files generate both more plan messages AND more conflicts -- planning alone does not prevent conflicts when both agents target the same code.

## Next Phase Readiness
This is the final plan of the final phase. The entire project is now complete:
- Phase 1: Benchmark execution (3/3 plans)
- Phase 2: Results collection (3/3 plans)
- Phase 3: Analysis modules (3/3 plans)
- Phase 4: Figure generation (2/2 plans)
- Phase 5: Qualitative analysis (1/1 plans)

Total: 12/12 plans complete across 5 phases.

## Self-Check: PASSED

- FOUND: scripts/analyze_qualitative.py
- FOUND: data/qualitative_metrics.json
- FOUND: 05-01-SUMMARY.md
- FOUND: 0700f4a (Task 1 commit)
- FOUND: 2d17467 (Task 2 commit)

---
*Phase: 05-qualitative-transcript-analysis*
*Completed: 2026-02-19*
