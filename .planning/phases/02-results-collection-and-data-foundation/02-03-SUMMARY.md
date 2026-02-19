---
phase: 02-results-collection-and-data-foundation
plan: 03
subsystem: data
tags: [json, collection, difficulty, buckets, normalization, cooperbench]

# Dependency graph
requires:
  - plan: 02-01
    provides: "300 eval.json files with per-run test pass/fail results and merge outcomes"
  - plan: 02-02
    provides: "200 additional solo eval.json files (2 seeds x 100 pairs) for difficulty scoring"
provides:
  - "scripts/collect_results.py: rerunnable pure-Python collection pipeline"
  - "data/results.json: unified data store with 500 records (24 fields each)"
  - "Per-pair difficulty scores computed from 3 solo seeds"
  - "10 equal-width difficulty bucket assignments for all records"
affects: [phase-3-analysis, figure-4, figure-5, figure-6]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Flat JSON array as unified data store", "Difficulty d(pair) = 1 - mean(solo_both_passed) from 3 seeds", "10 equal-width buckets: bucket = min(floor(d*10), 9)"]

key-files:
  created:
    - scripts/collect_results.py
    - data/results.json
  modified: []

key-decisions:
  - "infra_error defaults to false when field absent from result.json (none present in our data)"
  - "Coop-nocomm conversation.json files contain empty arrays -- embedded as-is in records"
  - "LimitsExceeded override applied to 4 records (all solo, all already both_passed=false)"
  - "No merge_union outcomes observed -- all coop merges were either merge_clean (naive) or merge_failed"

patterns-established:
  - "Unified record schema: 24 fields covering run metadata, eval results, merge outcomes, difficulty, and conversation messages"
  - "Pair key convention: (repo, task_id, tuple(features)) uniquely identifies a feature pair"
  - "Agent status flattening for coop: worst status wins (Error > LimitsExceeded > Submitted > Unknown)"
  - "Solo records always have merge_outcome=merge_clean, merge_status=null, merge_strategy=null"

requirements-completed: [DATA-01, DATA-02, DATA-03, FIG4-01, FIG4-02]

# Metrics
duration: 3min
completed: 2026-02-18
---

# Phase 2 Plan 3: Unified Results Collection and Data Store Summary

**Pure-Python collection pipeline producing data/results.json with 500 normalized records, per-pair difficulty from 3 solo seeds, and 10-bucket assignments across all settings**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-18T22:10:16Z
- **Completed:** 2026-02-18T22:13:31Z
- **Tasks:** 2
- **Files modified:** 2 (1 script + 1 data file)

## Accomplishments

- Built `scripts/collect_results.py` (276 lines, stdlib-only) that reads all result.json, eval.json, and conversation.json from 5 run directories
- Produced `data/results.json` with 500 records: 300 solo (3 seeds x 100) + 100 coop-comm + 100 coop-nocomm
- Computed difficulty scores from 3 solo seeds yielding d in {0.3333, 0.6667, 1.0} (98% of pairs at d=1.0)
- Validated all 5 phase requirements (DATA-01 through FIG4-02) with comprehensive assertion checks

## Key Results

### Record Counts
| Setting | Records |
|---------|---------|
| Solo (seed 0) | 100 |
| Solo (seed 1) | 100 |
| Solo (seed 2) | 100 |
| Coop-comm | 100 |
| Coop-nocomm | 100 |
| **Total** | **500** |

### Difficulty Distribution (100 unique pairs)
| Difficulty | Bucket | Pairs | Interpretation |
|-----------|--------|-------|----------------|
| 0.3333 | 3 | 1 | Medium-easy (passed 2/3 seeds) |
| 0.6667 | 6 | 1 | Medium-hard (passed 1/3 seeds) |
| 1.0 | 9 | 98 | Hard (passed 0/3 seeds) |

### Pass Rates
| Setting | Passed | Total | Rate |
|---------|--------|-------|------|
| Solo (seed 0) | 1 | 100 | 1.0% |
| Coop-comm | 0 | 100 | 0.0% |
| Coop-nocomm | 0 | 100 | 0.0% |

### Merge Outcome Distribution (coop only)
| Outcome | Count |
|---------|-------|
| merge_clean | 104 |
| merge_failed | 96 |
| merge_union | 0 |

### Error Breakdown
| Category | Count |
|----------|-------|
| infra_error=true | 0 |
| LimitsExceeded | 4 |
| eval_error!=null | 27 |

## Task Commits

Each task was committed atomically:

1. **Task 1: Build collection script and produce unified results store** - `883cecb` (feat)
2. **Task 2: Validate unified results store against phase requirements** - no commit (validation-only, no files changed)

## Files Created/Modified

- `scripts/collect_results.py` - Pure-Python collection pipeline: reads all log files, normalizes into unified schema, computes difficulty, writes data/results.json
- `data/results.json` - Unified data store: 500 records, 24 fields each, flat JSON array

## Decisions Made

- **infra_error field:** The `infra_error` field does not exist in any result.json files. Defaults to `false` for all records. This is consistent with Phase 1 output (no infrastructure errors reported).
- **conversation.json for coop-nocomm:** These files exist and contain empty arrays `[]`. Embedded as-is in records (messages=[], messages_count=0).
- **No merge_union outcomes:** All coop merges used either naive strategy (merge_clean) or failed entirely (merge_failed). No union strategy merges were observed. This means the agents' patches either applied cleanly or conflicted completely.
- **LimitsExceeded overlap with eval results:** All 4 LimitsExceeded records already had both_passed=false in eval.json, so the override did not change any values. The override is still applied as a safety measure per locked decision.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all data files were present and in expected format. Collection completed on first run with no errors.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `data/results.json` is the single data artifact consumed by all downstream analysis (Phases 3-5)
- Every figure, metric, and qualitative analysis can now read from this file
- Script is idempotent: re-running `python3 scripts/collect_results.py` regenerates identical output
- Known limitation: only 3 of 10 difficulty buckets are populated (buckets 3, 6, 9) due to single-model reproduction yielding very low pass rates

## Self-Check: PASSED

- scripts/collect_results.py: FOUND
- data/results.json: FOUND (500 records)
- 02-03-SUMMARY.md: FOUND
- Commit 883cecb: FOUND

---
*Phase: 02-results-collection-and-data-foundation*
*Completed: 2026-02-18*
