---
phase: 02-results-collection-and-data-foundation
plan: 01
subsystem: eval
tags: [docker, cooperbench, eval, benchmark, testing]

# Dependency graph
requires:
  - phase: 01-execution-infrastructure
    provides: "300 benchmark run results (solo, coop-comm, coop-nocomm) in logs/ directories"
provides:
  - "300 eval.json files with per-run test pass/fail results and merge outcomes"
  - "3 eval_summary.json files with aggregate pass/fail/error counts per setting"
  - "Docker eval backend entrypoint fix enabling container-based evaluation"
affects: [02-02, 02-03, figure-generation, analysis]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Docker entrypoint override for CooperBench images", "eval.json per-run result format"]

key-files:
  created:
    - repos/CooperBench/logs/command-a-solo/eval_summary.json
    - repos/CooperBench/logs/command-a-coop-comm/eval_summary.json
    - repos/CooperBench/logs/command-a-coop-nocomm/eval_summary.json
  modified:
    - repos/CooperBench/src/cooperbench/eval/backends/docker.py

key-decisions:
  - "entrypoint override: added entrypoint='' to DockerBackend.create_sandbox() to prevent CooperBench image runner.sh from exiting containers immediately"
  - "Regenerated eval_summary.json from actual eval.json files after retry pass to ensure accurate aggregate counts"

patterns-established:
  - "eval.json format: {repo, task_id, features, setting, merge, feature1, feature2, both_passed, error, evaluated_at}"
  - "eval_summary.json format: {run_name, evaluated_at, total_runs, passed, failed, errors, skipped, pass_rate, results[]}"

requirements-completed: [DATA-02, DATA-03]

# Metrics
duration: 65min
completed: 2026-02-18
---

# Phase 2 Plan 1: Eval All 300 Benchmark Runs Summary

**Docker eval backend entrypoint fix + full evaluation of 300 benchmark runs producing eval.json with test pass/fail and merge outcomes per run**

## Performance

- **Duration:** 65 min
- **Started:** 2026-02-18T18:35:55Z
- **Completed:** 2026-02-18T19:41:08Z
- **Tasks:** 2
- **Files modified:** 1 code file + 300 eval.json + 3 eval_summary.json (data artifacts)

## Accomplishments

- Fixed Docker eval backend entrypoint bug that caused containers to exit immediately
- Evaluated all 300 benchmark runs (100 solo + 100 coop-comm + 100 coop-nocomm)
- All 300 eval.json files validated: contain both_passed, feature-level results, and merge data (for coop settings)
- Coop eval.json files include merge.status and merge.strategy fields

## Eval Results Summary

| Setting        | Total | Passed | Failed | Errors | With Merge |
|----------------|-------|--------|--------|--------|------------|
| Solo           | 100   | 1      | 96     | 3      | 0          |
| Coop-Comm      | 100   | 0      | 96     | 4      | 100        |
| Coop-NoComm    | 100   | 0      | 88     | 12     | 100        |
| **Total**      | 300   | 1      | 280    | 19     | 200        |

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix Docker eval backend entrypoint and smoke-test eval** - `6cff50f` (fix)
2. **Task 2: Run full evaluation on all 300 existing benchmark results** - no code commit (data artifacts in gitignored logs/ directory)

## Files Created/Modified

- `repos/CooperBench/src/cooperbench/eval/backends/docker.py` - Added entrypoint="" to create_sandbox() to override image ENTRYPOINT
- `repos/CooperBench/logs/command-a-solo/**/eval.json` (100 files) - Solo eval results
- `repos/CooperBench/logs/command-a-coop-comm/**/eval.json` (100 files) - Coop-comm eval results with merge data
- `repos/CooperBench/logs/command-a-coop-nocomm/**/eval.json` (100 files) - Coop-nocomm eval results with merge data
- `repos/CooperBench/logs/command-a-{solo,coop-comm,coop-nocomm}/eval_summary.json` (3 files) - Aggregate summaries

## Decisions Made

- **Entrypoint override:** Added `entrypoint=""` to `DockerBackend.create_sandbox()` to prevent CooperBench Docker image `ENTRYPOINT=/usr/local/bin/runner.sh` from causing containers to exit immediately. This matches the pattern already used in `agents/mini_swe_agent/environments/docker.py`.
- **Retry strategy:** After initial eval pass left 28 tasks unevaluated (dspy and typst tasks with slow builds), retried once per plan directive. Typst tasks completed in initial pass after ~30 min of Rust compilation. Remaining 15 dspy tasks completed on retry.
- **Regenerated eval_summary.json:** The retry overwrote eval_summary.json with only retry counts, so regenerated accurate combined summaries from actual eval.json files on disk.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Regenerated eval_summary.json after retry overwrote with partial counts**
- **Found during:** Task 2 (after retry pass)
- **Issue:** The retry eval command overwrote eval_summary.json with only the 5 retry runs instead of the full 100 runs per setting
- **Fix:** Regenerated eval_summary.json by scanning all eval.json files and computing accurate aggregate counts
- **Files modified:** logs/command-a-{solo,coop-comm,coop-nocomm}/eval_summary.json
- **Verification:** Summary totals match eval.json file counts (100 per setting)
- **Committed in:** N/A (data artifacts in gitignored directory)

---

**Total deviations:** 1 auto-fixed (1 bug fix)
**Impact on plan:** Minor data correction. No scope creep.

## Issues Encountered

- **Typst task compilation time:** Typst (task 6554) is a Rust project requiring ~30 minutes for test compilation per container. 13 typst containers ran simultaneously across 3 settings, completing successfully but significantly slowing the eval pass.
- **DSPy task timeouts (initial pass):** 15 dspy tasks (across both task 8394 and 8587) failed to produce eval.json on the first pass. Some containers were killed (exit code 137) before writing results. Retry successfully evaluated all 15.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All 300 eval.json files ready for consumption by unified results store (02-02)
- Eval data includes all required fields: both_passed, feature-level results, merge status/strategy (coop), error capture
- eval_summary.json provides quick aggregate view per setting

## Self-Check: PASSED

- 02-01-SUMMARY.md: FOUND
- docker.py entrypoint fix: FOUND
- eval_summary.json (solo, coop-comm, coop-nocomm): ALL FOUND
- eval.json counts: 100 + 100 + 100 = 300 (CORRECT)
- Commit 6cff50f: FOUND

---
*Phase: 02-results-collection-and-data-foundation*
*Completed: 2026-02-18*
