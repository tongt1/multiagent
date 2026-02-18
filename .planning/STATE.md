# Project State

## Project Reference

See: /home/terry_tong_cohere_com/cooperbench-repro/.planning/PROJECT.md (updated 2026-02-14)

**Core value:** Produce verifiable figures (4, 5, 6) that replicate the CooperBench paper's key findings -- the solo-coop coordination gap, communication's failure to improve cooperation despite reducing merge conflicts, and the breakdown of communication errors -- using Command A instead of the paper's external models.

**Current focus:** Phase 2 -- Results Collection and Data Foundation (Plan 1 complete, 2 remaining)

## Current Position

Phase: 2 (Results Collection and Data Foundation)
Plan: 1 of 3 in current phase (02-01 complete)
Status: Phase 2 in progress
Last activity: 2026-02-18 -- Completed 02-01-PLAN.md (eval all 300 benchmark runs)

Progress: [###.......] 27% (Phase 1: 3/3 + Phase 2: 1/3 = 4 plans complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 4
- Average duration: 91 min
- Total execution time: 6 hours 5 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 3/3 | 300 min | 100 min |
| 2 | 1/3 | 65 min | 65 min |

**Recent Trend:**
- 01-01: 4 min (2 tasks, 2 files)
- 01-02: 6 min (2 tasks, 1 file)
- 01-03: 290 min (2 tasks, 300 result files)
- 02-01: 65 min (2 tasks, 1 code file + 303 data files)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in /home/terry_tong_cohere_com/cooperbench-repro/.planning/PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Use upstream CooperBench CLI via subprocess, NOT the custom wrapper pipeline: The custom wrapper uses prompt-based patch generation without real tool use, producing results architecturally incomparable to the paper.
- Command A only (no multi-model): Simplifies initial reproduction, validates pipeline first.
- Start with lite subset (26 tasks, 100 pairs): Sufficient for statistical analysis, faster iteration.
- Docker backend only: Images already partially available locally, simplest path.
- Pin pandas<3.0: Copy-on-Write defaults, PyArrow-backed strings, and NaN/NA unification in pandas 3.0 are too risky for a correctness-critical reproduction pipeline.
- Focus on relative patterns (retention ratios, difficulty curves) rather than absolute numbers: Single-model reproduction means absolute rates will differ from multi-model paper.
- Use uv for Python 3.12 virtualenv creation (system has only 3.11.2)
- Install cooperbench in editable mode to allow local patches if needed downstream
- Verify Docker images by probing /workspace/repo inside temporary containers
- Corrected MSWEA_MODEL_API_BASE to https://stg.api.cohere.com/v2/chat (litellm uses api_base as-is, does not append /chat)
- Agent budget (step_limit=100, cost_limit=0) produces 99.4% submission rate at $0.46/task avg
- MSWEA_COST_TRACKING=ignore_errors prevents cost calculation failures from halting runs
- Redis auto-started by cooperbench for coop mode; no manual setup needed
- Docker eval backend entrypoint override: added entrypoint="" to prevent CooperBench image runner.sh from exiting containers immediately
- Eval_summary.json regenerated from actual eval.json files after retry pass to ensure accurate aggregate counts

### Pending Todos

- ~~Pull/build 8 missing Docker images for lite subset before Phase 1 execution begins.~~ DONE (01-01)
- ~~Verify upstream CooperBench CLI installation compatibility with project virtualenv.~~ DONE (01-01)
- ~~Confirm Redis setup for coop mode inter-agent messaging.~~ DONE (01-03, auto-started by cooperbench)
- ~~Check Cohere API endpoint (stg vs production) for Command A availability.~~ DONE (01-02, staging endpoint validated)
- ~~Review agent per-task budget ceiling for full runs.~~ DONE (01-03, step_limit=100 produces 99.4% submission rate)

### Blockers/Concerns

- ~~8 of 26 Docker images for lite subset are not yet pulled/built.~~ RESOLVED (01-01)
- ~~Upstream CooperBench CLI may have pinned dependencies that conflict with project environment.~~ RESOLVED (01-01, editable install succeeded)
- ~~Redis required for coop mode -- setup status unknown.~~ RESOLVED (01-03, auto-started by cooperbench)
- ~~All 5 smoke test tasks hit LimitsExceeded with 0 patches.~~ RESOLVED (01-03, step_limit=100 fixed this; 99.4% submission rate in full runs)

### Benchmark Results (Phase 1 Output)

| Setting     | Pairs | Submitted | Cost     | Avg Steps |
|-------------|-------|-----------|----------|-----------|
| Solo        | 100   | 98/100    | $45.77   | 10.4      |
| Coop-Comm   | 100   | 200/200   | $47.32   | 21.4      |
| Coop-NoComm | 100   | 199/200   | $45.50   | 17.3      |
| **Total**   | 300   | 497/500   | $138.59  |           |

Results location: `repos/CooperBench/logs/{command-a-solo,command-a-coop-comm,command-a-coop-nocomm}/`

### Eval Results (Phase 2 Output)

| Setting     | Total | Passed | Failed | Errors | With Merge |
|-------------|-------|--------|--------|--------|------------|
| Solo        | 100   | 1      | 96     | 3      | 0          |
| Coop-Comm   | 100   | 0      | 96     | 4      | 100        |
| Coop-NoComm | 100   | 0      | 88     | 12     | 100        |
| **Total**   | 300   | 1      | 280    | 19     | 200        |

Eval location: `repos/CooperBench/logs/{command-a-solo,command-a-coop-comm,command-a-coop-nocomm}/**/eval.json`

## Session Continuity

Last session: 2026-02-18
Stopped at: Completed 02-01-PLAN.md (eval all 300 benchmark runs)
Resume file: None
Next action: Execute 02-02-PLAN.md (additional solo seeds for difficulty scoring)

---
*State initialized: 2026-02-14*
*Last updated: 2026-02-18 (02-01 complete)*
