---
phase: 01-execution-infrastructure
plan: 03
subsystem: benchmark-execution
tags: [cooperbench, benchmark, solo, coop, docker, cohere, command-a, litellm]

# Dependency graph
requires:
  - phase: 01-execution-infrastructure/01-02
    provides: "Orchestrator script with retry logic, validated pipeline"
provides:
  - "Complete solo mode results (100 pairs) in logs/command-a-solo/"
  - "Complete coop-with-comm results (100 pairs) in logs/command-a-coop-comm/"
  - "Complete coop-no-comm results (100 pairs) in logs/command-a-coop-nocomm/"
  - "Results manifest with per-setting cost and status metrics"
affects: [phase-2-data-collection]

# Tech tracking
tech-stack:
  added: [redis (auto-started for coop mode inter-agent messaging)]
  patterns: [sequential setting execution, background monitoring, long-running task patience]

key-files:
  created:
    - repos/CooperBench/logs/command-a-solo/ (100 result.json files)
    - repos/CooperBench/logs/command-a-coop-comm/ (100 result.json + 100 conversation.json)
    - repos/CooperBench/logs/command-a-coop-nocomm/ (100 result.json + 100 conversation.json)
    - repos/CooperBench/logs/results-manifest.json
  modified: []

key-decisions:
  - "Agent budget (step_limit=100, cost_limit=0) produces high submission rates (98-100%) with avg $0.46/task"
  - "Coop tasks with 2 agents each using 100-step limit occasionally take 60+ minutes for a single pair"
  - "Redis auto-started by cooperbench for coop mode; no manual setup needed"
  - "0 infrastructure errors across all 300 task-pair executions"

patterns-established:
  - "Solo first, then coop: validates pipeline at scale before committing to 2x agent runs"
  - "One long-tail task per run: expect 99/100 pairs to finish quickly, with 1 outlier taking 60+ minutes"

requirements-completed: [EXEC-01, EXEC-03, EXEC-04, EXEC-05]

# Metrics
duration: 290min
completed: 2026-02-18
---

# Phase 1 Plan 3: Full Benchmark Runs Summary

**300 task-pair executions across 3 settings (solo, coop-comm, coop-nocomm) on 100-pair lite subset using Command A with mini_swe_agent, $138.59 total cost, 0 infrastructure errors**

## Performance

- **Duration:** 290 minutes (~4.8 hours)
- **Started:** 2026-02-18T04:19:06Z
- **Completed:** 2026-02-18T09:09:06Z
- **Tasks:** 2
- **Files created:** 300 result.json + 200 conversation.json + 1 results-manifest.json

## Accomplishments

- Completed solo mode benchmark on all 100 lite subset pairs (45 minutes, $45.77)
- Completed coop-with-communication benchmark on all 100 pairs (120 minutes, $47.32)
- Completed coop-without-communication benchmark on all 100 pairs (110 minutes, $45.50)
- Zero infrastructure errors across all 300 executions -- no retries needed
- All conversation.json files present in both coop settings (428 messages in comm mode, 0 in no-comm mode)
- Results manifest created capturing per-setting metrics for downstream analysis

## Task Commits

Both tasks were long-running benchmark executions. The primary deliverable is the generated result data in the logs directory rather than code changes.

1. **Task 1: Run solo mode on full lite subset (100 pairs)** -- Solo completed, 100/100 results
2. **Task 2: Run coop-comm and coop-nocomm on full lite subset (100 pairs each)** -- Both completed, 100/100 each

## Results Summary

| Setting     | Pairs | Agents | Submitted | LimitsExceeded | Patches | Cost     | Avg Cost | Avg Steps |
|-------------|-------|--------|-----------|----------------|---------|----------|----------|-----------|
| Solo        | 100   | 100    | 98        | 2              | 82      | $45.77   | $0.46    | 10.4      |
| Coop-Comm   | 100   | 200    | 200       | 0              | 98      | $47.32   | $0.47    | 21.4      |
| Coop-NoComm | 100   | 200    | 199       | 1              | 93      | $45.50   | $0.46    | 17.3      |
| **TOTAL**   | **300** | **500** |         |                |         | **$138.59** |        |           |

### Key Observations

1. **High submission rate:** 497/500 agents (99.4%) submitted patches. Only 3 hit LimitsExceeded (2 solo, 1 coop-nocomm).
2. **Patch production:** 82% of solo, 98% of coop-comm, 93% of coop-nocomm pairs produced patches.
3. **Communication overhead:** Coop-comm used 21.4 avg steps vs 17.3 for coop-nocomm and 10.4 for solo. Messaging adds ~4 steps on average.
4. **Messages:** 428 total inter-agent messages in coop-comm (avg 4.3 per pair, max 11). Zero in coop-nocomm (as expected).
5. **Cost parity:** All three settings had nearly identical per-pair cost (~$0.46), despite coop using 2 agents. This is because coop agents took fewer steps per agent.
6. **Patch volume:** Solo produced 44K patch lines, coop-comm 69K, coop-nocomm 133K. The no-comm setting produced significantly more patch lines, suggesting uncoordinated agents make more (potentially overlapping) changes.

### Wall Time Breakdown

| Setting     | Wall Time | Notes |
|-------------|-----------|-------|
| Solo        | 45 min    | Fast due to single agent per pair |
| Coop-Comm   | 120 min   | One outlier task (pallets_jinja 1621) took 60+ minutes |
| Coop-NoComm | 110 min   | One outlier task (pillow 25) took 60+ minutes |
| **Total**   | **290 min** | Including orchestrator overhead |

## Infrastructure Details

- **Docker backend:** All 26 lite subset images available, containers managed by cooperbench CLI
- **Redis:** Auto-started by cooperbench for coop mode inter-agent messaging (container: `cooperbench-redis`)
- **Concurrency:** 4 parallel task-pairs at a time
- **Retry logic:** Orchestrator script configured for 3 attempts, but 0 retries were needed
- **infra_error tagging:** Applied to all 300 results -- 0 infrastructure errors detected

## Result Data Locations

- **Solo:** `repos/CooperBench/logs/command-a-solo/solo/` (100 result.json)
- **Coop-Comm:** `repos/CooperBench/logs/command-a-coop-comm/coop/` (100 result.json + 100 conversation.json)
- **Coop-NoComm:** `repos/CooperBench/logs/command-a-coop-nocomm/coop/` (100 result.json + 100 conversation.json)
- **Manifest:** `repos/CooperBench/logs/results-manifest.json`

## Decisions Made

- **Agent budget validated:** With step_limit=100 and cost_limit=0 (disabled), 99.4% of agents complete and submit. The $1/task ceiling from the smoke test (01-02) was caused by a lower step limit; with 100 steps the avg cost is $0.46/task.
- **Redis auto-management:** Cooperbench auto-starts Redis for coop mode. No manual setup was needed -- resolving the pending blocker from STATE.md.
- **Conversation files in nocomm:** Coop-nocomm still generates conversation.json files (they contain empty arrays), which is consistent behavior from the framework.

## Deviations from Plan

None -- plan executed exactly as written. All 3 settings completed on 100 pairs each, with the orchestrator script handling everything end-to-end.

## Issues Encountered

- **Long-tail tasks:** Each setting had 1-2 tasks that took 60+ minutes (compared to ~2 min average). These were pallets_jinja/1621 and pillow/25 tasks where agents used nearly all 100 steps. This is expected behavior -- some tasks are genuinely difficult.
- **Docker "container is not running" 409 errors:** Same benign errors seen in smoke test (01-02). These occur when the agent's step limit is reached and the container exits before all pending exec commands complete. No impact on results.

## Next Phase Readiness

- All raw benchmark data is ready for Phase 2 (Results Collection and Data Foundation)
- Solo results enable difficulty score computation: d(t) = 1 - Solo(t)
- Coop-comm conversation.json files enable communication analysis (Phase 3, Figures 5 and 6)
- Coop-comm vs coop-nocomm comparison enables merge conflict analysis (Phase 3, Figure 5)

## Self-Check: PASSED

- FOUND: logs/command-a-solo/ (100 result.json files)
- FOUND: logs/command-a-coop-comm/ (100 result.json files)
- FOUND: logs/command-a-coop-nocomm/ (100 result.json files)
- FOUND: logs/command-a-coop-comm/ (100 conversation.json files)
- FOUND: logs/command-a-coop-nocomm/ (100 conversation.json files)
- FOUND: results-manifest.json
- FOUND: 01-03-SUMMARY.md

---
*Phase: 01-execution-infrastructure*
*Completed: 2026-02-18*
