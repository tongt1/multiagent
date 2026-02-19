---
phase: 02-results-collection-and-data-foundation
plan: 02
subsystem: benchmarks
tags: [cooperbench, solo, difficulty, seeds]

# Dependency graph
requires:
  - plan: 02-01
    provides: "Fixed Docker eval backend with entrypoint override"
provides:
  - "200 additional solo result.json + eval.json files (2 seeds x 100 pairs)"
  - "3 total solo data points per pair for continuous difficulty scoring"
consumed_by:
  - plan: 02-03
    needs: "All solo seed eval.json files for difficulty computation"
---

# Plan 02-02 Summary: Additional Solo Seeds for Difficulty Scoring

## What was built

Ran 2 additional solo benchmark seeds (command-a-solo-seed1 and command-a-solo-seed2) on the full lite subset (100 feature pairs each), then evaluated both using the Docker backend (with entrypoint fix from Plan 02-01). Combined with the original solo run, this provides 3 independent data points per feature pair for computing continuous difficulty scores.

## Outcome

**Status:** Complete
**Duration:** ~150 min (2 benchmark runs ~45 min each + eval ~30 min + overhead)
**Cost:** ~$89 additional ($44.50/seed based on submission logs)

## Key results

### Benchmark Runs
| Seed | Run Name | Result Files | Submitted |
|------|----------|-------------|-----------|
| 0 | command-a-solo (original) | 100 result.json | 98/100 |
| 1 | command-a-solo-seed1 | 100 result.json | 100/100 |
| 2 | command-a-solo-seed2 | 100 result.json | 100/100 |

### Evaluation Results
| Seed | Eval Files | Passed | Failed | Errors | Pass Rate |
|------|-----------|--------|--------|--------|-----------|
| 1 | 100 eval.json | 1 | 95 | 4 | 1.0% |
| 2 | 100 eval.json | 1 | 95 | 4 | 1.0% |

### Pair Overlap Verification
- All 3 seeds cover identical 100 feature pairs (100% overlap confirmed)
- Deterministic pair assignment from lite subset definition

### Difficulty Distribution Preview (3 seeds combined)
| Difficulty | Count | Interpretation |
|-----------|-------|----------------|
| 0.0 | 0 | Easy (passed all 3 seeds) |
| 0.33 | 1 | Medium-easy (passed 2/3 seeds) |
| 0.67 | 1 | Medium-hard (passed 1/3 seeds) |
| 1.0 | 98 | Hard (passed 0/3 seeds) |

**Key finding:** 98% of pairs are maximally difficult (d=1.0) for Command A. The model rarely solves both features in a pair even in solo mode. This yields sparse difficulty buckets (only 3 of 10 populated with data), which is a known limitation of single-model reproduction vs the paper's multi-model approach.

## Decisions made

- Seeds run sequentially (not parallel) to avoid Docker resource contention
- Environment variables inherited from Phase 1 setup

## Key files

### created
- `repos/CooperBench/logs/command-a-solo-seed1/` (100 result.json + 100 eval.json)
- `repos/CooperBench/logs/command-a-solo-seed2/` (100 result.json + 100 eval.json)

### modified
None (no source code changes)

## Self-Check: PASSED

- [x] 200 result.json files (100 per seed)
- [x] 200 eval.json files (100 per seed)
- [x] All 3 seeds cover same 100 pairs (verified)
- [x] Difficulty distribution shows expected {0.33, 0.67, 1.0} values
- [x] Docker containers cleaned up after eval
