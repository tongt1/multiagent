# Project State

## Project Reference

See: /home/terry_tong_cohere_com/.planning/PROJECT.md (updated 2026-02-02)

**Core value:** Produce rigorous, reproducible comparisons between multi-agent debate RL and single-agent RLVR on math reasoning, with the only variable being how training data is generated (debate vs no debate).

**Current focus:** Phase 1 - Data Generation & Foundation

## Current Position

Phase: 1 of 4 (Data Generation & Foundation)
Plan: 2 of 4 in current phase
Status: In progress
Last activity: 2026-02-02 — Completed 01-02-PLAN.md (Ground truth rewards and early termination)

Progress: [██░░░░░░░░] 50% (Phase 1: 2/4 plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 2
- Average duration: 4 min
- Total execution time: 0.13 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-data-generation-foundation | 2 | 8 min | 4 min |

**Recent Trend:**
- Last 5 plans: 01-01 (4min), 01-02 (4min)
- Trend: Steady velocity at 4min/plan

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in /home/terry_tong_cohere_com/.planning/PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- POST_TRAINING as backend, not full replacement: Scaffold controls training loop for flexibility; POST_TRAINING handles infra concerns
- BEE for both reward and eval: Single eval system for consistency between training signal and benchmark evaluation
- Math-only for initial experiments: Verifiable rewards are cleanest for math; reduces variables
- Single-agent RLVR as sole baseline: Isolates the effect of multi-agent debate on training data quality
- REINFORCE/GRPO only initially: Simpler algorithms reduce debugging surface for initial comparison

**From 01-01 (MATH 500 loader):**
- Fixed random seed (42) for reproducible sampling: Ensures consistent 500 problems across experiments
- Filter before sampling: Remove ambiguous answers before stratification to guarantee all 500 are SymPy-verifiable
- Stratified sampling with redistribution: 100 per level with overflow redistribution maintains balance
- Cache to data/math500_cache.json: First load creates cache, subsequent loads instant

**From 01-02 (Ground truth rewards and early termination):**
- Binary ground truth reward (1.0/0.0) computed via compute_math_reward when problem_metadata contains ground_truth
- max_iterations changed from 7 to 5 as decided for Phase 1 MATH 500 generation
- PipelineConfig.mode field controls debate (1-solver) vs baseline (3-solver) architecture
- Termination metadata logged in trajectory for RLVR analysis of early vs max iteration stops

### Pending Todos

None yet.

### Blockers/Concerns

**From 01-01:**
- MATH dataset DMCA-restricted on HuggingFace: Local fallback implemented but may need local data/MATH/test/ files for production use beyond mocked tests

## Session Continuity

Last session: 2026-02-02 19:23:58 UTC
Stopped at: Completed 01-02-PLAN.md (Ground truth rewards and early termination)
Resume file: None
Next action: Execute plan 01-03 (MATH 500 dataset integration) or continue with remaining Phase 1 plans

---
*State initialized: 2026-02-02*
