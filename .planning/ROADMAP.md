# Roadmap: CooperBench Reproduction (Figures 4, 5, 6)

## Overview

Reproduce Figures 4, 5, and 6 from the CooperBench paper (arXiv:2601.13295) using Cohere's Command A model. The pipeline runs benchmark experiments in three settings (solo, coop-no-comm, coop-with-comm), collects and normalizes results, computes statistical metrics, generates publication-quality figures, and adds qualitative transcript analysis. Five phases follow the data dependency chain: execution produces raw results, collection normalizes them, analysis computes metrics, figure generation visualizes them, and enhancements add depth.

**Depth:** Standard (5 phases)
**Coverage:** 30/30 v1 requirements mapped

---

## Phase 1: Execution Infrastructure

**Goal:** Command A benchmark runs complete across all three experimental settings with reliable infrastructure and cost controls.

**Dependencies:** None (first phase)

**Requirements:** EXEC-01, EXEC-02, EXEC-03, EXEC-04, EXEC-05, EXEC-06, EXEC-07

**Plans:** 3 plans

Plans:
- [x] 01-01-PLAN.md -- Install cooperbench CLI, pull/verify all 26 Docker images, configure environment
- [x] 01-02-PLAN.md -- Create orchestrator script with retry logic, run smoke test (5 pairs)
- [x] 01-03-PLAN.md -- Run full benchmark: solo, coop-comm, coop-nocomm (100 pairs each)

**Success Criteria:**
1. All 26 lite subset Docker images are available locally (pulled or built), verified by `docker images | grep cooperbench` showing all 26 task images.
2. A smoke test of 5 pairs completes end-to-end through the upstream CooperBench CLI with Docker backend, producing log directories with test results.
3. Solo, coop-no-comm, and coop-with-comm runs on the full lite subset (100 pairs each) complete with results written to the logs directory.
4. Infrastructure failures (Docker OOM, timeouts) are automatically retried and tagged as `infra_error`, distinguishable from genuine test failures in the output.
5. Cumulative API cost is tracked and reported per run (no budget ceiling — costs are monitored but do not halt execution, per user decision).

---

## Phase 2: Results Collection and Data Foundation

**Goal:** Raw benchmark logs are normalized into a unified data store with difficulty scores computed and validated, ready for downstream analysis.

**Dependencies:** Phase 1 (requires benchmark results in logs directory)

**Requirements:** DATA-01, DATA-02, DATA-03, FIG4-01, FIG4-02

**Plans:** 3/3 plans complete

Plans:
- [x] 02-01-PLAN.md -- Fix Docker eval backend entrypoint bug, run evaluation on all 300 existing benchmark results
- [x] 02-02-PLAN.md -- Run 2 additional solo seeds (3 total) and evaluate them for difficulty scoring
- [x] 02-03-PLAN.md -- Build collection script, compute difficulty scores, produce unified data/results.json

**Success Criteria:**
1. A unified JSON results store exists containing all benchmark outcomes from all three settings, with each record including task ID, setting, merge outcome, test outcome, and status (pass/fail/infra_error).
2. Merge outcomes and test outcomes are tracked as independent dimensions: `{merge_clean, merge_union, merge_failed} x {tests_pass, tests_fail}`.
3. Per-task difficulty scores d(t) = 1 - Solo(t) are computed and each task is assigned to one of 10 equal-width buckets over [0,1], with bucket population counts reported.
4. Infrastructure errors are excluded from metric computations and their count is reported separately.

---

## Phase 3: Analysis Modules

**Goal:** All statistical metrics required for Figures 4, 5, and 6 are computed from the normalized data store, with each figure's analysis independently testable.

**Dependencies:** Phase 2 (requires normalized results store and difficulty scores)

**Requirements:** FIG4-03, FIG4-04, FIG4-05, FIG4-06, FIG5-01, FIG5-02, FIG5-03, FIG5-04, FIG6-01, FIG6-02

**Plans:** 3/3 plans complete

Plans:
- [x] 03-01-PLAN.md -- Compute Figure 4 metrics: per-bucket success rates, Wilson CIs, AUC, retention
- [x] 03-02-PLAN.md -- Compute Figure 5 metrics: success rates, merge conflict rates, speech acts, overhead
- [x] 03-03-PLAN.md -- Classify communication errors using LLM-based C1a-C4b taxonomy classifier

**Success Criteria:**
1. Per-bucket solo and coop success rates with Wilson 95% confidence intervals are computed for all populated difficulty buckets, and AUC (trapezoidal integration) and retention (AUC_coop / AUC_solo) metrics are available as named outputs.
2. Comm vs no-comm success rates and merge conflict rates are computed with differences reported, ready for bar chart and significance testing.
3. Agent messages from coop-with-comm transcripts are classified into speech act types (plan/question/update) and communication overhead is expressed as a percentage of total action budget.
4. An LLM-based classifier labels communication errors in all coop-with-comm transcripts using the paper's C1a/C1b/C2/C3b/C4a/C4b taxonomy, with per-category frequency counts available.

**Parallelization note:** The three analysis modules (Figure 4 stats, Figure 5 communication, Figure 6 failure modes) have no cross-dependencies and can be built/executed in parallel.

---

## Phase 4: Figure Generation and Paper Comparison

**Goal:** Publication-quality Figures 4, 5, and 6 are generated as PDF/PNG files, with paper baseline numbers overlaid for direct visual comparison.

**Dependencies:** Phase 3 (requires all computed metrics from analysis modules)

**Requirements:** FIG4-07, FIG5-05, FIG6-03, COMP-01

**Success Criteria:**
1. Figure 4 is saved as PDF/PNG at 300+ DPI showing difficulty-stratified success curves for solo and coop settings, with Wilson CI shaded bands, AUC values annotated, and retention metric displayed.
2. Figure 5 is saved as a 3-panel plot: (a) comm vs no-comm success rates, (b) merge conflict rates with/without communication, (c) communication overhead breakdown by speech act type.
3. Figure 6 is saved as a bar chart showing communication error frequency by taxonomy category (C1a, C1b, C2, C3b, C4a, C4b).
4. The paper's published baseline numbers are overlaid on all three figures, enabling direct visual comparison between Command A results and the paper's reported values.

---

## Phase 5: Qualitative Transcript Analysis

**Goal:** Qualitative metrics from communication transcripts are computed and summarized, revealing structural patterns that correlate with cooperation outcomes.

**Dependencies:** Phase 2 (requires normalized results store with transcripts); can partially overlap with Phase 4.

**Requirements:** QUAL-01, QUAL-02, QUAL-03, QUAL-04

**Success Criteria:**
1. Plan:Question ratio is computed per trajectory and its correlation with merge conflict outcomes is reported (e.g., higher planning correlates with fewer conflicts).
2. Trajectories with first-turn planning (Plan message in the first turn) are identified, and the conflict rate reduction compared to trajectories without first-turn planning is quantified.
3. Specificity metrics (line number mentions, file path mentions) are counted per trajectory, distinguishing high-specificity from low-specificity communication.
4. A summary table is generated comparing all qualitative metrics (Plan:Question ratio, first-turn planning rate, specificity counts) for conflict vs no-conflict trajectory groups.

---

## Progress

| Phase | Name | Status | Requirements | Plans |
|-------|------|--------|--------------|-------|
| 1 | Execution Infrastructure | Complete | 7 | 3/3 |
| 2 | Results Collection and Data Foundation | Complete | 5 | 3/3 |
| 3 | Analysis Modules | Complete | 10 | 3/3 |
| 4 | Figure Generation and Paper Comparison | Not Started | 4 | -- |
| 5 | Qualitative Transcript Analysis | Not Started | 4 | -- |

**Total:** 22/30 requirements complete

---
*Roadmap created: 2026-02-14*
*Last updated: 2026-02-18 (03-03 complete: Phase 3 complete)*
