# Project Research Summary

**Project:** CooperBench Paper Reproduction (Figures 4, 5, 6)
**Domain:** Scientific benchmark reproduction -- statistical analysis and figure generation for multi-agent LLM cooperation evaluation
**Researched:** 2026-02-14
**Confidence:** HIGH

## Executive Summary

This project reproduces the core figures (4, 5, 6) from the CooperBench paper (arXiv:2601.13295) using Cohere's Command A model. The paper evaluates multi-agent LLM cooperation on software engineering tasks, measuring success rates across difficulty levels (Figure 4), communication effects on coordination (Figure 5), and failure mode taxonomy (Figure 6). The recommended approach is a two-layer architecture: (1) an execution layer that invokes the **upstream CooperBench CLI directly** via subprocess for benchmark runs across three settings (solo, coop-with-comm, coop-no-comm), and (2) an analysis layer with modular components per figure, backed by a standard scientific Python stack (numpy, pandas, scipy, statsmodels, matplotlib). The existing `cooperbench-eval` package provides reusable failure mode classifiers for Figure 6. The existing custom wrapper pipeline (`src/evaluation/cooperbench/pipeline.py`) should NOT be used for execution because it differs architecturally from the paper's agent SDK.

The critical risk is **comparability**: reproducing with a single model (Command A) against the paper's multi-model design means difficulty scores, absolute success rates, and failure mode distributions will differ. The mitigation is to focus on reproducing **relative patterns** (retention ratios, difficulty curves, communication effects) rather than matching absolute numbers, while clearly documenting deviations. Secondary risks include Docker infrastructure failures at scale (300+ container runs), sparse difficulty buckets from the 100-pair lite subset, and LLM classifier noise in the failure taxonomy. All are manageable with the prevention strategies identified in the pitfalls research.

The stack is well-understood and mostly already in the project's dependency tree. The analysis work is primarily statistical computation and visualization -- no novel engineering is required. The main complexity lies in correctly orchestrating the upstream CLI, normalizing its output format, and implementing the paper's exact statistical methodology (Wilson CIs, trapezoidal AUC, equal-width bucketing).

## Key Findings

### Recommended Stack

The analysis pipeline adds a focused set of scientific Python libraries on top of the project's existing dependencies. See [STACK.md](./STACK.md) for full details.

**Core technologies:**
- **statsmodels** (>=0.14.0): Wilson score confidence intervals via `proportion_confint(method='wilson')` -- the only library with a one-call implementation, critical for all error bars in Figures 4/5
- **scipy** (>=1.14.0): Trapezoidal AUC integration via `scipy.integrate.trapezoid` -- canonical API for the paper's primary quantitative metric
- **pandas** (>=2.2.0, <3.0): DataFrame operations for task results, difficulty scores, and aggregation -- pin below 3.0 to avoid breaking changes (pandas 3.0 released only 3 weeks ago; statsmodels/seaborn untested against it)
- **matplotlib** (>=3.9.0): Publication-quality 3-panel figure generation -- already lazy-imported in the codebase
- **seaborn** (>=0.13.0) + **SciencePlots** (>=2.2.0): Plot aesthetics and publication styling -- optional convenience, not hard dependencies

**Already in project (no changes needed):** instructor, litellm, pydantic, cohere SDK, loguru, pytest, ruff, mypy.

**Key version decision:** Pin `pandas<3.0`. Copy-on-Write defaults, PyArrow-backed strings, and NaN/NA unification in pandas 3.0 are too risky for a correctness-critical reproduction pipeline.

### Expected Features

The feature landscape maps directly to the three paper figures. See [FEATURES.md](./FEATURES.md) for the complete feature matrix with dependencies.

**Must have (table stakes -- figures are invalid without these):**
- F16: Results JSONL ingestion (pipeline entry point)
- F1+F2: Difficulty score computation + equal-width bucketing (Figure 4 x-axis)
- F3: Wilson 95% confidence intervals (all error bars)
- F4+F5+F6: Per-bucket success rates, AUC, retention metric (Figure 4 content)
- F7+F8: Comm vs no-comm comparison, merge conflict rates (Figure 5a/b)
- F9+F10: Speech act classification, communication overhead (Figure 5c)
- F11+F12: Error taxonomy classifier + bar chart (Figure 6)
- F17: Matplotlib figure generation for all three figures
- F13+F14+F15: Experiment runs in solo, coop-comm-on, coop-comm-off settings

**Should have (adds reproduction value, low effort):**
- D4: Paper baseline overlay on figures (direct visual comparison)
- D7: Wilson CI shaded bands on Figure 4
- D1+D2: Plan:Question ratio and first-turn planning analysis (key qualitative findings)
- D3: Specificity metrics (regex-based, low effort)
- D8: Statistical significance tests on Figure 5 differences

**Defer (v2+):**
- D5: Per-repository breakdown
- D6: Spatial vs semantic coordination analysis (HIGH complexity)
- D10: Multi-format export (PDF/SVG/CSV)
- D11: Transcript excerpt extraction
- D9: Automated figure validation

### Architecture Approach

The architecture is a clean two-layer design: execution via the upstream CooperBench CLI (subprocess orchestration), and analysis via modular Python components per figure. See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full system diagram and data flow.

**Major components:**
1. **Experiment Orchestrator** (`scripts/run_experiment.py`) -- invokes `cooperbench run` for 3 settings across the lite subset, manages retries and progress
2. **Results Collector** (`src/analysis/collector.py`) -- normalizes upstream `logs/` directory structure into a unified JSON results store, adapts to `cooperbench-eval` TaskData schema
3. **Difficulty Scorer** (`src/analysis/difficulty.py`) -- computes per-task difficulty from solo results, buckets into 10 equal-width bins, calculates Wilson CIs and AUC
4. **Communication Analyzer** (`src/analysis/communication.py`) -- extracts speech acts, Plan:Question ratios, first-turn planning, specificity metrics from transcripts
5. **Failure Mode Classifier** (extends existing `cooperbench-eval/`) -- maps existing 10 classifiers to paper's C1a-C4b taxonomy codes
6. **Figure Generator** (`src/analysis/figures.py`) -- produces publication-quality matplotlib output for all three figures

**Key architectural decision:** Use the upstream CooperBench CLI as-is (subprocess), NOT the custom wrapper pipeline. The custom wrapper uses prompt-based patch generation without real tool use, producing results that are architecturally incomparable to the paper.

### Critical Pitfalls

Seven pitfalls identified, three of which are critical. See [PITFALLS.md](./PITFALLS.md) for full analysis and recovery strategies.

1. **Difficulty score mismatch** -- Paper averages d(t) across 5 models; we compute from Command A solo only. Mitigation: label axes as "Difficulty (Command A solo)", consider gold conflict rate as model-independent proxy.
2. **Docker container failures counted as test failures** -- Infrastructure errors (OOM, disk, timeout) inflate failure rates. Mitigation: tag results with `pass/fail/infra_error/timeout` status, implement retry policy, audit infra_error rate before computing metrics.
3. **Wilson CI degenerate buckets** -- 100 pairs across 10 buckets yields ~10 per bucket with some empty. Mitigation: consider adaptive (quantile) bucketing, report sample sizes, handle empty buckets via interpolation.
4. **LLM classifier noise** -- Stochastic classification shifts failure mode frequencies on small samples. Mitigation: majority vote (3 runs), manual ground truth on 20 transcripts, flag low-prevalence categories.
5. **Merge conflict conflation** -- Merge failures and test failures must be tracked as separate dimensions. Mitigation: record `{merge_clean, merge_union, merge_failed} x {tests_pass, tests_fail}`.

## Implications for Roadmap

Based on combined research, the project naturally decomposes into 5 phases following the data dependency chain: execution infrastructure must come first, then data normalization, then parallel analysis modules, then visualization, and finally enhancement.

### Phase 1: Execution Infrastructure
**Rationale:** Everything downstream depends on having benchmark results. The upstream CLI must be set up and validated before any analysis can begin. This phase also resolves the critical architecture decision (use upstream CLI, not custom wrapper).
**Delivers:** Working experiment orchestrator, 3-setting execution capability, cost tracking, Docker health monitoring
**Addresses:** F13 (solo runs), F14 (coop-comm-on), F15 (coop-comm-off)
**Avoids:** Pitfall 2 (Docker failures), Pitfall 6 (agent SDK mismatch), Pitfall 7 (cost blowup)
**Key tasks:** Install upstream CLI as submodule, write orchestrator script, implement infra_error tagging and retry policy, set budget ceiling, run smoke test on 5 pairs

### Phase 2: Results Collection and Data Foundation
**Rationale:** Analysis components cannot be built or tested without normalized data. The results collector is the single normalization layer that all analyzers depend on. Difficulty scoring goes here because it depends on solo results AND gates Figure 4 analysis.
**Delivers:** Unified results store, log-to-JSON normalization, difficulty scoring with bucketing, JSONL ingestion
**Addresses:** F16 (results ingestion), F1 (difficulty scores), F2 (bucketing)
**Avoids:** Pitfall 1 (difficulty score mismatch -- must decide methodology here), Pitfall 3 (degenerate buckets -- must validate bucket populations)
**Uses:** pandas, numpy (from STACK.md)
**Implements:** Results Collector, Difficulty Scorer (from ARCHITECTURE.md)

### Phase 3: Analysis Modules (Parallelizable)
**Rationale:** Once normalized data exists, the three analysis modules (Figure 4 stats, Figure 5 communication, Figure 6 failure modes) can be built in parallel because they have no cross-dependencies. This is the core statistical computation phase.
**Delivers:** Per-bucket success rates, Wilson CIs, AUC/retention, communication analysis, failure taxonomy classification
**Addresses:** F3 (Wilson CIs), F4 (per-bucket rates), F5 (AUC), F6 (retention), F7 (comm vs no-comm), F8 (merge conflict rates), F9 (speech act classifier), F10 (comm overhead), F11 (error taxonomy)
**Avoids:** Pitfall 3 (Wilson CI edge cases), Pitfall 4 (LLM classifier noise), Pitfall 5 (merge conflict conflation)
**Uses:** statsmodels, scipy, instructor (from STACK.md)
**Implements:** Communication Analyzer, Failure Mode Bridge (from ARCHITECTURE.md)

### Phase 4: Figure Generation
**Rationale:** Visualization depends on all analysis outputs. Build after the computation modules produce validated intermediate data.
**Delivers:** Publication-quality Figures 4, 5, 6 as PDF/PNG, end-to-end generation script
**Addresses:** F12 (Figure 6 chart), F17 (all figure generation), D7 (CI shaded bands)
**Uses:** matplotlib, seaborn, SciencePlots (from STACK.md)
**Implements:** Figure Generator (from ARCHITECTURE.md)

### Phase 5: Enhancements and Comparison
**Rationale:** Once core figures are validated, add the differentiator features that make this a compelling reproduction rather than just a mechanical replication.
**Delivers:** Paper baseline overlay, Plan:Question ratio analysis, first-turn planning effect, specificity metrics, statistical significance tests
**Addresses:** D4 (paper overlay), D1 (Plan:Question ratio), D2 (first-turn planning), D3 (specificity), D8 (significance tests), D10 (multi-format export)
**Avoids:** Pitfall 1 (difficulty comparability -- overlay paper's own data for direct comparison)

### Phase Ordering Rationale

- **Phases 1-2 are strictly sequential** because analysis requires normalized data, which requires benchmark results. No parallelization opportunity here.
- **Phase 3 components are parallelizable** because each analyzer reads from the unified results store independently. This is a deliberate architecture choice (Pattern 3 from ARCHITECTURE.md: analysis module per figure).
- **Phase 4 depends on all of Phase 3** because the figure generator takes data dicts from all three analyzers.
- **Phase 5 is additive** and can be partially done alongside Phase 4 (e.g., paper overlay can be added to figures as they're built).
- **Pitfalls are front-loaded**: the most critical pitfalls (Docker failures, agent SDK mismatch, cost blowup) are addressed in Phase 1; statistical pitfalls (Wilson CI, difficulty scoring) in Phase 2-3; classification noise in Phase 3.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1:** Upstream CooperBench CLI setup and Docker image management. The upstream repo has its own installation requirements (Modal, Redis, Docker images per task) that need investigation. The CLI's concurrency model and error handling need testing.
- **Phase 3 (Failure Mode Bridge):** Mapping existing 10-classifier taxonomy to paper's C1a-C4b codes requires understanding both taxonomies in detail. The existing `unverifiable_claims` LLM judge partially overlaps with C3b but frames the task differently.

Phases with standard patterns (skip research-phase):
- **Phase 2:** Results collection and difficulty scoring use standard file I/O and numpy/pandas operations. Well-documented patterns.
- **Phase 4:** Figure generation with matplotlib is thoroughly documented. The `SciencePlots` library and `subplot_mosaic` patterns are standard.
- **Phase 5:** Enhancement features are straightforward statistical computations and regex-based text analysis.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All libraries verified against PyPI and official docs. Version compatibility confirmed. pandas<3.0 pin is a clear, well-justified decision. |
| Features | HIGH | Features map directly to paper figures with clear traceability. Dependency graph is well-understood. MVP vs defer boundary is clean. |
| Architecture | HIGH | Based on direct inspection of upstream repo, existing codebase, and cooperbench-eval package. The subprocess-orchestration pattern is the right call. |
| Pitfalls | HIGH | Critical pitfalls verified against paper methodology, codebase inspection, and domain literature. Recovery strategies are concrete. |

**Overall confidence:** HIGH

The research draws from the paper itself, the upstream codebase, the existing cooperbench-eval package, and verified library documentation. The main source of uncertainty is operational: how the upstream CLI behaves at scale with Command A, and whether Docker infrastructure is reliable for 300+ runs. This is knowable only through execution, not more research.

### Gaps to Address

- **Upstream CLI installation and compatibility:** The upstream repo may have pinned dependencies that conflict with ours. Need to test `pip install -e repos/CooperBench` in the project's virtual environment.
- **Docker image availability:** Task-specific Docker images (`akhatua/cooperbench-*:task{id}`) need to be pulled or built. Build time and registry access are unknown.
- **Redis requirement for coop mode:** The upstream CLI requires Redis for inter-agent messaging. Need to verify Redis setup and whether it can be dockerized alongside the evaluation containers.
- **Cohere API staging vs production:** The current codebase defaults to `stg.api.cohere.com`. If Command A is only available on production, endpoint configuration needs updating.
- **Gold conflict report completeness:** `gold_conflict_report.json` is referenced but its coverage of the lite subset (100 pairs) needs verification.
- **Paper's exact difficulty scores:** If the paper authors release per-pair difficulty data, we could use their x-axis directly rather than computing a single-model proxy. Check for supplementary materials.

## Sources

### Primary (HIGH confidence)
- [CooperBench paper (arXiv:2601.13295)](https://arxiv.org/abs/2601.13295) -- methodology, figure specifications, statistical approach
- [statsmodels proportion_confint (0.14.6)](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html) -- Wilson CI API
- [scipy.integrate.trapezoid (1.17.0)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.trapezoid.html) -- AUC computation API
- [pandas 3.0 What's New](https://pandas.pydata.org/docs/whatsnew/v3.0.0.html) -- breaking changes justifying <3.0 pin
- Upstream CooperBench repo: `repos/CooperBench/` -- CLI, runner, eval, dataset structure
- Existing cooperbench-eval package: `cooperbench-eval/src/` -- classifiers, report generator, data schemas

### Secondary (MEDIUM confidence)
- [SciencePlots 2.2.0 (PyPI)](https://pypi.org/project/SciencePlots/) -- publication styling (not verified against our specific matplotlib version)
- [Rating Roulette: Self-Inconsistency in LLM-As-A-Judge (EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.1361.pdf) -- LLM judge reliability
- [instructor (python-useinstructor.com)](https://python.useinstructor.com/) -- Cohere integration for structured output

### Tertiary (needs validation during execution)
- Docker image availability and build times for CooperBench task images
- Cohere Command A pricing and rate limits for cost estimation
- Redis inter-agent messaging stability under concurrent evaluation load

---
*Research completed: 2026-02-14*
*Ready for roadmap: yes*
