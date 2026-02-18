# Requirements: CooperBench Reproduction

**Defined:** 2026-02-14
**Core Value:** Produce verifiable figures 4, 5, 6 from CooperBench paper using Command A, with qualitative transcript analysis

## v1 Requirements

Requirements for initial reproduction. Each maps to roadmap phases.

### Execution Infrastructure

- [x] **EXEC-01**: Pull/build all Docker images required for lite subset (26 tasks, 8 currently missing)
- [x] **EXEC-02**: Verify upstream CooperBench CLI works end-to-end with Docker backend on a smoke test (5 pairs)
- [x] **EXEC-03**: Run Command A in solo mode on full lite subset (100 pairs)
- [x] **EXEC-04**: Run Command A in coop mode with communication on full lite subset
- [x] **EXEC-05**: Run Command A in coop mode without communication on full lite subset
- [x] **EXEC-06**: Implement cost tracking per run (no budget ceiling — track and report costs but don't halt execution)
- [x] **EXEC-07**: Implement retry policy for infrastructure failures (Docker OOM, timeouts) with infra_error tagging

### Results Collection

- [x] **DATA-01**: Normalize upstream log directory structure into unified JSON results store
- [x] **DATA-02**: Distinguish infrastructure errors from genuine test failures in results
- [x] **DATA-03**: Track merge outcomes as separate dimension from test outcomes (merge_clean/merge_union/merge_failed x tests_pass/tests_fail)

### Figure 4: Difficulty-Stratified Success Curves

- [x] **FIG4-01**: Compute per-task difficulty score d(t) = 1 - Solo(t) in [0,1]
- [x] **FIG4-02**: Partition tasks into 10 equal-width buckets over [0,1]
- [ ] **FIG4-03**: Compute per-bucket solo and coop success rates
- [ ] **FIG4-04**: Compute Wilson 95% confidence intervals for all rates
- [ ] **FIG4-05**: Compute AUC via trapezoidal integration for solo and coop curves
- [ ] **FIG4-06**: Compute retention metric (AUC_coop / AUC_solo)
- [ ] **FIG4-07**: Generate Figure 4 with CI shaded bands, publication-quality PDF/PNG at 300+ DPI

### Figure 5: Communication Effects

- [ ] **FIG5-01**: Compute comm vs no-comm success rates
- [ ] **FIG5-02**: Compute merge conflict rates with and without communication
- [ ] **FIG5-03**: Classify agent messages into speech act types (plan/question/update)
- [ ] **FIG5-04**: Compute communication overhead as percentage of total action budget
- [ ] **FIG5-05**: Generate Figure 5 as 3-panel plot: (a) success rates, (b) conflict rates, (c) overhead breakdown

### Figure 6: Communication Error Taxonomy

- [ ] **FIG6-01**: Implement LLM-based communication error classifier using paper's taxonomy prompt (C1a unanswered no-reply, C1b unanswered ignored, C2 non-answer/vague, C3b incorrect claim corrected, C4a spammy same info, C4b spammy near-duplicate blocks)
- [ ] **FIG6-02**: Run classifier on all coop-with-comm transcripts
- [ ] **FIG6-03**: Generate Figure 6 as error frequency bar chart

### Qualitative Analysis

- [ ] **QUAL-01**: Compute Plan:Question ratio per trajectory and correlate with merge conflict outcomes
- [ ] **QUAL-02**: Detect first-turn planning (Plan message in first turn) and measure conflict rate reduction
- [ ] **QUAL-03**: Count specificity metrics per trajectory (line number mentions, file path mentions)
- [ ] **QUAL-04**: Generate summary table comparing qualitative metrics for conflict vs no-conflict trajectories

### Paper Comparison

- [ ] **COMP-01**: Overlay paper's published baseline numbers on Figures 4, 5, 6 for direct visual comparison

## v2 Requirements

Deferred to future work. Tracked but not in current roadmap.

### Extended Analysis

- **EXT-01**: Per-repository performance breakdown across 12 repos
- **EXT-02**: Spatial vs semantic coordination analysis (file-overlap reduction vs test-pass improvement)
- **EXT-03**: Multi-format export (PDF/SVG/CSV) for all figures and data
- **EXT-04**: Transcript excerpt extraction for paper-appendix-style evidence
- **EXT-05**: Automated figure validation (monotonicity, CI containment, retention bounds)

### Scale

- **SCALE-01**: Run full dataset (652 tasks) after lite validation
- **SCALE-02**: Multi-model comparison if additional Cohere models become available

## Out of Scope

| Feature | Reason |
|---------|--------|
| Custom pipeline wrapper for execution | Paper uses OpenHands agent SDK; our custom wrapper generates patches via raw prompts -- architecturally incomparable |
| Interactive Streamlit dashboard | Static figures are sufficient for reproduction; dashboard adds maintenance burden |
| Real-time experiment monitoring | Experiments run asynchronously; analysis is post-hoc |
| Figures 1, 2, 3 from paper | Not requested; focusing on figures 4, 5, 6 |
| Full upstream CooperBench fork | Build analysis as new module consuming results, not a fork |
| LLM-based speech act classifier | Paper uses heuristic patterns; regex/keyword heuristics are sufficient and deterministic |
| Automated LaTeX generation | Figures + data are sufficient; LaTeX authoring is manual |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| EXEC-01 | Phase 1 | Complete |
| EXEC-02 | Phase 1 | Complete |
| EXEC-03 | Phase 1 | Complete |
| EXEC-04 | Phase 1 | Complete |
| EXEC-05 | Phase 1 | Complete |
| EXEC-06 | Phase 1 | Complete |
| EXEC-07 | Phase 1 | Complete |
| DATA-01 | Phase 2 | Complete |
| DATA-02 | Phase 2 | Complete |
| DATA-03 | Phase 2 | Complete |
| FIG4-01 | Phase 2 | Complete |
| FIG4-02 | Phase 2 | Complete |
| FIG4-03 | Phase 3 | Pending |
| FIG4-04 | Phase 3 | Pending |
| FIG4-05 | Phase 3 | Pending |
| FIG4-06 | Phase 3 | Pending |
| FIG4-07 | Phase 4 | Pending |
| FIG5-01 | Phase 3 | Pending |
| FIG5-02 | Phase 3 | Pending |
| FIG5-03 | Phase 3 | Pending |
| FIG5-04 | Phase 3 | Pending |
| FIG5-05 | Phase 4 | Pending |
| FIG6-01 | Phase 3 | Pending |
| FIG6-02 | Phase 3 | Pending |
| FIG6-03 | Phase 4 | Pending |
| QUAL-01 | Phase 5 | Pending |
| QUAL-02 | Phase 5 | Pending |
| QUAL-03 | Phase 5 | Pending |
| QUAL-04 | Phase 5 | Pending |
| COMP-01 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 30 total
- Mapped to phases: 30
- Unmapped: 0

---
*Requirements defined: 2026-02-14*
*Last updated: 2026-02-18 after 02-03-PLAN.md completion (Phase 2 complete: DATA-01, DATA-02, DATA-03, FIG4-01, FIG4-02)*
