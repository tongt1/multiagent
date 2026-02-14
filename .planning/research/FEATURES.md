# Feature Research

**Domain:** Paper figure reproduction pipeline (CooperBench arxiv:2601.13295, Figures 4/5/6)
**Researched:** 2026-02-14
**Confidence:** HIGH

## Feature Landscape

### Table Stakes (Results Invalid Without These)

Features the paper's methodology requires. Missing any of these means figures are wrong or non-reproducible.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **F1: Difficulty score computation** | Figure 4 axis. `d(t) = 1 - Solo(t)` per-task, rescaled to [0,1]. Single-model variant since we only have Command A. | LOW | Paper uses multi-model average; we use single-model solo success rate as proxy. Document deviation. |
| **F2: Equal-width bucket partitioning** | Algorithm 1 in paper. 10 bins over [0,1], tasks assigned by difficulty score. | LOW | Straightforward `numpy.digitize` or manual bin edges at 0.1 increments. |
| **F3: Wilson 95% confidence intervals** | Paper specifies Wilson intervals because they "remain well-calibrated near 0 and 1" (unlike Wald). All error bars in Figs 4/5 use these. | LOW | `statsmodels.stats.proportion.proportion_confint(method='wilson')` or ~15-line manual implementation. |
| **F4: Per-bucket solo and coop success rates** | Core data for Figure 4 curves. Each bucket gets `n_passed / n_total` for solo and coop modes separately. | LOW | Group results.jsonl by bucket, count `both_passed == true`. |
| **F5: AUC computation (trapezoidal)** | Paper's primary quantitative metric. `AUC_solo` and `AUC_coop` computed via trapezoidal integration over the 10-bucket curves. | LOW | `numpy.trapz` over bucket midpoints. |
| **F6: Retention metric** | `Retention = AUC_coop / AUC_solo`. Key summary statistic quoted per model. Paper shows GPT-5: 0.64, Claude: 0.60, MiniMax: 0.46. | LOW | Trivial division once F5 exists. |
| **F7: Comm vs no-comm success rate comparison** | Figure 5(a). Requires running experiments with `messaging_enabled: true` vs `messaging_enabled: false`. | MEDIUM | Not an analysis feature per se -- requires two experiment runs. Analysis is just grouped bar chart. Existing config supports this toggle. |
| **F8: Merge conflict rate computation** | Figure 5(b). `n_conflict / n_total` for comm-on vs comm-off. Conflict = `merge_status != "clean"` in eval results. | LOW | Already tracked in `CooperBenchEvalResult.merge_status`. |
| **F9: Speech act classification** | Figure 5(c). Classify each agent message as plan/question/update. Paper shows ~1/3 each. | MEDIUM | Question detection exists in `schemas.py` (`is_question` property). Plan/update classification needs regex or keyword heuristics. Existing `INTERROGATIVE_PATTERNS` is a starting point. |
| **F10: Communication overhead measurement** | Figure 5(c). Comm steps as fraction of total action budget. Paper reports ~20%. | LOW | `n_message_actions / n_total_actions` from trajectory data. |
| **F11: Communication error taxonomy classifier** | Figure 6. Classify errors into categories: Repetition, Unresponsiveness, Hallucination (paper's three main categories). Sub-codes: C1a (unanswered, no reply), C1b (unanswered, ignored), C2 (non-answer/vague), C3b (incorrect claim corrected), C4a (spammy same info), C4b (near-duplicate blocks). | HIGH | Existing classifiers cover repetition + unresponsiveness. Need to add/adapt for vagueness (C2) and hallucination/incorrect-claim (C3b). Existing LLM judges for "unverifiable_claims" partially cover C3b. |
| **F12: Error frequency bar chart generation** | Figure 6 visual output. Horizontal bar chart with error category on y-axis, frequency % on x-axis. | LOW | Existing `generate_figure()` in `report/generator.py` already does this for the 10 failure modes. Needs mapping from old taxonomy to paper's C1a-C4b codes. |
| **F13: Solo experiment runs** | Figure 4 requires both solo and coop data points. Must run full pipeline in `mode: solo`. | MEDIUM | Infrastructure exists (`evaluate_solo`, `_run_solo`). Need actual experiment execution with Command A. |
| **F14: Coop experiment runs (comm on)** | Figure 4/5 coop data. Must run pipeline in `mode: coop, messaging_enabled: true`. | MEDIUM | Infrastructure exists. Need actual experiment execution. |
| **F15: Coop experiment runs (comm off)** | Figure 5(a,b) requires no-comm baseline. `mode: coop, messaging_enabled: false`. | MEDIUM | Config toggle exists. Need third experiment run. |
| **F16: Results JSONL ingestion** | Analysis pipeline must load `results.jsonl` from experiment runs. Each line has: repo, task_id, features, both_passed, merge_status, rounds, messages, tokens, wall_time, error. | LOW | Runner already writes this format. Need a loader for the analysis side. |
| **F17: Matplotlib figure generation** | All three figures need publication-quality matplotlib output. PDF/PNG at 300+ DPI. | LOW | Already using matplotlib in `report/generator.py`. Extend for Figs 4 and 5. |

### Differentiators (Competitive Advantage)

Features that add value beyond raw reproduction. Not required for correctness, but valuable for a compelling reproduction.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **D1: Plan:Question ratio analysis** | Paper's qualitative finding: successful trajectories have 2.04 plan:question ratio vs 1.31 for conflicted. Reproducing this with Command A shows whether the finding generalizes. | MEDIUM | Requires F9 speech act classifier. Compute ratio per trajectory, correlate with `merge_status`. |
| **D2: First-turn planning analysis** | Paper shows first-turn plan "nearly halves conflict rate (29.4% vs 51.5%)". Strong predictor. | LOW | Check if first message from either agent is classified as "plan" (from F9). Binary predicate per trajectory. |
| **D3: Specificity metrics** | Paper: successful trajectories have 32.6 line-number mentions vs 22.5, 13.1 file-path mentions vs 10.0. | LOW | Regex count of `/path/to/file` patterns and `line \d+`/`L\d+` patterns in message corpus. |
| **D4: Paper baseline overlay** | Show paper's published numbers alongside Command A results on same axes. Direct visual comparison. | LOW | Hardcode paper's reported values (per-model retention, conflict rates) as reference lines/bars. Some already in `PAPER_BASELINE_RATES` dict. |
| **D5: Per-repository breakdown** | Paper discusses cross-repo variance. Showing Command A performance per repo (12 repos) reveals model-specific strengths. | LOW | Group results by `repo` field. Already partially done in `ExperimentResults.summary()`. |
| **D6: Spatial vs semantic coordination analysis** | Paper's key insight: comm helps spatial coordination (fewer overlapping edits) but fails at semantic coordination (compatible logic). | HIGH | Requires patch-level analysis: measure file-overlap reduction (spatial) vs test-pass improvement (semantic) when comm is on. Uses existing `PatchInfo.files_modified`. |
| **D7: Confidence interval visualization** | Shaded bands on Figure 4 curves showing Wilson CIs. Publication-quality presentation. | LOW | `matplotlib fill_between` with CI bounds from F3. |
| **D8: Statistical significance testing** | Chi-squared or Fisher exact test on comm-vs-no-comm success rates (Fig 5a) and conflict rates (Fig 5b). | LOW | `scipy.stats.chi2_contingency` or `fisher_exact`. Reports p-values alongside visual comparison. |
| **D9: Automated figure validation** | Programmatic checks: curve monotonicity in Fig 4 (harder tasks = lower success), CI containment, retention in [0,1]. | LOW | Assert-based sanity checks in the pipeline. Catches data bugs before manual review. |
| **D10: Multi-format export** | Generate figures as PDF (LaTeX), PNG (slides), and SVG (web). Also export underlying data as CSV for external analysis. | LOW | Matplotlib `savefig` with multiple formats. CSV via pandas or manual write. |
| **D11: Transcript excerpt extraction** | Supporting qualitative examples for each error category. The existing report generator shows 3 evidence lines per detection -- extend for paper-appendix-style excerpts. | MEDIUM | Already partially implemented in `generate_text_report`. Need richer selection: best/worst examples per category. |

### Anti-Features (Deliberately NOT Building)

Features that seem good but create problems or scope creep.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **X1: Multi-model comparison** | Paper compares 5 models (GPT-5, Claude, MiniMax, Qwen3-Coder, Qwen3). | We only have Command A access. Building multi-model infra is pure YAGNI. Overlay paper's published numbers instead. | Use D4 (paper baseline overlay) for cross-model context. |
| **X2: Interactive dashboard** | Streamlit viewer already exists in repo. Tempting to add figure exploration. | Analysis pipeline should produce static artifacts (figures + data). Dashboard is a separate concern, adds maintenance burden, and isn't needed for paper reproduction. | Generate static HTML report with embedded PNGs. |
| **X3: Real-time experiment monitoring** | Watch experiments as they run, live-updating figures. | Experiments run for hours. Analysis is post-hoc. Real-time adds WebSocket/polling complexity for zero benefit. | Run experiments, then run analysis on completed results. |
| **X4: Custom Wilson CI implementation** | Avoid dependency on statsmodels. | Statsmodels is already available (in scientific Python stack). Reimplementing is a correctness risk. | Use `statsmodels.stats.proportion.proportion_confint(method='wilson')`. |
| **X5: LLM-based speech act classifier** | Use Cohere API for plan/question/update classification instead of heuristics. | Expensive (API calls per message), slow, non-deterministic. The paper itself uses heuristic patterns. Overkill for ~3 categories. | Regex/keyword heuristics matching paper's approach. |
| **X6: Automated paper LaTeX generation** | Generate LaTeX source for a reproduction report. | Scope creep. Figures + CSV data are sufficient. LaTeX authoring is a human task. | Export figures as PDF, data as CSV. Authors write LaTeX manually. |
| **X7: SWE-bench difficulty score integration** | Use SWE-bench difficulty ratings instead of computing from solo runs. | CooperBench is not SWE-bench. Difficulty must come from our actual solo experiment results (F1). External scores are methodologically wrong. | Compute difficulty from own solo run data per the paper's Algorithm 1. |
| **X8: Full upstream CooperBench fork** | Fork and modify the upstream evaluation harness. | We already have our own pipeline (`cooperbench-eval/`, `src/evaluation/cooperbench/`). Forking upstream adds merge conflict burden. Analysis pipeline is a new module, not a fork. | Build analysis as new module consuming existing results data. |

## Feature Dependencies

```
[F13: Solo runs]
    |
    v
[F1: Difficulty scores] ----requires----> [F2: Bucketing]
                                               |
                                               v
                                          [F4: Per-bucket rates]
                                               |
                                               +--requires--> [F3: Wilson CIs]
                                               |
                                               v
                                          [F5: AUC] --requires--> [F6: Retention]
                                               |
                                               v
                                          [F17: Figure 4 plot] --enhances--> [D4: Paper overlay]
                                                                            [D7: CI visualization]

[F14: Coop runs (comm on)]
[F15: Coop runs (comm off)]
    |         |
    v         v
[F7: Comm vs no-comm comparison] --requires--> [F17: Figure 5a plot]
[F8: Merge conflict rates]       --requires--> [F17: Figure 5b plot]
[F9: Speech act classifier]      --requires--> [F10: Comm overhead]
                                               [F17: Figure 5c plot]
                                               [D1: Plan:Question ratio]
                                               [D2: First-turn planning]

[F11: Error taxonomy classifier] --requires--> [F12: Figure 6 bar chart]
    ^                                              |
    |                                              v
    +--- partially exists (repetition,         [D11: Transcript excerpts]
         unresponsiveness classifiers)

[F16: Results JSONL ingestion] --required-by--> ALL analysis features

[D3: Specificity metrics] --independent-- (only needs raw transcript text)
[D5: Per-repo breakdown]  --independent-- (only needs results.jsonl)
[D8: Statistical tests]   --requires----> [F7, F8 computed rates]
[D9: Figure validation]   --requires----> [F17 generated figures]
```

### Dependency Notes

- **F1 requires F13:** Difficulty scores are computed FROM solo run results. Cannot bucket tasks without solo data.
- **F7 requires F14+F15:** Communication effect analysis needs both comm-on and comm-off experiment data.
- **F11 partially exists:** Existing classifiers (repetition, unresponsiveness, work_overlap) cover C4a/C4b and C1a/C1b. Need new classifiers for C2 (vagueness) and C3b (incorrect claims). The existing `unverifiable_claims` LLM judge partially covers C3b.
- **F16 is a foundation:** Every analysis feature reads from results.jsonl. This is the pipeline's input boundary.
- **D1 requires F9:** Plan:Question ratio needs speech acts classified first.
- **D6 conflicts with timeline:** Spatial vs semantic analysis is HIGH complexity and can be deferred without impacting figure correctness.

## MVP Definition

### Launch With (v1)

Minimum to produce valid Figures 4, 5, 6 with Command A data.

- [ ] **F16: Results JSONL ingestion** -- pipeline entry point, everything reads from here
- [ ] **F1+F2: Difficulty scores + bucketing** -- Figure 4 x-axis
- [ ] **F3: Wilson CIs** -- all error bars
- [ ] **F4+F5+F6: Per-bucket rates + AUC + Retention** -- Figure 4 content
- [ ] **F8: Merge conflict rates** -- Figure 5b (simplest panel)
- [ ] **F9: Speech act classifier** -- Figure 5c (plan/question/update)
- [ ] **F10: Comm overhead** -- Figure 5c budget fraction
- [ ] **F11: Error taxonomy mapping** -- Figure 6 (extend existing classifiers to paper's C-codes)
- [ ] **F12+F17: Figure generation (all three)** -- visual output

### Add After Validation (v1.x)

Features to add once core figures are generating correctly.

- [ ] **D4: Paper baseline overlay** -- makes figures immediately compelling for comparison
- [ ] **D7: CI visualization** -- shaded bands on Figure 4
- [ ] **D1: Plan:Question ratio** -- key qualitative finding to reproduce
- [ ] **D2: First-turn planning analysis** -- simple binary test of paper's strongest predictor
- [ ] **D3: Specificity metrics** -- regex-based, low effort, high insight
- [ ] **D8: Statistical significance** -- p-values for Fig 5 differences

### Future Consideration (v2+)

Defer until initial reproduction is validated.

- [ ] **D5: Per-repository breakdown** -- useful but not in paper's main figures
- [ ] **D6: Spatial vs semantic coordination** -- HIGH complexity, paper discusses qualitatively
- [ ] **D10: Multi-format export** -- PDF/SVG/CSV, nice to have
- [ ] **D11: Transcript excerpts** -- for appendix-style supporting evidence
- [ ] **D9: Automated figure validation** -- quality gate, not user-facing

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| F16: Results ingestion | HIGH | LOW | P1 |
| F1: Difficulty scores | HIGH | LOW | P1 |
| F2: Bucketing | HIGH | LOW | P1 |
| F3: Wilson CIs | HIGH | LOW | P1 |
| F4: Per-bucket rates | HIGH | LOW | P1 |
| F5: AUC | HIGH | LOW | P1 |
| F6: Retention | HIGH | LOW | P1 |
| F17: Figure 4 plot | HIGH | MEDIUM | P1 |
| F7: Comm vs no-comm | HIGH | LOW | P1 |
| F8: Merge conflict rate | HIGH | LOW | P1 |
| F9: Speech act classifier | HIGH | MEDIUM | P1 |
| F10: Comm overhead | HIGH | LOW | P1 |
| F17: Figure 5 plots | HIGH | MEDIUM | P1 |
| F11: Error taxonomy | HIGH | HIGH | P1 |
| F12: Figure 6 chart | HIGH | LOW | P1 |
| D4: Paper overlay | MEDIUM | LOW | P2 |
| D7: CI visualization | MEDIUM | LOW | P2 |
| D1: Plan:Question ratio | MEDIUM | MEDIUM | P2 |
| D2: First-turn planning | MEDIUM | LOW | P2 |
| D3: Specificity metrics | MEDIUM | LOW | P2 |
| D8: Significance tests | MEDIUM | LOW | P2 |
| D5: Per-repo breakdown | LOW | LOW | P3 |
| D6: Spatial/semantic | MEDIUM | HIGH | P3 |
| D10: Multi-format | LOW | LOW | P3 |
| D11: Transcript excerpts | LOW | MEDIUM | P3 |
| D9: Figure validation | LOW | LOW | P3 |

**Priority key:**
- P1: Must have for valid reproduction (table stakes)
- P2: Should have, add after core figures validate (differentiators)
- P3: Nice to have, defer (low value or high cost)

## Existing Codebase Assets

What already exists and can be reused vs what needs to be built.

| Asset | Location | Reuse Potential | Gap |
|-------|----------|-----------------|-----|
| Repetition classifier | `cooperbench-eval/src/classifiers/repetition.py` | Direct reuse for C4a/C4b | Need to map to paper's C-codes |
| Unresponsiveness classifier | `cooperbench-eval/src/classifiers/unresponsiveness.py` | Direct reuse for C1a/C1b | Need to split "no reply" vs "ignored" |
| Unverifiable claims (LLM) | `cooperbench-eval/src/llm_judge/unverifiable_claims.py` | Partial reuse for C3b | Different framing: "incorrect claim" vs "unverifiable" |
| Question detection | `cooperbench-eval/src/data_loading/schemas.py` | `Message.is_question` for F9 | Need plan + update classifiers too |
| Report generator | `cooperbench-eval/src/report/generator.py` | Figure 6 bar chart template | Needs new chart types for Figs 4, 5 |
| Paper baseline rates | `cooperbench-eval/src/report/generator.py` | `PAPER_BASELINE_RATES` dict | Maps failure modes, not Fig 4/5 data |
| Results JSONL writer | `src/evaluation/cooperbench/runner.py` | Output format for F16 | Need corresponding reader in analysis module |
| CooperBench config | `configs/cooperbench_default.yaml` | Solo/coop/comm toggles | Already supports all needed experiment modes |
| Evaluator | `src/evaluation/cooperbench/evaluator.py` | Merge status tracking | `merge_status` field directly feeds F8 |
| Pipeline result model | `src/evaluation/cooperbench/models.py` | `messages_exchanged`, `rounds_completed` | Feeds F10 overhead calculation |

## Sources

- CooperBench paper (arxiv:2601.13295v2) -- full methodology details from HTML version
- Existing codebase: `cooperbench-eval/src/` (classifiers, report generator, data schemas)
- Existing codebase: `src/evaluation/cooperbench/` (pipeline, evaluator, models, runner)
- Upstream CooperBench repo: `repos/CooperBench/` (dataset structure, gold conflict checker)
- Wilson score interval: statsmodels documentation (HIGH confidence -- well-established statistical method)

---
*Feature research for: CooperBench paper figure reproduction pipeline*
*Researched: 2026-02-14*
