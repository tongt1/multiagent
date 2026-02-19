# Project State

## Project Reference

See: /home/terry_tong_cohere_com/cooperbench-repro/.planning/PROJECT.md (updated 2026-02-14)

**Core value:** Produce verifiable figures (4, 5, 6) that replicate the CooperBench paper's key findings -- the solo-coop coordination gap, communication's failure to improve cooperation despite reducing merge conflicts, and the breakdown of communication errors -- using Command A instead of the paper's external models.

**Current focus:** PROJECT COMPLETE -- All 5 phases done. 12/12 plans executed.

## Current Position

Phase: 5 (Qualitative Transcript Analysis)
Plan: 1 of 1 in current phase (05-01 complete -- phase done)
Status: PROJECT COMPLETE
Last activity: 2026-02-19 -- Completed 05-01-PLAN.md (Qualitative transcript analysis)

Progress: [##########] 100% (Phase 1: 3/3 + Phase 2: 3/3 + Phase 3: 3/3 + Phase 4: 2/2 + Phase 5: 1/1 = 12 plans complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 12
- Average duration: 38 min
- Total execution time: 7 hours 36 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 3/3 | 300 min | 100 min |
| 2 | 3/3 | 218 min | 73 min |
| 3 | 3/3 | 12 min | 4 min |
| 4 | 2/2 | 4 min | 2 min |
| 5 | 1/1 | 2 min | 2 min |

**Recent Trend:**
- 01-01: 4 min (2 tasks, 2 files)
- 01-02: 6 min (2 tasks, 1 file)
- 01-03: 290 min (2 tasks, 300 result files)
- 02-01: 65 min (2 tasks, 1 code file + 303 data files)
- 02-02: 150 min (2 tasks, 200 data files)
- 02-03: 3 min (2 tasks, 2 files)
- 03-01: 3 min (2 tasks, 2 files)
- 03-02: 3 min (2 tasks, 2 files)
- 03-03: 6 min (2 tasks, 2 files)
- 04-01: 2 min (2 tasks, 4 files)
- 04-02: 2 min (2 tasks, 6 files)
- 05-01: 2 min (2 tasks, 2 files)

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
- infra_error defaults to false (field absent from all result.json files)
- No merge_union outcomes observed: all coop merges are merge_clean (naive) or merge_failed
- LimitsExceeded override applied to 4 records (all already both_passed=false)
- Only 3 of 10 difficulty buckets populated (buckets 3, 6, 9) due to single-model low pass rates
- Solo rates use seed=0 only for Figure 4 per-bucket comparisons (97 valid records after eval error exclusion)
- Wilson CI returns (0.0, 1.0) for n=0 edge case (complete uncertainty)
- AUC integrates over populated buckets only (3 points), not all 10 -- documented as rough approximation
- Merge conflict rates use ALL coop records (including eval_error) because merge outcomes are valid even when eval subsequently fails
- Speech act percentages use largest-remainder rounding to guarantee sum == 100.0%
- Speech acts collected from ALL coop-comm records (428 messages) regardless of eval_error
- NEW LLM classifier for Figure 6 taxonomy (not cooperbench-eval) -- message-quality errors, not coordination failures
- Command A via staging endpoint for taxonomy classification (same key as benchmark runs)
- Temperature 0.0 for deterministic classification results
- Paper baselines centralized in single PAPER_BASELINES dict for all 3 figures
- CI bands rendered as narrow fill_between rectangles at discrete bucket centers (not continuous) -- honest sparse data representation
- AUC/retention comparison shown as text annotation box, not reference lines on y-axis -- AUC is area under curve, not a point value
- Figure 6 categories ordered by paper grouping (C4a,C4b,C1a,C1b,C2,C3b) for visual clustering of bracket groups
- Grouping brackets drawn below x-axis with colored lines and centered labels for paper category mapping
- Paper baselines shown as dashed reference lines (Fig 5 panel b) and annotation box (Fig 6) -- different visualization for different data types
- Reuse exact classify_speech_act from analyze_fig5.py for Phase 5 consistency (do not improve the classifier)
- Store infinite Plan:Question ratios as None (JSON null), run Mann-Whitney U on finite ratios only
- Report counter-intuitive findings accurately: conflict trajectories have HIGHER planning rates and P:Q ratios -- do not imply causation
- Include small-sample caveat for n=11 no-plan-first group in Fisher's exact test results
- Note line mention sparsity (1/428 messages) rather than omitting the metric

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

### Unified Data Store (Phase 2 Output)

| Setting | Records | Pass Rate |
|---------|---------|-----------|
| Solo (seed 0) | 100 | 1.0% |
| Solo (seed 1) | 100 | 1.0% |
| Solo (seed 2) | 100 | 1.0% |
| Coop-Comm | 100 | 0.0% |
| Coop-NoComm | 100 | 0.0% |
| **Total** | **500** | |

Data store: `data/results.json` (500 records, 24 fields each)
Collection script: `scripts/collect_results.py` (rerunnable)
Difficulty: {0.3333: 1 pair, 0.6667: 1 pair, 1.0: 98 pairs}
Populated buckets: 3/10 (buckets 3, 6, 9)

### Figure 4 Analysis (Phase 3 Output)

| Setting | AUC | Retention |
|---------|-----|-----------|
| Solo | 0.15 | -- |
| Coop-Comm | 0.0 | 0.0 |
| Coop-NoComm | 0.0 | 0.0 |

Key: Bucket 3 solo 1/1 passed (100%), all other bucket/setting combos 0%. Solo AUC driven entirely by bucket 3.
Data files: `scripts/analyze_fig4.py`, `data/fig4_metrics.json`

### Figure 5 Analysis (Phase 3 Output)

| Metric | Coop-Comm | Coop-NoComm | Difference |
|--------|-----------|-------------|------------|
| Success Rate | 0/96 = 0.0% | 0/88 = 0.0% | -- |
| Merge Conflict Rate | 41/100 = 41.0% | 55/100 = 55.0% | Comm reduces by 14% |

Speech acts (428 messages): plan 46.7%, question 26.0%, update 10.5%, other 16.8%
Communication overhead: mean 22.8%, median 21.1%, range 2.9%--53.8%
Data files: `scripts/analyze_fig5.py`, `data/fig5_metrics.json`

### Figure 6 Analysis (Phase 3 Output)

| Category | Count | % of Errors | % of Transcripts |
|----------|-------|-------------|------------------|
| C1a (Unanswered - No Reply) | 9 | 11.7% | 8.0% |
| C1b (Unanswered - Ignored) | 7 | 9.1% | 5.0% |
| C2 (Non-answer/Vague) | 1 | 1.3% | 1.0% |
| C3b (Incorrect Claim) | 3 | 3.9% | 3.0% |
| C4a (Spammy - Same Info) | 32 | 41.6% | 25.0% |
| C4b (Spammy - Near-duplicate) | 25 | 32.5% | 20.0% |

Total: 77 errors across 100 transcripts, 51 with errors, 0 API failures
Dominant: Spammy categories (C4a+C4b) = 74.0% of all errors
Data files: `scripts/analyze_fig6.py`, `data/fig6_metrics.json`

### Figure 4 Generation (Phase 4 Output)

Publication-quality Figure 4 generated as PDF (TrueType fonts) and PNG (300 DPI):
- 3 populated bucket centers (0.35, 0.65, 0.95) with scatter+line markers
- CI shaded bands via fill_between at each data point
- Unpopulated bucket regions shaded light gray
- Paper baseline comparison: our solo AUC=0.150 vs paper 0.338, coop AUC=0.000 vs 0.200
- Output files: `figures/fig4_difficulty_curves.{pdf,png}`
- Script: `scripts/generate_fig4.py`, shared module: `scripts/paper_baselines.py`

### Figure 5 Generation (Phase 4 Output)

Publication-quality Figure 5 generated as PDF (TrueType fonts) and PNG (300 DPI):
- 3-panel layout: (a) success rates with Wilson CI error bars, (b) merge conflict rates, (c) speech acts
- Panel (a): 0% in both settings, CI upper bounds annotated (3.8% comm, 4.2% nocomm)
- Panel (b): 41% comm vs 55% nocomm with paper planning baselines (29.4% with, 51.5% without)
- Panel (c): plan 46.7%, question 26.0%, update 10.5%, other 16.8%; overhead 22.8% (paper ~20%)
- Output files: `figures/fig5_communication.{pdf,png}`
- Script: `scripts/generate_fig5.py`

### Figure 6 Generation (Phase 4 Output)

Publication-quality Figure 6 generated as PDF (TrueType fonts) and PNG (300 DPI):
- 6-category bar chart with count+pct annotations (C4a=41.6%, C4b=32.5%, C1a=11.7%, C1b=9.1%, C2=1.3%, C3b=3.9%)
- Colored grouping brackets mapping to paper's 3 categories: Repetition 74.0%, Unresponsiveness 20.8%, Hallucination 5.2%
- Annotation box with aggregated paper category mapping
- Output files: `figures/fig6_error_taxonomy.{pdf,png}`
- Script: `scripts/generate_fig6.py`

### Qualitative Transcript Analysis (Phase 5 Output)

| Metric | No Conflict (n=59) | Conflict (n=41) | p-value | Direction |
|--------|-------------------|-----------------|---------|-----------|
| P:Q ratio (mean, finite) | 1.15 | 1.96 | 0.0016 | Higher ratio -> more conflicts |
| First-turn planning rate | 83.1% | 97.6% | 0.0249 | Plan-first -> more conflicts |
| File mentions/trajectory | 3.61 | 3.61 | 0.9173 | No difference |
| Line mentions/trajectory | 0.017 | 0.000 | sparse | Insufficient data |

Key insight: Conflict trajectories show MORE planning behavior (counter-intuitive). Likely reflects that tasks where agents plan to modify overlapping files generate both more plan messages AND more conflicts.
Data files: `scripts/analyze_qualitative.py`, `data/qualitative_metrics.json`

## Session Continuity

Last session: 2026-02-19
Stopped at: Completed 05-01-PLAN.md (Qualitative transcript analysis -- Phase 5 complete -- PROJECT COMPLETE)
Resume file: None
Next action: None -- all 12 plans across 5 phases complete

---
*State initialized: 2026-02-14*
*Last updated: 2026-02-19 (05-01 complete, Phase 5 done, PROJECT COMPLETE)*
