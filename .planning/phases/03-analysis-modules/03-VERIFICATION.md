---
phase: 03-analysis-modules
verified: 2026-02-18T23:30:00Z
status: passed
score: 7/7 must-haves verified
re_verification: false
---

# Phase 3: Analysis Modules Verification Report

**Phase Goal:** All statistical metrics required for Figures 4, 5, and 6 are computed from the normalized data store, with each figure's analysis independently testable.
**Verified:** 2026-02-18T23:30:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

Must-haves are aggregated from the `must_haves.truths` in all three PLAN frontmatters. Seven top-level truths are derived from the four Success Criteria in ROADMAP.md.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Per-bucket solo and coop success rates with Wilson 95% CIs are computed for all populated difficulty buckets, and AUC + retention metrics are available as named outputs | VERIFIED | `data/fig4_metrics.json`: 3 buckets (3,6,9), Wilson CIs for every rate (bounds valid: 0 <= lower <= rate <= upper <= 1), AUC solo=0.15 via trapezoidal integration over 3 populated bucket centers, retention coop_comm=0.0 and coop_nocomm=0.0. Solo uses seed=0 only (97 records, no inflation). |
| 2 | Comm vs no-comm success rates and merge conflict rates are computed with differences reported | VERIFIED | `data/fig5_metrics.json`: success rates coop_comm=0.0, coop_nocomm=0.0 with Wilson CIs. Merge conflict rates comm=41%, nocomm=55% with CIs. Difference: "comm reduces conflicts by 14.0%". |
| 3 | Agent messages from coop-with-comm transcripts are classified into speech act types and communication overhead is expressed as percentage of total action budget | VERIFIED | `data/fig5_metrics.json`: 428 messages classified (plan=46.7%, question=26.0%, update=10.5%, other=16.8%, sum=100.0%). No double-counting (sum of category counts == total_messages). Overhead mean=22.8%, median=21.1%, range 2.9%-53.8%, n_tasks=100 with per-task breakdown. |
| 4 | An LLM-based classifier labels communication errors in all coop-with-comm transcripts using the paper's C1a-C4b taxonomy, with per-category frequency counts available | VERIFIED | `data/fig6_metrics.json`: 100 transcripts classified, 77 errors found across 51 transcripts, 0 API failures. All 6 categories present (C1a=9, C1b=7, C2=1, C3b=3, C4a=32, C4b=25). Evidence strings populated for all 77 errors. No cooperbench-eval taxonomy leakage. |
| 5 | Each figure's analysis is independently testable (separate scripts, separate outputs) | VERIFIED | Three independent scripts: `scripts/analyze_fig4.py` (320 lines), `scripts/analyze_fig5.py` (429 lines), `scripts/analyze_fig6.py` (552 lines). Each reads `data/results.json` and writes its own output JSON. No cross-dependencies between scripts. |
| 6 | Solo rates use seed=0 only, empty buckets excluded, eval_error records excluded | VERIFIED | fig4_metrics.json: metadata.solo_seed=0, solo_seed0_records=97 (all <= 100), populated_buckets=[3,6,9] (3 of 10). records_used=473 of 500 total (27 excluded). |
| 7 | Speech act percentages sum to exactly 100% and each message gets exactly one classification | VERIFIED | fig5_metrics.json: plan 46.7 + question 26.0 + update 10.5 + other 16.8 = 100.0%. Category count sum (200+111+45+72=428) == total_messages (428). Largest-remainder rounding ensures exact 100.0%. |

**Score:** 7/7 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/analyze_fig4.py` | Figure 4 analysis script (min 120 lines) | VERIFIED | 320 lines. Wilson CI, trapezoidal AUC, per-bucket rates, CLI with --input/--output. Stdlib only. |
| `data/fig4_metrics.json` | Structured JSON for Phase 4 figure generation | VERIFIED | Valid JSON with buckets (3), auc (3 settings), retention (2 settings), metadata. |
| `scripts/analyze_fig5.py` | Figure 5 analysis script (min 150 lines) | VERIFIED | 429 lines. Success rates, merge conflicts, speech act regex classifier, overhead computation. Stdlib only. |
| `data/fig5_metrics.json` | Structured JSON for Phase 4 figure generation | VERIFIED | Valid JSON with success_rates, merge_conflict_rates, speech_acts, overhead, metadata. |
| `scripts/analyze_fig6.py` | LLM communication error classifier (min 180 lines) | VERIFIED | 552 lines. Cohere API via httpx, taxonomy prompt, JSON parser with 3-stage fallback, --dry-run/--limit/--resume CLI. |
| `data/fig6_metrics.json` | Per-transcript classifications and frequency counts | VERIFIED | Valid JSON with 100 classifications, 6-category frequency table, summary statistics, metadata. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `scripts/analyze_fig4.py` | `data/results.json` | `json.load` reads unified results store | WIRED | Line 88: `json.load(f)` in `load_and_filter()`. DEFAULT_INPUT points to `data/results.json`. |
| `scripts/analyze_fig4.py` | `data/fig4_metrics.json` | `json.dump` writes metrics output | WIRED | Line 287: `json.dump(output, f, indent=2)`. DEFAULT_OUTPUT points to `data/fig4_metrics.json`. |
| `scripts/analyze_fig5.py` | `data/results.json` | `json.load` reads unified results store | WIRED | Line 386: `json.load(f)` in `main()`. DEFAULT_INPUT points to `data/results.json`. |
| `scripts/analyze_fig5.py` | `data/fig5_metrics.json` | `json.dump` writes metrics output | WIRED | Line 420: `json.dump(metrics, f, indent=2)`. DEFAULT_OUTPUT points to `data/fig5_metrics.json`. |
| `scripts/analyze_fig6.py` | `data/results.json` | `json.load` reads unified results store | WIRED | Line 381: `json.load(f)` in `main()`. DEFAULT_INPUT points to `data/results.json`. |
| `scripts/analyze_fig6.py` | Cohere API | `httpx.Client.post` for LLM classification | WIRED | Line 210: `client.post(COHERE_API_URL, json=payload, headers=headers)`. API URL configurable via env. |
| `scripts/analyze_fig6.py` | `data/fig6_metrics.json` | `json.dump` writes classification results | WIRED | Line 521: `json.dump(output, f, indent=2)`. DEFAULT_OUTPUT points to `data/fig6_metrics.json`. |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| FIG4-03 | 03-01 | Compute per-bucket solo and coop success rates | SATISFIED | fig4_metrics.json buckets array with solo/coop_comm/coop_nocomm rates for 3 buckets |
| FIG4-04 | 03-01 | Compute Wilson 95% CIs for all rates | SATISFIED | Every rate entry has ci_lower and ci_upper fields, bounds validated (0 <= lower <= rate <= upper <= 1) |
| FIG4-05 | 03-01 | Compute AUC via trapezoidal integration | SATISFIED | fig4_metrics.json auc object with solo=0.15, coop_comm=0.0, coop_nocomm=0.0, n_points and x_range included |
| FIG4-06 | 03-01 | Compute retention metric (AUC_coop / AUC_solo) | SATISFIED | fig4_metrics.json retention object: coop_comm=0.0, coop_nocomm=0.0 (AUC_solo > 0, so division valid) |
| FIG5-01 | 03-02 | Compute comm vs no-comm success rates | SATISFIED | fig5_metrics.json success_rates with Wilson CIs for both settings |
| FIG5-02 | 03-02 | Compute merge conflict rates with/without communication | SATISFIED | fig5_metrics.json merge_conflict_rates: comm=41%, nocomm=55%, difference reported |
| FIG5-03 | 03-02 | Classify agent messages into speech act types | SATISFIED | fig5_metrics.json speech_acts: 428 messages classified into plan/question/update/other via regex heuristics |
| FIG5-04 | 03-02 | Compute communication overhead as % of action budget | SATISFIED | fig5_metrics.json overhead: mean=22.8%, with per-task breakdown for 100 tasks |
| FIG6-01 | 03-03 | Implement LLM-based communication error classifier | SATISFIED | scripts/analyze_fig6.py: 552-line script using Cohere Command A with C1a-C4b taxonomy prompt |
| FIG6-02 | 03-03 | Run classifier on all coop-with-comm transcripts | SATISFIED | fig6_metrics.json: 100 transcripts classified, 0 API failures, 77 errors found with evidence |

**Coverage:** 10/10 requirements satisfied. No orphaned requirements.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | - |

No TODO, FIXME, PLACEHOLDER, or stub patterns found in any of the three analysis scripts. No empty implementations detected.

### Human Verification Required

### 1. Script Rerun Idempotency

**Test:** Run `python scripts/analyze_fig4.py && python scripts/analyze_fig5.py` and verify JSON outputs are identical to existing files.
**Expected:** Output files should be byte-identical (deterministic computation).
**Why human:** Requires running the scripts in the correct Python environment.

### 2. Figure 6 Classifier Reproducibility

**Test:** Run `python scripts/analyze_fig6.py --limit 3` and compare results against existing classifications for those 3 task IDs.
**Expected:** Results should be similar (temperature=0.0 provides near-deterministic output, but LLM outputs can vary slightly).
**Why human:** Requires active Cohere API key and network access to staging endpoint.

### 3. Speech Act Classification Quality

**Test:** Manually review 5 random messages classified as "plan" and 5 classified as "question" in the results.
**Expected:** Classifications should match human judgment for the regex heuristic approach.
**Why human:** Requires qualitative assessment of regex pattern accuracy.

### Gaps Summary

No gaps found. All seven observable truths verified, all six artifacts pass three-level verification (exists, substantive, wired), all seven key links confirmed, all ten requirements satisfied, and no anti-patterns detected. The three analysis modules are independent, each reading from `data/results.json` and writing structured JSON output ready for Phase 4 figure generation.

### Commit Verification

All three referenced commits exist in the repository:
- `0d6ba14` -- feat(03-01): implement Figure 4 analysis script
- `8f5f1e8` -- feat(03-02): implement Figure 5 analysis script with speech act classifier
- `9ccd46f` -- feat(03-03): implement LLM communication error classifier and run on all 100 transcripts

---

_Verified: 2026-02-18T23:30:00Z_
_Verifier: Claude (gsd-verifier)_
