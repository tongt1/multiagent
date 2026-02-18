# Phase 2: Results Collection and Data Foundation - Context

**Gathered:** 2026-02-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Normalize raw benchmark logs into a unified data store with merge/test outcomes and difficulty scores, ready for downstream analysis (Figures 4, 5, 6). This includes running `cooperbench eval` to produce eval results, collecting all data into a single JSON store, and computing per-pair difficulty scores with bucket assignments.

</domain>

<decisions>
## Implementation Decisions

### Evaluation execution
- Check eval backend (eval/backends/docker.py) for entrypoint compatibility BEFORE running — don't waste time on 300 failed evals
- Concurrency: 10 (cooperbench eval default — eval is lighter than agent runs, no LLM calls)
- Run all 3 settings (solo, coop-comm, coop-nocomm) in parallel — 30 concurrent Docker containers
- Error handling: retry once on failure, then record as error. Errors excluded from metrics, count reported separately

### Success & merge definitions
- Merge outcome mapping (by final outcome):
  - status=clean + strategy=naive → `merge_clean`
  - status=clean + strategy=union → `merge_union` (conflicts existed but resolved)
  - status=conflicts or status=error → `merge_failed`
- Solo mode: merge_outcome = `merge_clean` (implicit — no merge step, no conflicts by definition)
- Success metric: `both_passed` only (binary, matches paper). No partial credit tracking.
- LimitsExceeded pairs: count as failure (both_passed=false). Include in denominator — do not exclude.

### Difficulty score granularity
- Unit: per feature-pair (each of the 100 pairs gets its own difficulty score)
- Formula: d(pair) = 1 - Solo_pass_rate(pair), averaged across multiple solo seeds
- Run 2 additional solo seeds (3 total) to get continuous difficulty values {0, 0.33, 0.67, 1.0}
  - Additional cost: ~$92 (2 × $46)
  - This populates more difficulty buckets for meaningful Figure 4 curves
- 10 equal-width buckets over [0, 1] as specified in roadmap
- Note as known limitation: single-model reproduction yields coarser difficulty than paper's multi-model approach

### Data store structure
- Format: single JSON file at `data/results.json` (project root)
- Flat array — one record per feature-pair per setting
- Each record includes: repo, task_id, features, setting, run metadata (cost, steps, duration), eval results (both_passed, merge_outcome), difficulty score, bucket assignment
- Conversation messages embedded directly in coop-comm records (messages array)
- Difficulty scores and bucket assignments stored in-record (not separate file)

### Claude's Discretion
- Exact JSON schema field names and nesting
- How to handle edge cases in eval (e.g., empty patches, missing files)
- Script structure for the collection pipeline
- Whether to use pandas internally or pure JSON processing

</decisions>

<specifics>
## Specific Ideas

- The additional solo seeds should use the same 100 feature pairs to get per-pair variance
- The unified store should be self-contained: any downstream script should only need `data/results.json` plus the paper baseline numbers
- Update Experiment 12 (`~/multiagent/experiments/12_cooperbench_repro.md`) with eval results once available

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-results-collection-and-data-foundation*
*Context gathered: 2026-02-18*
