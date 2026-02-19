# Phase 5: Qualitative Transcript Analysis - Research

**Researched:** 2026-02-19
**Domain:** Qualitative communication analysis and statistical correlation of agent transcripts
**Confidence:** HIGH

## Summary

Phase 5 computes qualitative metrics from the 100 coop-comm transcripts in `data/results.json` and correlates them with merge conflict outcomes. Four metrics are required: Plan:Question ratio per trajectory, first-turn planning detection, specificity metrics (line number and file path mentions), and a summary comparison table. All metrics build on the speech act classifier already implemented in `scripts/analyze_fig5.py` (Phase 3) and the existing data structures.

The data reveals important characteristics that shape implementation decisions. The existing speech act classifier has a known limitation: it misses contractions like "I'm modifying" (only matching "I am modifying"), causing 27/428 messages (6.3%) to be classified as "other" when they are functionally plan messages. Additionally, the priority ordering (question > update > plan) means that 66/111 messages classified as "question" also contain plan language (e.g., "I will be modifying X. Please ensure your changes don't conflict."). For consistency with Phase 3's published fig5_metrics.json, the Phase 5 analysis MUST reuse the exact same classifier from `scripts/analyze_fig5.py` rather than improving it. This ensures Plan:Question ratios are computed on the same basis as the speech act percentages already reported.

The data also shows counter-intuitive correlations: trajectories with merge conflicts actually have HIGHER first-turn planning rates (97.6% vs 83.1%) and higher Plan:Question ratios (mean 1.96 vs 1.15). Line number mentions are nearly absent (1/428 messages). These findings are genuine and should be reported accurately -- they reflect that planning alone does not prevent conflicts when both agents announce they will modify the same file.

**Primary recommendation:** Implement as a single Python script `scripts/analyze_qualitative.py` following the established pattern (read results.json, compute metrics, write `data/qualitative_metrics.json`, print summary table). Reuse the speech act classifier from `scripts/analyze_fig5.py` via import. Use Fisher's exact test (scipy.stats) for first-turn planning correlation and Mann-Whitney U for continuous metrics, since scipy is available on system Python.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| QUAL-01 | Compute Plan:Question ratio per trajectory and correlate with merge conflict outcomes | Reuse `classify_speech_act()` from `analyze_fig5.py`. Per-trajectory: count plan and question classifications, compute ratio. Handle 0-question trajectories (34/100 have zero questions). Correlation via Mann-Whitney U test on finite ratios grouped by merge outcome. Data shows: clean mean=1.15, failed mean=1.96 (finite ratios only). |
| QUAL-02 | Detect first-turn planning (Plan message in first turn) and measure conflict rate reduction | "First turn" = first message in the trajectory (index 0). Apply `classify_speech_act()` to first message. 89/100 first messages classify as plan. Use Fisher's exact test on 2x2 contingency table (plan-first x merge-outcome). Data shows: plan-first conflict rate 44.9% vs no-plan-first 9.1% (n=11, small sample). |
| QUAL-03 | Count specificity metrics per trajectory (line number mentions, file path mentions) | File paths: regex `(?:[\w.-]+/)+[\w.-]+\.\w{1,5}` plus standalone filenames (`\b[\w.-]+\.(?:py|js|go|...)\b`). Line numbers: `\bline\s+\d+\b` etc. Data shows: 374/428 messages mention file paths (87.4%), only 1/428 mentions line numbers. Per-trajectory: mean 3.6 file mentions for both groups. Line mentions effectively zero. |
| QUAL-04 | Generate summary table comparing qualitative metrics for conflict vs no-conflict trajectories | Table with rows for each metric (Plan:Question ratio, first-turn planning rate, file mentions/trajectory, line mentions/trajectory) and columns for no-conflict (n=59) and conflict (n=41) groups, plus p-values from statistical tests. Output both as JSON (for programmatic use) and formatted console output. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python stdlib json | 3.12 | Read results.json, write qualitative_metrics.json | Consistent with all Phase 2-4 scripts |
| Python stdlib re | 3.12 | Specificity pattern matching (file paths, line numbers) | Same regex approach as speech act classifier |
| Python stdlib statistics | 3.12 | Mean, median computations | Already used in analyze_fig5.py |
| Python stdlib math | 3.12 | Wilson CI for rate comparisons | Consistent with Phase 3 |
| scipy.stats | 1.17.0 | Fisher's exact test, Mann-Whitney U, point-biserial correlation | Available on system Python; provides rigorous p-values for correlation claims |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pathlib | stdlib | File path handling | Script I/O |
| argparse | stdlib | Script CLI (--input, --output) | Consistent with analyze_fig4.py/fig5.py pattern |
| collections.Counter | stdlib | Category counting | Frequency tallies |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| scipy.stats.fisher_exact | Manual chi-squared | Fisher's exact is correct for small samples (n=11 in no-plan-first group); chi-squared requires n>=5 per cell. scipy is available, use it. |
| scipy.stats.mannwhitneyu | Manual rank-sum | Mann-Whitney is the standard non-parametric test for comparing two groups; hand-rolling is error-prone. scipy is available, use it. |
| Import classifier from analyze_fig5 | Copy-paste classifier | Import ensures consistency if patterns are ever updated. The module is importable (no `if __name__` guard issues). |
| Single script | Multiple scripts (one per QUAL-ID) | Single script is cleaner: all metrics share the same data loading, same per-trajectory loop, same output structure. Requirements are tightly coupled. |

**Installation:**
```bash
# No additional installation needed
# System Python 3.12 has scipy 1.17.0, numpy 2.4.2
# Scripts use #!/usr/bin/env python3
```

## Architecture Patterns

### Recommended Project Structure
```
scripts/
    analyze_fig5.py          # Existing: speech act classifier (imported by Phase 5)
    analyze_qualitative.py   # NEW: QUAL-01 through QUAL-04
data/
    results.json             # Input (500 records, 100 coop-comm with transcripts)
    qualitative_metrics.json # NEW Output: per-trajectory metrics + summary table
```

### Pattern 1: Import and Reuse Speech Act Classifier
**What:** Import `classify_speech_act` from `analyze_fig5.py` rather than duplicating the regex patterns.
**When to use:** QUAL-01, QUAL-02.
**Example:**
```python
import sys
from pathlib import Path

# Add scripts dir to path for import
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_fig5 import classify_speech_act
```
**Why:** Ensures Plan:Question ratios use the identical classifier that produced the fig5_metrics.json speech act distribution. Any discrepancy between Phase 3 and Phase 5 classifications would undermine the analysis.

### Pattern 2: Per-Trajectory Metric Computation
**What:** Single pass through coop-comm records computing all four qualitative metrics per trajectory.
**When to use:** All QUAL requirements.
**Example:**
```python
def compute_trajectory_metrics(record: dict) -> dict:
    """Compute all qualitative metrics for a single coop-comm trajectory.

    Returns dict with plan_count, question_count, plan_question_ratio,
    first_turn_is_plan, file_mentions, line_mentions, etc.
    """
    messages = record.get("messages", [])

    # Speech act counts
    acts = Counter(classify_speech_act(msg["message"]) for msg in messages)
    plans = acts.get("plan", 0)
    questions = acts.get("question", 0)

    # Plan:Question ratio
    if questions > 0:
        pq_ratio = plans / questions
    elif plans > 0:
        pq_ratio = None  # Infinite (plans but no questions)
    else:
        pq_ratio = 0.0

    # First-turn planning
    first_turn_is_plan = False
    if messages:
        first_turn_is_plan = classify_speech_act(messages[0]["message"]) == "plan"

    # Specificity
    file_mentions = sum(1 for msg in messages if FILE_PATTERN.search(msg["message"]))
    line_mentions = sum(1 for msg in messages if LINE_PATTERN.search(msg["message"]))

    return {
        "task_id": record["task_id"],
        "repo": record["repo"],
        "merge_outcome": record["merge_outcome"],
        "messages_count": len(messages),
        "plan_count": plans,
        "question_count": questions,
        "update_count": acts.get("update", 0),
        "other_count": acts.get("other", 0),
        "plan_question_ratio": pq_ratio,
        "first_turn_is_plan": first_turn_is_plan,
        "file_mentions": file_mentions,
        "line_mentions": line_mentions,
    }
```

### Pattern 3: Statistical Correlation Testing
**What:** Use appropriate statistical tests for each metric type.
**When to use:** QUAL-01 (continuous metric vs binary outcome), QUAL-02 (2x2 contingency table).
**Example:**
```python
from scipy.stats import fisher_exact, mannwhitneyu

# QUAL-01: Plan:Question ratio (continuous) vs merge outcome (binary)
# Use Mann-Whitney U test (non-parametric, handles non-normal distributions)
clean_ratios = [t["plan_question_ratio"] for t in trajectories
                if t["merge_outcome"] == "merge_clean" and t["plan_question_ratio"] is not None]
failed_ratios = [t["plan_question_ratio"] for t in trajectories
                 if t["merge_outcome"] == "merge_failed" and t["plan_question_ratio"] is not None]

if clean_ratios and failed_ratios:
    stat, p_value = mannwhitneyu(clean_ratios, failed_ratios, alternative='two-sided')

# QUAL-02: First-turn planning (binary) vs merge outcome (binary)
# Use Fisher's exact test (correct for small expected cell counts)
# Contingency table: [[plan_clean, plan_failed], [noplan_clean, noplan_failed]]
table = [[plan_clean, plan_failed], [noplan_clean, noplan_failed]]
odds_ratio, p_value = fisher_exact(table)
```

### Pattern 4: Summary Table Output
**What:** Structured comparison table as both JSON and console output.
**When to use:** QUAL-04.
**Example output structure:**
```json
{
    "summary_table": {
        "plan_question_ratio": {
            "no_conflict": {"mean": 1.15, "median": 1.00, "n_finite": 40, "n_infinite": 19},
            "conflict": {"mean": 1.96, "median": 2.00, "n_finite": 29, "n_infinite": 12},
            "test": "Mann-Whitney U",
            "p_value": 0.045,
            "direction": "higher ratio correlates with more conflicts"
        },
        "first_turn_planning": {
            "no_conflict": {"rate": 0.831, "count": 49, "total": 59},
            "conflict": {"rate": 0.976, "count": 40, "total": 41},
            "test": "Fisher's exact",
            "p_value": 0.031,
            "odds_ratio": 7.35,
            "direction": "first-turn planning correlates with more conflicts"
        },
        "file_mentions_per_trajectory": {
            "no_conflict": {"mean": 3.6, "median": 3.0},
            "conflict": {"mean": 3.6, "median": 3.0},
            "test": "Mann-Whitney U",
            "p_value": 0.89,
            "direction": "no significant difference"
        },
        "line_mentions_per_trajectory": {
            "no_conflict": {"mean": 0.02, "median": 0.0},
            "conflict": {"mean": 0.0, "median": 0.0},
            "test": "not applicable",
            "note": "insufficient data (1/428 messages mentions line numbers)"
        }
    }
}
```

### Anti-Patterns to Avoid
- **Improving the speech act classifier for Phase 5:** Tempting to add "I'm modifying" to PLAN_PATTERNS, but this would create inconsistency with Phase 3's published fig5_metrics.json. The Plan:Question ratio must be computed on the same classification basis. Document the limitation instead.
- **Treating infinite Plan:Question ratios as numeric values:** 34/100 trajectories have zero questions. Do not substitute infinity with a large number or exclude these. Report them separately (n_finite, n_infinite) and run statistical tests on finite ratios only.
- **Claiming causal direction from correlations:** The data shows higher planning correlates with MORE conflicts. This does not mean planning causes conflicts. More likely: tasks where agents plan to modify the same file are both more likely to generate plan messages AND more likely to conflict. Report correlations with direction but do not imply causation.
- **Using chi-squared test for small samples:** The no-plan-first group has only n=11. Fisher's exact test is correct; chi-squared approximation breaks down when expected cell counts are below 5.
- **Creating separate output files per requirement:** All four QUAL requirements are tightly coupled (same data, same per-trajectory loop, same grouping). A single `qualitative_metrics.json` output file is cleaner.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Speech act classification | New regex patterns | Import from `analyze_fig5.py` | Consistency with Phase 3 results |
| Fisher's exact test | Manual combinatorics | `scipy.stats.fisher_exact` | Exact computation for 2x2 tables, handles small samples correctly |
| Mann-Whitney U test | Manual rank-sum computation | `scipy.stats.mannwhitneyu` | Well-tested implementation, handles ties correctly |
| File path regex | Simple substring matching | Compiled regex with common extensions | Need to handle paths like `src/jinja2/loaders.py` and standalone files like `__init__.py` |

**Key insight:** The "hard part" of Phase 5 is not computation (it is straightforward) but interpretation. The counter-intuitive correlations (more planning = more conflicts) require careful framing. The analysis script should compute and report; interpretation belongs in the summary table notes and any downstream write-up.

## Common Pitfalls

### Pitfall 1: Classifier Inconsistency Between Phases
**What goes wrong:** Creating a new or improved speech act classifier for Phase 5 that produces different counts than Phase 3's fig5_metrics.json.
**Why it happens:** The existing classifier has known limitations (missing "I'm modifying", "I'm planning" contractions; "please ensure" classified as question when semantically it is a plan-with-coordination). Natural instinct is to fix these.
**How to avoid:** Import and reuse `classify_speech_act` from `scripts/analyze_fig5.py` exactly as-is. Document the known limitations in the output metadata. If classifier improvements are desired, they should be done as a separate Phase 3 update that also re-generates fig5_metrics.json.
**Warning signs:** Plan counts in Phase 5 not matching 200 plans from fig5_metrics.json.

### Pitfall 2: Infinite Plan:Question Ratio Handling
**What goes wrong:** Using `float('inf')` in statistical tests or mean calculations, causing NaN/errors.
**Why it happens:** 34 trajectories have zero questions. Of these, 32 have plans (infinite ratio) and 2 have neither (0.0 ratio).
**How to avoid:** Store `plan_question_ratio` as `None` when questions=0 and plans>0 (conceptually infinite). Report n_finite and n_infinite separately. Run Mann-Whitney U test on finite ratios only. Report the proportion of infinite-ratio trajectories in each group as a separate metric.
**Warning signs:** NaN in summary table, Mann-Whitney U test failing with infinity values.

### Pitfall 3: Confounding in First-Turn Planning Analysis
**What goes wrong:** Reporting that first-turn planning INCREASES conflict rate (97.6% vs 9.1%) without context.
**Why it happens:** 89/100 trajectories have plan-first, so the non-plan-first group is only n=11. The 11 non-plan-first messages are classified differently because: (a) "I'm modifying" doesn't match "I am modifying" in PLAN_PATTERNS (classifier limitation), or (b) "please let me know" triggers question classification before plan check. Many of these 11 ARE functionally planning messages.
**How to avoid:** Report the finding with the small sample caveat. Include the Fisher's exact test p-value (which accounts for sample size). Note that the classifier limitation contributes to the small no-plan group size. Consider also reporting: "at least one agent plans first" vs "neither agent plans first" as an alternative grouping.
**Warning signs:** Reporting a strong causal claim based on n=11 comparison group.

### Pitfall 4: File Path Regex Over-Matching
**What goes wrong:** Matching non-file-path text as file mentions (e.g., "Please avoid changes" matching as having a file extension).
**Why it happens:** Overly broad regex that matches any word.word pattern.
**How to avoid:** Use two complementary patterns: (1) path-like patterns requiring at least one `/` separator: `(?:[\w.-]+/)+[\w.-]+\.\w{1,5}`, and (2) standalone filenames with known code extensions: `\b[\w.-]+\.(?:py|js|ts|go|rs|java|...)\b`. Count messages (not individual matches) to avoid inflating counts from multiple file mentions in one message.
**Warning signs:** File mention counts exceeding message counts, or dramatically different from 374/428 (87.4%).

### Pitfall 5: Ignoring Single-Agent Trajectories
**What goes wrong:** Assuming all trajectories have messages from both agents. 5/100 trajectories have messages from only one agent.
**Why it happens:** Not checking for agent participation diversity.
**How to avoid:** Handle single-agent trajectories in first-turn planning analysis. "Both agents plan first" is only meaningful for the 95 two-agent trajectories. Document: "5 trajectories have messages from only 1 agent."
**Warning signs:** Agent count assumptions breaking, division by zero when computing per-agent metrics.

### Pitfall 6: Misinterpreting Line Number Sparsity
**What goes wrong:** Reporting line number mentions as a useful differentiating metric when only 1/428 messages contains one.
**Why it happens:** The requirement says "count line number mentions" so it seems like a meaningful metric.
**How to avoid:** Compute and report the metric as required, but note: "Line number mentions are too sparse (1/428 messages, 0.2%) to serve as a differentiating metric between conflict and no-conflict groups." Include the metric in the summary table with a note rather than omitting it.
**Warning signs:** Reporting a 0.0 vs 0.02 difference as meaningful without the sparsity context.

## Code Examples

### Reusing the Phase 3 Speech Act Classifier
```python
#!/usr/bin/env python3
"""Compute qualitative transcript analysis metrics.

Reads: data/results.json
Writes: data/qualitative_metrics.json

Requirements: QUAL-01, QUAL-02, QUAL-03, QUAL-04

Usage:
    python scripts/analyze_qualitative.py
    python scripts/analyze_qualitative.py --input data/results.json --output data/qualitative_metrics.json
"""

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

# Import speech act classifier from Phase 3
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from analyze_fig5 import classify_speech_act

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "results.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "qualitative_metrics.json"
```
Source: Follows the exact pattern used by `scripts/analyze_fig4.py`, `scripts/analyze_fig5.py`, and `scripts/analyze_fig6.py`.

### Specificity Regex Patterns
```python
# File path detection: path/to/file.ext format
FILE_PATH_PATTERN = re.compile(r'(?:[\w.-]+/)+[\w.-]+\.\w{1,5}')

# Standalone filename with known code extensions
STANDALONE_FILE_PATTERN = re.compile(
    r'\b[\w.-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|cpp|c|h|rb|yml|yaml|json|toml|cfg|txt|md|html|css|scss)\b'
)

# Line number patterns
LINE_NUMBER_PATTERNS = [
    re.compile(r'\bline\s+\d+', re.IGNORECASE),
    re.compile(r'\blines?\s+\d+\s*[-\u2013to]+\s*\d+', re.IGNORECASE),
    re.compile(r'\bL\d+\b'),
    re.compile(r'\bat\s+line\b', re.IGNORECASE),
]


def count_file_mentions(text: str) -> int:
    """Count file path/name mentions in a message.

    Returns 1 if any file pattern matches, 0 otherwise.
    Counts per-message (not per-match) to avoid inflation.
    """
    if FILE_PATH_PATTERN.search(text) or STANDALONE_FILE_PATTERN.search(text):
        return 1
    return 0


def count_line_mentions(text: str) -> int:
    """Count line number mentions in a message.

    Returns 1 if any line number pattern matches, 0 otherwise.
    """
    return 1 if any(p.search(text) for p in LINE_NUMBER_PATTERNS) else 0
```
Source: Patterns verified against actual message data. FILE_PATH_PATTERN matches 374/428 messages (87.4%). LINE_NUMBER_PATTERNS match 1/428 messages.

### Fisher's Exact Test for First-Turn Planning
```python
from scipy.stats import fisher_exact, mannwhitneyu


def compute_first_turn_correlation(trajectories: list[dict]) -> dict:
    """QUAL-02: Correlate first-turn planning with merge outcomes.

    Uses Fisher's exact test on 2x2 contingency table.
    """
    plan_clean = sum(1 for t in trajectories
                     if t["first_turn_is_plan"] and t["merge_outcome"] == "merge_clean")
    plan_failed = sum(1 for t in trajectories
                      if t["first_turn_is_plan"] and t["merge_outcome"] == "merge_failed")
    noplan_clean = sum(1 for t in trajectories
                       if not t["first_turn_is_plan"] and t["merge_outcome"] == "merge_clean")
    noplan_failed = sum(1 for t in trajectories
                        if not t["first_turn_is_plan"] and t["merge_outcome"] == "merge_failed")

    table = [[plan_clean, plan_failed], [noplan_clean, noplan_failed]]
    odds_ratio, p_value = fisher_exact(table)

    total_plan = plan_clean + plan_failed
    total_noplan = noplan_clean + noplan_failed

    return {
        "contingency_table": {
            "plan_first_clean": plan_clean,
            "plan_first_conflict": plan_failed,
            "no_plan_first_clean": noplan_clean,
            "no_plan_first_conflict": noplan_failed,
        },
        "plan_first_conflict_rate": round(plan_failed / total_plan, 4) if total_plan > 0 else None,
        "no_plan_first_conflict_rate": round(noplan_failed / total_noplan, 4) if total_noplan > 0 else None,
        "fishers_exact_odds_ratio": round(odds_ratio, 4),
        "fishers_exact_p_value": round(p_value, 4),
        "n_plan_first": total_plan,
        "n_no_plan_first": total_noplan,
        "note": (
            "Small no-plan-first sample (n=11). Some messages classified as non-plan "
            "contain planning language with contractions (e.g., 'I'm modifying') not "
            "captured by the Phase 3 classifier."
        ),
    }
```

### Console Summary Table
```python
def print_summary_table(summary: dict) -> None:
    """Print human-readable comparison table."""
    print("\n" + "=" * 80)
    print("QUALITATIVE METRICS: CONFLICT vs NO-CONFLICT TRAJECTORIES")
    print("=" * 80)

    pq = summary["plan_question_ratio"]
    print("\n--- Plan:Question Ratio (QUAL-01) ---")
    print(f"  No conflict (finite, n={pq['no_conflict']['n_finite']}):")
    print(f"    Mean: {pq['no_conflict']['mean']:.2f}, Median: {pq['no_conflict']['median']:.2f}")
    print(f"    Infinite ratio (plans, no questions): {pq['no_conflict']['n_infinite']}")
    print(f"  Conflict (finite, n={pq['conflict']['n_finite']}):")
    print(f"    Mean: {pq['conflict']['mean']:.2f}, Median: {pq['conflict']['median']:.2f}")
    print(f"    Infinite ratio (plans, no questions): {pq['conflict']['n_infinite']}")
    print(f"  Test: {pq['test']}, p={pq['p_value']:.4f}")
    print(f"  Direction: {pq['direction']}")

    ft = summary["first_turn_planning"]
    print(f"\n--- First-Turn Planning (QUAL-02) ---")
    print(f"  Plan-first conflict rate: {ft['plan_first_conflict_rate']*100:.1f}% (n={ft['n_plan_first']})")
    print(f"  No-plan-first conflict rate: {ft['no_plan_first_conflict_rate']*100:.1f}% (n={ft['n_no_plan_first']})")
    print(f"  Fisher's exact: OR={ft['fishers_exact_odds_ratio']:.2f}, p={ft['fishers_exact_p_value']:.4f}")

    # ... file and line mentions ...

    print("\n" + "=" * 80)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Chi-squared for small contingency tables | Fisher's exact test | Always (but often misused) | Fisher's exact is the correct test when expected cell counts < 5; chi-squared approximation fails |
| Pearson correlation for binary outcomes | Point-biserial / Mann-Whitney U | Standard practice | Point-biserial is Pearson for one binary variable; Mann-Whitney U is non-parametric alternative for non-normal data |
| Simple p-values without effect sizes | Report both p-value and effect size | Emerging standard | p-value alone says significance, not magnitude; odds ratio and mean differences provide practical significance |

**Deprecated/outdated:**
- Using chi-squared test for 2x2 tables with small expected counts -- Fisher's exact test is strictly superior.
- Reporting only "statistically significant" without noting sample size limitations.

## Open Questions

1. **Should an alternative "first-turn planning" definition be computed?**
   - What we know: The primary definition (first message index 0 is Plan) yields n=89 plan-first vs n=11 no-plan-first. The n=11 group is small. An alternative: "both agents' first messages are Plan" yields 53 vs 47 (better balanced) but shows weaker effect (43.4% vs 38.3% conflict rate).
   - What's unclear: Which definition better captures the research question. The paper's intent may be about whether planning *happens early*, not whether the very first message is a plan.
   - Recommendation: Compute BOTH definitions and report both. Primary: first-message-is-plan (matches the requirement's literal wording). Secondary: both-agents-plan-first (better statistical power). Let the planner decide if both should be in the summary table or if one is relegated to metadata.

2. **Should the summary table include additional context metrics?**
   - What we know: The four QUAL metrics are sparse in discriminative power. File mentions show no difference between groups. Line mentions are nearly absent. Plan:Question ratio and first-turn planning show counter-intuitive direction.
   - What's unclear: Whether adding context metrics (e.g., total message count, mean message length, proportion of messages mentioning the same file as partner) would enrich the table.
   - Recommendation: Compute the four required metrics. Add message count per trajectory as a context row in the table (it is already available). Do NOT add unrequested metrics without planner approval -- scope creep is a risk.

3. **How to handle "unique file mentions" vs "messages with file mentions"?**
   - What we know: A trajectory might mention `__init__.py` in 3 messages. We can count: (a) messages that mention any file, (b) total file mentions across all messages, (c) unique files mentioned.
   - What's unclear: Which granularity the requirement intends.
   - Recommendation: Count per-message (did this message mention a file? yes/no) and sum per trajectory. This gives "number of messages with file path mentions" which is the most natural reading of "count specificity metrics per trajectory." Also track unique files mentioned as a secondary metric in the per-trajectory data.

## Sources

### Primary (HIGH confidence)
- `data/results.json` -- Actual data analyzed: 100 coop-comm records, 428 messages, message structure verified (from/to/message/timestamp/feature_id fields)
- `scripts/analyze_fig5.py` -- Phase 3 speech act classifier: PLAN_PATTERNS, QUESTION_PATTERNS, UPDATE_PATTERNS, classify_speech_act() function, priority ordering (question > update > plan > other)
- `data/fig5_metrics.json` -- Phase 3 output: 200 plan, 111 question, 45 update, 72 other (total 428)
- Empirical data analysis performed during research: Plan:Question ratios, first-turn planning rates, specificity counts, all computed directly from results.json

### Secondary (MEDIUM confidence)
- `scipy.stats.fisher_exact` -- Standard implementation of Fisher's exact test for 2x2 contingency tables; scipy 1.17.0 verified available on system Python
- `scipy.stats.mannwhitneyu` -- Standard implementation of Mann-Whitney U test; verified available
- Wilson score interval formula -- Same implementation used in Phase 3, verified correct

### Tertiary (LOW confidence)
- Interpretation of "first turn" in QUAL-02 -- requirement says "Plan message in first turn" which we interpret as message index 0; could also mean "first message from each agent" or "first exchange round"
- Appropriate significance threshold -- using standard alpha=0.05 but with n=100 and multiple comparisons, may need Bonferroni correction (alpha=0.0125 for 4 tests)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- All tools are stdlib Python + scipy (available on system Python). No new dependencies.
- Architecture: HIGH -- Single-script pattern proven in Phase 3. Data structures well understood from empirical analysis.
- Speech act reuse: HIGH -- Import from analyze_fig5.py verified feasible. Classifier behavior confirmed against fig5_metrics.json.
- Statistical tests: HIGH -- Fisher's exact and Mann-Whitney U are standard, scipy implementation verified available.
- Specificity regex: HIGH -- Patterns tested against all 428 messages, match counts verified.
- Pitfalls: HIGH -- All identified from actual data analysis, especially counter-intuitive correlation directions and classifier limitations.
- Interpretation: MEDIUM -- Counter-intuitive findings require careful framing. The correlation between planning and conflicts is genuine but likely confounded by task characteristics.

**Research date:** 2026-02-19
**Valid until:** 2026-03-19 (stable -- data is fixed, no external dependencies changing)
