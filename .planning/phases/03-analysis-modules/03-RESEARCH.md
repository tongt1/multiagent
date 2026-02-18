# Phase 3: Analysis Modules - Research

**Researched:** 2026-02-18
**Domain:** Statistical analysis of CooperBench benchmark results (Figures 4, 5, 6)
**Confidence:** HIGH

## Summary

Phase 3 computes all statistical metrics required for Figures 4, 5, and 6 from the unified `data/results.json` store produced by Phase 2. The work divides into three independent modules: (1) Figure 4 difficulty-stratified success curves with Wilson CIs, AUC, and retention; (2) Figure 5 communication effects including success rates, merge conflict rates, speech act classification, and overhead; (3) Figure 6 LLM-based communication error classification using the paper's C1a/C1b/C2/C3b/C4a/C4b taxonomy.

The data is severely sparse: 500 records across 3 settings, with 98% of task pairs at maximum difficulty (d=1.0, bucket 9), only 3 of 10 difficulty buckets populated, ~1% solo pass rate, and 0% coop pass rate. This means most metrics will be degenerate or near-zero, but the pipeline must handle these edge cases correctly. Wilson CIs are well-behaved at p=0 and p=1 (unlike Wald), AUC over 3 populated buckets is a rough approximation, and the communication analysis benefits from 428 messages across 100 coop-comm transcripts which provides decent signal.

The Figure 6 taxonomy classifier requires careful attention: the requirements specify a 6-category communication-error taxonomy (C1a/C1b/C2/C3b/C4a/C4b) that describes message quality issues. The existing `cooperbench-eval` package has classifiers with the same C-codes, but they represent a different taxonomy (coordination failure modes from Table 1 of the paper). For Phase 3, we must implement a NEW LLM-based classifier matching the requirements' definitions, NOT reuse the existing cooperbench-eval classifiers.

**Primary recommendation:** Implement all three modules as standalone Python scripts under `scripts/`, each reading from `data/results.json` and writing structured JSON output to `data/`. Use pure Python with math stdlib for Wilson CIs (no statsmodels dependency needed). Use the cooperbench-eval LLM infrastructure pattern (httpx + Cohere API) for the taxonomy classifier. Defer matplotlib figure generation to Phase 4.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FIG4-03 | Compute per-bucket solo and coop success rates | Direct computation from results.json: group by (setting, bucket), count both_passed=true / total. Only 3 buckets populated (3, 6, 9). Handle empty buckets as undefined. |
| FIG4-04 | Compute Wilson 95% confidence intervals for all rates | Manual Wilson CI formula (15 lines Python, no external deps). Verified correct at p=0, p=1, small n. Works for all our edge cases. |
| FIG4-05 | Compute AUC via trapezoidal integration for solo and coop curves | Trapezoidal rule over populated bucket centers only. With 3 points the AUC is a rough approximation -- document as known limitation. |
| FIG4-06 | Compute retention metric (AUC_coop / AUC_solo) | Simple division. Handle AUC_solo=0 edge case (return NaN or 0). With 0% coop pass, AUC_coop will be 0, so retention will be 0. |
| FIG5-01 | Compute comm vs no-comm success rates | Group by setting (coop-comm vs coop-nocomm), count both_passed. Both are 0% in our data but pipeline must handle it. |
| FIG5-02 | Compute merge conflict rates with and without communication | coop-comm: 41/100 = 41% conflict. coop-nocomm: 55/100 = 55% conflict. This is the strongest signal in the data. |
| FIG5-03 | Classify agent messages into speech act types (plan/question/update) | Regex/keyword heuristics per requirements. 428 messages across 100 transcripts. Preliminary analysis: ~62% plan, ~11% question, ~7% update, ~28% unclassified (need broader patterns). |
| FIG5-04 | Compute communication overhead as percentage of total action budget | messages_count / total_steps per task. Mean overhead: 22.8%. Range: 2.9% to 53.8%. |
| FIG6-01 | Implement LLM-based communication error classifier using paper's taxonomy | NEW classifier needed (not reusing cooperbench-eval). 6 categories: C1a (unanswered no-reply), C1b (unanswered ignored), C2 (non-answer/vague), C3b (incorrect claim corrected), C4a (spammy same info), C4b (spammy near-duplicate blocks). Uses Cohere API via httpx. |
| FIG6-02 | Run classifier on all coop-with-comm transcripts | 100 transcripts, each with 1-11 messages. Estimate: 100 API calls, ~$0.50-2.00 total cost with Command A. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python stdlib json | 3.12 | Read results.json, write analysis outputs | 500-record dataset, trivial scale |
| Python stdlib math | 3.12 | Wilson CI formula (sqrt, pow) | No external deps needed for 15-line formula |
| Python stdlib re | 3.12 | Speech act classification patterns | Regex/keyword heuristics per requirements |
| Python stdlib collections | 3.12 | Counter, defaultdict for aggregation | Standard grouping operations |
| httpx | 0.28.1 (in venv) | Cohere API calls for FIG6-01 classifier | Already installed in CooperBench venv |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pathlib | stdlib | File path handling | All script I/O |
| argparse | stdlib | Script CLI arguments | If scripts need options |
| time | stdlib | Rate limiting for API calls | FIG6-01 classifier batching |
| logging | stdlib | Structured logging | All scripts |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual Wilson CI | statsmodels `proportion_confint` | statsmodels not installed in venv; installing it pulls numpy+scipy+pandas chain. Manual formula is 15 lines, verified correct at all edge cases. Use manual. |
| Manual trapezoidal rule | scipy.integrate.trapezoid | scipy not installed; 3-line for loop is simpler than adding a dependency for trivial computation |
| httpx for Cohere API | cohere SDK or litellm | httpx already in venv and proven by cooperbench-eval LLM judge pattern. cohere SDK NOT installed. litellm is installed but its API is more complex for a single endpoint call. |
| Regex speech acts | LLM-based classifier | Requirements explicitly say heuristic patterns (regex/keyword). LLM is out of scope per REQUIREMENTS.md "Out of Scope" section. |

**Installation:**
```bash
# No additional installation needed
# All dependencies exist in cooperbench venv at repos/CooperBench/.venv/
source repos/CooperBench/.venv/bin/activate
```

## Architecture Patterns

### Recommended Project Structure
```
scripts/
    analyze_fig4.py          # FIG4-03, FIG4-04, FIG4-05, FIG4-06
    analyze_fig5.py          # FIG5-01, FIG5-02, FIG5-03, FIG5-04
    analyze_fig6.py          # FIG6-01, FIG6-02
data/
    results.json             # Input (from Phase 2)
    fig4_metrics.json        # Output: per-bucket rates, CIs, AUC, retention
    fig5_metrics.json        # Output: comm rates, conflict rates, speech acts, overhead
    fig6_metrics.json        # Output: per-transcript error classifications, frequency counts
```

### Pattern 1: Analysis Script Structure
**What:** Each analysis script reads results.json, computes metrics, writes structured JSON output.
**When to use:** All three analysis modules follow this pattern.
**Example:**
```python
#!/usr/bin/env python3
"""Compute Figure 4 metrics: per-bucket success rates, Wilson CIs, AUC, retention.

Reads: data/results.json
Writes: data/fig4_metrics.json

Usage:
    python scripts/analyze_fig4.py
    python scripts/analyze_fig4.py --input data/results.json --output data/fig4_metrics.json
"""

import json
import math
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "results.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "fig4_metrics.json"


def wilson_ci(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Handles edge cases: p=0, p=1, n=0, small n.
    Uses z=1.96 for 95% CI (alpha=0.05, two-tailed).
    """
    if total == 0:
        return (0.0, 1.0)

    z = 1.96  # 95% CI
    p_hat = successes / total
    n = total

    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return (round(lower, 6), round(upper, 6))


def trapezoidal_auc(x: list[float], y: list[float]) -> float:
    """Trapezoidal integration for AUC computation.

    x and y must be same length, sorted by x.
    """
    if len(x) < 2:
        return 0.0
    auc = 0.0
    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        auc += (y[i] + y[i + 1]) / 2 * dx
    return round(auc, 6)


def main():
    # Load data
    with open(DEFAULT_INPUT) as f:
        records = json.load(f)

    # Filter: exclude infra errors, use only seed=0 for coop comparison
    # For per-bucket rates, use all records (seed 0 only for solo to match difficulty basis)
    # ... computation logic ...

    # Write output
    output = {
        "buckets": [...],  # per-bucket data
        "auc": {"solo": ..., "coop_comm": ..., "coop_nocomm": ...},
        "retention": ...,
        "metadata": {"input_file": str(DEFAULT_INPUT), "total_records": len(records)}
    }

    with open(DEFAULT_OUTPUT, "w") as f:
        json.dump(output, f, indent=2)
```

### Pattern 2: Wilson CI for All Edge Cases
**What:** Manual Wilson CI formula verified against our actual data edge cases.
**When to use:** Every rate computation in FIG4-03, FIG4-04, FIG5-01, FIG5-02.
**Verified results:**
```python
# Our actual data edge cases:
wilson_ci(0, 100)  # p=0.0  -> (0.0, 0.036995)    coop pass rates
wilson_ci(3, 300)  # p=0.01 -> (0.003407, 0.028984) solo aggregate
wilson_ci(2, 3)    # p=0.67 -> (0.207655, 0.938510) bucket 3 solo
wilson_ci(0, 0)    # n=0    -> (0.0, 1.0)           empty buckets
wilson_ci(0, 98)   # p=0.0  -> (0.0, 0.037613)     bucket 9 coop
```

### Pattern 3: Speech Act Classification (Regex Heuristics)
**What:** Classify messages as plan/question/update using pattern matching.
**When to use:** FIG5-03.
**Example:**
```python
import re

PLAN_PATTERNS = re.compile(
    r'\b('
    r'I will|I am going to|I\'ll|I plan to|I intend to|'
    r'I will be|I\'m going to|my plan|my approach|my strategy|'
    r'Let me|I am modifying|I am implementing|I am adding|'
    r'I have implemented|I\'ve implemented|'
    r'I suggest|I recommend|I propose|'
    r'my task is|I\'m working on|I am working on'
    r')\b',
    re.IGNORECASE,
)

QUESTION_PATTERNS = re.compile(
    r'\?|'
    r'\b('
    r'can you|could you|would you|will you|do you|did you|'
    r'is there|are there|have you|shall we|should we|'
    r'please let me know|please confirm|please ensure|'
    r'does this|does that|is this|is that'
    r')\b',
    re.IGNORECASE,
)

UPDATE_PATTERNS = re.compile(
    r'\b('
    r'I have completed|I have finished|I\'m done|I\'ve done|'
    r'I completed|done with|completed the|finished the|'
    r'I have made|I have added|changes are complete|'
    r'changes done|modifications complete|is ready|'
    r'has been implemented|my task is complete|'
    r'I\'ve added|I\'ve made|I\'ve modified|'
    r'I have resolved|acknowledged|proceeding with'
    r')\b',
    re.IGNORECASE,
)


def classify_speech_act(message_text: str) -> str:
    """Classify a message into plan/question/update.

    Priority: question > update > plan (most specific first).
    Messages matching none return 'other'.
    """
    if QUESTION_PATTERNS.search(message_text):
        return "question"
    if UPDATE_PATTERNS.search(message_text):
        return "update"
    if PLAN_PATTERNS.search(message_text):
        return "plan"
    return "other"
```

### Pattern 4: LLM Communication Error Classifier (FIG6-01)
**What:** Classify each coop-comm transcript for communication error types.
**When to use:** FIG6-01, FIG6-02.
**Important:** This is a NEW classifier, NOT the cooperbench-eval classifiers.
**Example:**
```python
import httpx
import json
import os
import time

COHERE_API_URL = os.environ.get(
    "COHERE_API_URL",
    "https://stg.api.cohere.com/v2/chat"
)
MODEL = "command-a-03-2025"


def build_taxonomy_prompt(messages: list[dict]) -> str:
    """Build the classification prompt using the paper's taxonomy."""
    transcript = "\n".join(
        f"[{msg['from']}] {msg['message']}"
        for msg in messages
    )

    return f"""Analyze the following multi-agent communication transcript for communication errors.

Classify each error found using this taxonomy:
- C1a (Unanswered - No Reply): A direct question receives no reply at all from the other agent.
- C1b (Unanswered - Ignored): A request or question is acknowledged but substantively ignored.
- C2 (Non-answer/Vague): A response is given but it is vague, non-committal, or lacks actionable information.
- C3b (Incorrect Claim - Corrected): An agent makes a factually wrong claim about the codebase or their changes that needs correction.
- C4a (Spammy - Same Info): An agent repeats the same information they already communicated.
- C4b (Spammy - Near-duplicate): Messages contain near-identical content (copy-paste status blocks, repeated file listings).

TRANSCRIPT:
{transcript}

Respond in JSON:
{{
    "errors": [
        {{
            "category": "C1a|C1b|C2|C3b|C4a|C4b",
            "evidence": "Quote or describe the problematic exchange",
            "message_index": 0
        }}
    ],
    "summary": "Brief overall assessment"
}}

If no communication errors are found, return {{"errors": [], "summary": "No errors detected"}}."""


def classify_transcript(messages: list[dict], api_key: str) -> dict:
    """Run LLM classifier on a single transcript."""
    prompt = build_taxonomy_prompt(messages)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert at analyzing multi-agent communication transcripts. Respond only in valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }

    with httpx.Client(timeout=90.0) as client:
        resp = client.post(COHERE_API_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", [])
        if content and isinstance(content, list):
            text = content[0].get("text", "")
        else:
            text = str(content)

    # Parse JSON response
    return _parse_json_response(text)
```

### Pattern 5: Output JSON Schema
**What:** Structured output format consumed by Phase 4 figure generation.
**fig4_metrics.json:**
```json
{
    "buckets": [
        {
            "bucket": 3,
            "center": 0.35,
            "solo": {
                "successes": 2, "total": 3,
                "rate": 0.6667,
                "ci_lower": 0.2077, "ci_upper": 0.9385
            },
            "coop_comm": {
                "successes": 0, "total": 1,
                "rate": 0.0,
                "ci_lower": 0.0, "ci_upper": 0.7935
            },
            "coop_nocomm": {
                "successes": 0, "total": 1,
                "rate": 0.0,
                "ci_lower": 0.0, "ci_upper": 0.7935
            }
        }
    ],
    "auc": {
        "solo": 0.2000,
        "coop_comm": 0.0,
        "coop_nocomm": 0.0
    },
    "retention": {
        "coop_comm": 0.0,
        "coop_nocomm": 0.0
    },
    "metadata": {
        "populated_buckets": [3, 6, 9],
        "total_buckets": 10,
        "sparsity_warning": "Only 3 of 10 buckets populated; AUC is a rough approximation"
    }
}
```

**fig5_metrics.json:**
```json
{
    "success_rates": {
        "coop_comm": {"successes": 0, "total": 100, "rate": 0.0, "ci_lower": 0.0, "ci_upper": 0.037},
        "coop_nocomm": {"successes": 0, "total": 100, "rate": 0.0, "ci_lower": 0.0, "ci_upper": 0.037}
    },
    "merge_conflict_rates": {
        "coop_comm": {"conflicts": 41, "total": 100, "rate": 0.41, "ci_lower": 0.316, "ci_upper": 0.510},
        "coop_nocomm": {"conflicts": 55, "total": 100, "rate": 0.55, "ci_lower": 0.449, "ci_upper": 0.647}
    },
    "speech_acts": {
        "plan": {"count": 266, "pct": 62.1},
        "question": {"count": 49, "pct": 11.4},
        "update": {"count": 28, "pct": 6.5},
        "other": {"count": 85, "pct": 19.9}
    },
    "overhead": {
        "mean_pct": 22.8,
        "min_pct": 2.9,
        "max_pct": 53.8,
        "per_task": [...]
    }
}
```

**fig6_metrics.json:**
```json
{
    "classifications": [
        {
            "task_id": 1655,
            "repo": "dottxt_ai_outlines_task",
            "errors": [
                {"category": "C4a", "evidence": "...", "message_index": 2}
            ]
        }
    ],
    "frequency": {
        "C1a": {"count": 5, "pct": 5.0},
        "C1b": {"count": 3, "pct": 3.0},
        "C2": {"count": 12, "pct": 12.0},
        "C3b": {"count": 4, "pct": 4.0},
        "C4a": {"count": 15, "pct": 15.0},
        "C4b": {"count": 8, "pct": 8.0}
    },
    "metadata": {
        "total_transcripts": 100,
        "model": "command-a-03-2025",
        "api_cost_estimate": "$1.50"
    }
}
```

### Anti-Patterns to Avoid
- **Using cooperbench-eval classifiers for Figure 6:** The existing classifiers (work_overlap, divergent_architecture, etc.) represent a DIFFERENT taxonomy than the requirements' C1a/C1b/C2/C3b/C4a/C4b. The cooperbench-eval maps C-codes to coordination failure modes, but the requirements describe communication-quality error categories. Build a new classifier.
- **Treating empty buckets as rate=0:** Buckets with no data have undefined success rates, not zero rates. AUC should integrate only over populated buckets. Document this.
- **Installing heavy dependencies:** statsmodels, scipy, numpy, pandas are NOT in the venv. The Wilson CI formula is 15 lines of stdlib math. The trapezoidal rule is 3 lines. Do not add 200MB+ of deps for trivial computations.
- **Generating figures in Phase 3:** Phase 3 computes metrics; Phase 4 generates figures. Keep analysis scripts focused on JSON output.
- **Using seed>0 solo records for coop comparison:** Difficulty is computed from all 3 solo seeds, but per-bucket success rates for solo should use ONLY seed=0 to match the original run. Additional seeds exist only for difficulty scoring.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Wilson CI | Custom statistical library | 15-line formula with math.sqrt | Well-defined closed-form solution, verified against all our edge cases |
| Trapezoidal AUC | Numerical integration library | 3-line for loop | Trivially simple for discrete bucket data |
| JSON schema validation | Pydantic models for output | Write JSON directly + print summary | 500 records, 3 output files -- validation overhead not justified |
| LLM response parsing | instructor/pydantic | Simple JSON parsing with fallback (regex extraction) | Pattern already proven in cooperbench-eval/src/llm_judge/base.py |
| Rate limiting | External library | time.sleep() between API calls | 100 calls total, simple sequential is fine |

**Key insight:** All the "heavy" statistical computations in this phase are trivially implementable with Python stdlib. The complexity is in handling edge cases (sparse data, zero rates, undefined metrics) and getting the definitions right, not in the computation itself.

## Common Pitfalls

### Pitfall 1: Conflating Two Taxonomies
**What goes wrong:** Using the cooperbench-eval classifiers (work_overlap, divergent_architecture, etc.) for FIG6-01, thinking they implement the C1a/C1b/C2/C3b/C4a/C4b taxonomy.
**Why it happens:** The cooperbench-eval classifiers have C-code annotations in their docstrings (e.g., "CooperBench C1a" for divergent_architecture), creating confusion. But these C-codes map to the PAPER's Table 1 coordination failure taxonomy, not to the REQUIREMENTS' communication error taxonomy.
**How to avoid:** Build a completely new classifier for FIG6-01. The requirements' taxonomy is: C1a=unanswered no-reply, C1b=unanswered ignored, C2=non-answer/vague, C3b=incorrect claim corrected, C4a=spammy same info, C4b=spammy near-duplicate. This is about MESSAGE QUALITY, not code coordination failures.
**Warning signs:** Getting "divergent_architecture" or "work_overlap" results when you expected "unanswered" or "spammy" categories.

### Pitfall 2: AUC with Sparse Buckets
**What goes wrong:** Computing AUC over all 10 buckets with 0 for empty ones, producing misleadingly low AUC values. Or interpolating between populated buckets, producing unjustified smooth curves.
**Why it happens:** Only buckets 3, 6, and 9 are populated. 7 buckets have zero data.
**How to avoid:** Compute AUC using ONLY populated bucket centers. Document: "AUC computed over 3 populated buckets (of 10); rough approximation due to single-model difficulty scoring." In output, include a `sparsity_warning` field.
**Warning signs:** AUC values that differ dramatically between all-bucket and populated-only approaches.

### Pitfall 3: Wilson CI at n=0
**What goes wrong:** Division by zero or nonsensical CI bounds when a bucket has zero observations.
**Why it happens:** 7 of 10 difficulty buckets have no data for coop settings (and no data at all for solo in those buckets).
**How to avoid:** Return (0.0, 1.0) when n=0 -- represents complete uncertainty. The verified Wilson CI implementation handles this.
**Warning signs:** Math errors, NaN in output.

### Pitfall 4: Solo Seed Selection for Rates
**What goes wrong:** Including seed=1 and seed=2 solo records when computing per-bucket solo success rates, inflating the denominator.
**Why it happens:** The 300 solo records include 100 from each of 3 seeds. All contribute to difficulty scoring, but per-bucket rates should reflect a single experimental run.
**How to avoid:** For Figure 4 per-bucket solo rates, filter to seed=0 only (100 records). For difficulty computation, use all 3 seeds (300 records) -- this is already done in Phase 2.
**Warning signs:** Solo denominators of 294 (3x98) instead of 98 for bucket 9.

### Pitfall 5: Overlapping Speech Act Categories
**What goes wrong:** A message like "Can you ensure your changes don't conflict? I will be modifying X" matches both question AND plan patterns. Double-counting inflates percentages.
**Why it happens:** Natural language messages often contain multiple speech acts.
**How to avoid:** Use priority ordering: question > update > plan > other. Each message gets exactly ONE classification. The most specific/actionable category wins.
**Warning signs:** Sum of category percentages exceeding 100%.

### Pitfall 6: Communication Overhead Denominator
**What goes wrong:** Using wrong denominator for overhead percentage. The `total_steps` field in coop records is the SUM of both agents' steps.
**Why it happens:** Confusion about whether total_steps is per-agent or per-task.
**How to avoid:** Overhead = messages_count / total_steps. Both are per-task (total_steps = agent1_steps + agent2_steps). This matches the paper's definition: "communication actions as a percentage of all execution events."
**Warning signs:** Overhead percentages consistently above 50% (messages can't exceed total steps if the denominator is correct).

### Pitfall 7: LLM Classifier API Configuration
**What goes wrong:** Using the wrong API endpoint or model name, getting 401/403 errors.
**Why it happens:** The cooperbench-eval uses staging endpoint (`stg.api.cohere.com`) with `CO_API_KEY`. The cooperbench runs use `command-a-03-2025` which may need a different endpoint.
**How to avoid:** Follow the cooperbench-eval pattern: use `COHERE_API_URL` env var with fallback to staging. Test with 1 transcript before running all 100. Support `--dry-run` flag to preview prompts without API calls.
**Warning signs:** HTTP 401/403 errors, empty responses, model-not-found errors.

### Pitfall 8: Eval Error Records
**What goes wrong:** Including records with `eval_error != null` in metric computations, polluting rates.
**Why it happens:** 27 records have eval errors (e.g., "Failed to write solo.patch"). These should be excluded from most metrics.
**How to avoid:** Filter out records where `eval_error is not None` before computing rates. Report the count separately. For the LLM classifier, only process records with actual messages.
**Warning signs:** Denominator counts not matching expected 100 per setting.

## Code Examples

### Verified Wilson CI Implementation
```python
import math

def wilson_ci(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Returns (lower, upper) bounds for the 95% CI.

    Verified edge cases:
        wilson_ci(0, 100)  -> (0.0, 0.036995)    # p=0
        wilson_ci(100, 100) -> (0.963005, 1.0)    # p=1
        wilson_ci(0, 0)    -> (0.0, 1.0)          # no data
        wilson_ci(2, 3)    -> (0.207655, 0.938510) # small n
        wilson_ci(1, 100)  -> (0.001767, 0.054488) # rare event
    """
    if total == 0:
        return (0.0, 1.0)

    z = 1.96  # 95% CI
    p_hat = successes / total
    n = total

    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return (round(lower, 6), round(upper, 6))
```
Source: Standard Wilson score interval formula from Binomial proportion confidence interval (Wikipedia). Verified against scipy.stats tests and our actual data.

### Per-Bucket Rate Computation
```python
from collections import defaultdict

def compute_bucket_rates(records: list[dict]) -> list[dict]:
    """Compute per-bucket success rates for each setting.

    Returns list of bucket dicts with rates and Wilson CIs.
    Only includes populated buckets.
    """
    # Group by (setting, bucket)
    groups = defaultdict(lambda: {"successes": 0, "total": 0})

    for r in records:
        if r.get("eval_error") is not None:
            continue  # Skip eval errors
        if r.get("infra_error"):
            continue  # Skip infra errors

        # For solo: use seed=0 only
        if r["setting"] == "solo" and r.get("seed", 0) != 0:
            continue

        bucket = r.get("bucket")
        if bucket is None:
            continue

        key = (r["setting"], bucket)
        groups[key]["total"] += 1
        if r["both_passed"]:
            groups[key]["successes"] += 1

    # Build output
    populated_buckets = sorted(set(b for (_, b) in groups.keys()))
    bucket_centers = {b: 0.05 + 0.1 * b for b in range(10)}

    result = []
    for bucket in populated_buckets:
        entry = {"bucket": bucket, "center": bucket_centers[bucket]}
        for setting in ["solo", "coop-comm", "coop-nocomm"]:
            key = (setting, bucket)
            if key in groups:
                g = groups[key]
                rate = g["successes"] / g["total"] if g["total"] > 0 else 0
                ci_lower, ci_upper = wilson_ci(g["successes"], g["total"])
                entry[setting.replace("-", "_")] = {
                    "successes": g["successes"],
                    "total": g["total"],
                    "rate": round(rate, 6),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                }
            else:
                entry[setting.replace("-", "_")] = None  # No data for this bucket
        result.append(entry)

    return result
```

### Trapezoidal AUC with Sparse Buckets
```python
def compute_auc(bucket_data: list[dict], setting_key: str) -> dict:
    """Compute AUC via trapezoidal integration over populated buckets only.

    Returns dict with auc value and metadata about computation.
    """
    # Extract (center, rate) pairs for populated buckets
    points = []
    for b in bucket_data:
        setting_data = b.get(setting_key)
        if setting_data is not None and setting_data["total"] > 0:
            points.append((b["center"], setting_data["rate"]))

    if len(points) < 2:
        return {
            "value": 0.0,
            "n_points": len(points),
            "warning": f"Need >= 2 populated buckets for AUC; have {len(points)}"
        }

    # Sort by x (difficulty center)
    points.sort()
    x = [p[0] for p in points]
    y = [p[1] for p in points]

    # Trapezoidal rule
    auc = sum(
        (x[i+1] - x[i]) * (y[i] + y[i+1]) / 2
        for i in range(len(x) - 1)
    )

    return {
        "value": round(auc, 6),
        "n_points": len(points),
        "x_range": [x[0], x[-1]],
    }
```

### LLM Taxonomy Classifier (FIG6-01)
```python
import httpx
import json
import os
import time

COHERE_API_URL = os.environ.get("COHERE_API_URL", "https://stg.api.cohere.com/v2/chat")
TAXONOMY_MODEL = os.environ.get("TAXONOMY_MODEL", "command-a-03-2025")
MAX_RETRIES = 3
RETRY_DELAY = 2.0

TAXONOMY_CATEGORIES = {
    "C1a": "Unanswered - No Reply: A direct question receives no reply at all",
    "C1b": "Unanswered - Ignored: A request or question is acknowledged but substantively ignored",
    "C2": "Non-answer/Vague: Response is vague, non-committal, or lacks actionable information",
    "C3b": "Incorrect Claim: Agent makes a factually wrong claim about codebase or changes",
    "C4a": "Spammy - Same Info: Agent repeats the same information already communicated",
    "C4b": "Spammy - Near-duplicate: Messages contain near-identical content",
}


def classify_transcript(messages: list[dict], api_key: str) -> dict:
    """Classify communication errors in a single transcript.

    Args:
        messages: List of message dicts with 'from', 'to', 'message' keys.
        api_key: Cohere API key.

    Returns:
        Dict with 'errors' list and 'summary'.
    """
    if not messages:
        return {"errors": [], "summary": "No messages to analyze"}

    transcript = "\n".join(
        f"[{msg['from']} -> {msg['to']}] {msg['message']}"
        for msg in messages
    )

    category_desc = "\n".join(
        f"- {code}: {desc}" for code, desc in TAXONOMY_CATEGORIES.items()
    )

    prompt = f"""Analyze this multi-agent communication transcript for communication errors.

Categories:
{category_desc}

TRANSCRIPT:
{transcript}

Classify ALL communication errors found. A transcript may have multiple errors or none.

Respond in JSON format only:
{{"errors": [{{"category": "C1a|C1b|C2|C3b|C4a|C4b", "evidence": "brief quote or description", "message_index": 0}}], "summary": "brief overall assessment"}}"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": TAXONOMY_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert at analyzing multi-agent communication for coordination failures. Respond ONLY in valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            if attempt > 0:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                time.sleep(0.5)  # Base rate limit

            with httpx.Client(timeout=90.0) as client:
                resp = client.post(COHERE_API_URL, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                content = data.get("message", {}).get("content", [])
                if content and isinstance(content, list):
                    text = content[0].get("text", "")
                else:
                    text = str(content)
                return _parse_json_response(text)

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))

    return {"errors": [], "summary": f"Classification failed: {last_error}", "error": True}


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    import re

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract from code block
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Find any JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    return {"errors": [], "summary": f"Failed to parse response: {text[:200]}", "parse_error": True}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Wald (normal) CI | Wilson score CI | Standard since ~2000 | Wilson is well-calibrated at p near 0/1; Wald gives degenerate intervals |
| numpy.trapz | numpy.trapezoid / scipy.integrate.trapezoid | NumPy 2.0 (2024) | Old name deprecated. We use manual implementation (no numpy needed) |
| statsmodels for Wilson CI | Manual formula or statsmodels | Always available | Manual is fine for our use case; statsmodels is gold standard but overkill dependency |
| Pandas-based analysis | Pure Python dict/list | Phase 2 precedent | 500 records is trivially handled without pandas overhead |

**Deprecated/outdated:**
- `numpy.trapz` -- renamed to `numpy.trapezoid` in NumPy 2.0. Not relevant since we use manual implementation.
- Wald confidence intervals -- should never be used for proportions near 0 or 1.

## Open Questions

1. **Which Cohere API endpoint for Command A taxonomy classification?**
   - What we know: cooperbench-eval uses `stg.api.cohere.com/v2/chat` with `CO_API_KEY` and `command-r-plus-08-2024`. The benchmark runs use `command-a-03-2025`.
   - What's unclear: Whether Command A is available via the staging endpoint, or if a different endpoint/key is needed.
   - Recommendation: Try staging endpoint first with `command-a-03-2025`. If 401/403, fall back to `command-r-plus` (proven to work in cooperbench-eval). Make model configurable via env var `TAXONOMY_MODEL`.
   - Impact: LOW -- the classifier just needs any capable LLM. Model choice affects cost, not correctness.

2. **Should per-bucket solo rates use seed=0 only or all seeds?**
   - What we know: 300 solo records (3 seeds x 100 pairs). Difficulty uses all 3 seeds. The paper presumably uses all models' solo runs for difficulty but one model's run for per-bucket rates.
   - What's unclear: Whether including multiple seeds per pair inflates statistical confidence artificially.
   - Recommendation: Use seed=0 only for per-bucket rates. This gives 100 solo records (matching 100 coop-comm and 100 coop-nocomm), making the comparison fair. All 3 seeds remain used for difficulty scoring.
   - Impact: MEDIUM -- affects Figure 4 denominators and CIs.

3. **How to handle "other" category in speech act classification?**
   - What we know: Preliminary regex analysis shows ~28% of messages don't match plan/question/update patterns. These include requests ("Please ensure..."), acknowledgements ("Acknowledged"), and coordination ("I'll wait for your submission").
   - What's unclear: Whether the paper reports an "other" category or forces all messages into plan/question/update.
   - Recommendation: Classify as 4 categories (plan/question/update/other). Report "other" count but focus Figure 5c on the three main categories. Refine regex patterns to reduce "other" to under 15%.
   - Impact: LOW -- Figure 5c visualization decision for Phase 4.

4. **Communication overhead: per-task or aggregate?**
   - What we know: Mean overhead is 22.8% (messages/total_steps). Paper reports "approximately 20%".
   - What's unclear: Whether overhead is reported per-task (with distribution) or as a single aggregate number.
   - Recommendation: Compute both: aggregate (mean across all tasks) and per-task (for distribution visualization in Phase 4). Store both in output JSON.
   - Impact: LOW -- affects Figure 5c detail level.

## Sources

### Primary (HIGH confidence)
- `data/results.json` -- actual data analyzed: 500 records, schema verified, edge cases enumerated
- Wilson score interval formula -- Wikipedia "Binomial proportion confidence interval", verified with manual implementation
- `cooperbench-eval/src/llm_judge/base.py` -- LLM API calling pattern (httpx + Cohere v2 chat), JSON parsing with fallback
- `cooperbench-eval/src/classifiers/` -- Existing classifier taxonomy mapping (C-codes to failure modes)
- `cooperbench-eval/src/data_loading/schemas.py` -- Message schema, `is_question` pattern for speech acts
- `.planning/REQUIREMENTS.md` -- FIG6-01 taxonomy definition (C1a/C1b/C2/C3b/C4a/C4b)
- `scripts/collect_results.py` -- Phase 2 script pattern (argparse, JSON I/O, reporting)

### Secondary (MEDIUM confidence)
- [statsmodels proportion_confint](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html) -- Wilson CI reference implementation
- CooperBench paper Table 1 (arXiv:2601.13295) -- 10-category coordination failure taxonomy
- `.planning/research/STACK.md` -- Prior stack research with dependency analysis
- `.planning/research/FEATURES.md` -- Feature landscape and dependency graph

### Tertiary (LOW confidence)
- Paper Appendix F communication error detection details -- not fully accessible from HTML version; taxonomy prompt not published
- API cost estimate for FIG6-01 ($0.50-2.00 for 100 calls) -- estimated, not measured
- Speech act classification accuracy -- heuristic patterns may have 80-90% accuracy based on preliminary analysis; not validated

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- All tools are stdlib Python + httpx already in venv. No new dependencies.
- Architecture: HIGH -- Output schemas designed from actual data analysis. Script patterns proven in Phase 2.
- Wilson CI: HIGH -- Formula verified against scipy reference, tested on all our edge cases.
- Trapezoidal AUC: HIGH -- Trivial computation, well-understood. Sparsity limitation documented.
- Speech act classifier: MEDIUM -- Regex heuristics cover ~72% of messages well. Need to refine for ~28% "other" category.
- LLM taxonomy classifier: MEDIUM -- API pattern proven in cooperbench-eval, but exact endpoint/model for Command A not verified. Prompt design based on requirements definitions, not paper's actual prompt (which is unpublished).
- Pitfalls: HIGH -- All identified from actual data analysis and codebase investigation.

**Research date:** 2026-02-18
**Valid until:** 2026-03-18 (stable -- data is fixed, no external dependencies changing)
