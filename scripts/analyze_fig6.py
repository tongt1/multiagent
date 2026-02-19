#!/usr/bin/env python3
"""Classify communication errors in coop-comm transcripts using LLM-based taxonomy.

Implements the paper's C1a/C1b/C2/C3b/C4a/C4b message-quality error taxonomy.
This is a NEW classifier -- NOT reusing cooperbench-eval classifiers (which use the
same C-codes for a different taxonomy: coordination failure modes from Table 1).

The taxonomy here describes MESSAGE QUALITY errors:
  C1a: Unanswered - No Reply (a direct question receives no reply at all)
  C1b: Unanswered - Ignored (a request is acknowledged but substantively ignored)
  C2:  Non-answer/Vague (response is vague, non-committal, lacks actionable info)
  C3b: Incorrect Claim - Corrected (factually wrong claim about codebase/changes)
  C4a: Spammy - Same Info (repeats information already communicated)
  C4b: Spammy - Near-duplicate (near-identical content, copy-paste blocks)

Reads: data/results.json
Writes: data/fig6_metrics.json

Usage:
    python scripts/analyze_fig6.py                     # Classify all transcripts
    python scripts/analyze_fig6.py --dry-run            # Preview prompt without API calls
    python scripts/analyze_fig6.py --limit 3            # Classify first 3 only
    python scripts/analyze_fig6.py --resume             # Resume from partial output
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    print(
        "ERROR: httpx not installed. Activate the cooperbench venv first:\n"
        "  source repos/CooperBench/.venv/bin/activate",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "results.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "fig6_metrics.json"

COHERE_API_URL = os.environ.get(
    "COHERE_API_URL", "https://stg.api.cohere.com/v2/chat"
)
TAXONOMY_MODEL = os.environ.get("TAXONOMY_MODEL", "command-a-03-2025")
MAX_RETRIES = 3
RETRY_DELAY = 2.0
BASE_RATE_LIMIT = 0.5  # seconds between requests

VALID_CATEGORIES = {"C1a", "C1b", "C2", "C3b", "C4a", "C4b"}

TAXONOMY_DESCRIPTIONS = {
    "C1a": "Unanswered - No Reply: A direct question receives no reply at all from the other agent.",
    "C1b": "Unanswered - Ignored: A request or question is acknowledged but substantively ignored.",
    "C2": "Non-answer/Vague: A response is given but it is vague, non-committal, or lacks actionable information.",
    "C3b": "Incorrect Claim - Corrected: An agent makes a factually wrong claim about the codebase or their changes that needs correction.",
    "C4a": "Spammy - Same Info: An agent repeats the same information they already communicated.",
    "C4b": "Spammy - Near-duplicate: Messages contain near-identical content (copy-paste status blocks, repeated file listings).",
}

# ---------------------------------------------------------------------------
# Taxonomy prompt builder
# ---------------------------------------------------------------------------


def build_taxonomy_prompt(messages: list[dict]) -> str:
    """Build the classification prompt for a transcript.

    Args:
        messages: List of message dicts with 'from', 'to', 'message' keys.

    Returns:
        The prompt string for the LLM classifier.
    """
    transcript_lines = []
    for i, msg in enumerate(messages):
        sender = msg.get("from", "unknown")
        receiver = msg.get("to", "unknown")
        text = msg.get("message", "")
        transcript_lines.append(f"[{i}] [{sender} -> {receiver}] {text}")

    transcript = "\n".join(transcript_lines)

    category_desc = "\n".join(
        f"- {code}: {desc}" for code, desc in TAXONOMY_DESCRIPTIONS.items()
    )

    return f"""Analyze the following multi-agent communication transcript for communication errors.

Classify each error found using ONLY these categories:
{category_desc}

TRANSCRIPT:
{transcript}

Classify ALL communication errors found. A transcript may have multiple errors or none.
For each error, quote or describe the specific problematic exchange as evidence.

Respond in JSON format ONLY (no markdown, no explanation):
{{"errors": [{{"category": "C1a|C1b|C2|C3b|C4a|C4b", "evidence": "quote or describe the problematic exchange", "message_index": 0}}], "summary": "brief overall assessment"}}

If no communication errors are found, return:
{{"errors": [], "summary": "No errors detected"}}"""


# ---------------------------------------------------------------------------
# JSON response parser
# ---------------------------------------------------------------------------


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response with fallback strategies.

    Tries:
    1. Direct json.loads
    2. Extract from ```json ... ``` code block
    3. Find first { to last }
    4. Fallback: return error dict
    """
    # Strategy 1: Direct parse
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Extract from code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: Find any JSON object (first { to last })
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            pass

    return {
        "errors": [],
        "summary": f"Failed to parse response: {text[:200]}",
        "parse_error": True,
    }


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def classify_transcript(messages: list[dict], api_key: str) -> dict:
    """Run LLM classifier on a single transcript.

    Args:
        messages: List of message dicts with 'from', 'to', 'message' keys.
        api_key: Cohere API key.

    Returns:
        Dict with 'errors' list and 'summary'. On failure after retries,
        returns error dict with 'error': True.
    """
    if not messages:
        return {"errors": [], "summary": "No messages to analyze"}

    prompt = build_taxonomy_prompt(messages)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": TAXONOMY_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert at analyzing multi-agent communication "
                    "transcripts. Respond only in valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            if attempt > 0:
                delay = RETRY_DELAY * (2 ** attempt)
                print(f"    Retry {attempt + 1}/{MAX_RETRIES} after {delay:.1f}s...")
                time.sleep(delay)

            with httpx.Client(timeout=90.0) as client:
                resp = client.post(COHERE_API_URL, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

                # Extract text from Cohere v2 response format
                content = data.get("message", {}).get("content", [])
                if content and isinstance(content, list):
                    text = content[0].get("text", "")
                else:
                    text = str(content)

                return _parse_json_response(text)

        except httpx.HTTPStatusError as e:
            last_error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            print(f"    API error: {last_error}", file=sys.stderr)
        except httpx.TimeoutException as e:
            last_error = f"Timeout: {e}"
            print(f"    Timeout: {last_error}", file=sys.stderr)
        except Exception as e:
            last_error = str(e)
            print(f"    Error: {last_error}", file=sys.stderr)

    return {
        "errors": [],
        "summary": f"Classification failed after {MAX_RETRIES} attempts: {last_error}",
        "error": True,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_classification(result: dict, task_id: int) -> list[str]:
    """Validate a single classification result. Returns list of warnings."""
    warnings = []

    if not isinstance(result.get("errors"), list):
        warnings.append(f"Task {task_id}: 'errors' is not a list")
        return warnings

    for i, err in enumerate(result["errors"]):
        cat = err.get("category", "MISSING")
        if cat not in VALID_CATEGORIES:
            warnings.append(
                f"Task {task_id}, error {i}: invalid category '{cat}'"
            )
        if not err.get("evidence"):
            warnings.append(
                f"Task {task_id}, error {i}: missing evidence"
            )

    return warnings


# ---------------------------------------------------------------------------
# Frequency aggregation
# ---------------------------------------------------------------------------


def compute_frequency(classifications: list[dict]) -> dict:
    """Compute per-category frequency counts from all classifications.

    Returns dict with:
    - Per-category: count, pct_of_errors, pct_of_transcripts
    - Summary statistics
    """
    total_transcripts = len(classifications)
    total_errors = 0
    category_counts = {cat: 0 for cat in sorted(VALID_CATEGORIES)}
    category_transcripts = {cat: set() for cat in sorted(VALID_CATEGORIES)}
    transcripts_with_errors = 0
    api_failures = 0

    for i, cls in enumerate(classifications):
        if cls.get("api_error", False):
            api_failures += 1
            continue

        errors = cls.get("errors", [])
        if errors:
            transcripts_with_errors += 1

        for err in errors:
            cat = err.get("category")
            if cat in VALID_CATEGORIES:
                category_counts[cat] += 1
                category_transcripts[cat].add(i)
                total_errors += 1

    # Build frequency table
    frequency = {}
    for cat in sorted(VALID_CATEGORIES):
        count = category_counts[cat]
        pct_of_errors = (count / total_errors * 100) if total_errors > 0 else 0.0
        pct_of_transcripts = (
            (len(category_transcripts[cat]) / total_transcripts * 100)
            if total_transcripts > 0
            else 0.0
        )
        frequency[cat] = {
            "count": count,
            "pct_of_errors": round(pct_of_errors, 2),
            "pct_of_transcripts": round(pct_of_transcripts, 2),
        }

    transcripts_without_errors = total_transcripts - transcripts_with_errors - api_failures

    summary = {
        "total_transcripts": total_transcripts,
        "transcripts_with_errors": transcripts_with_errors,
        "transcripts_without_errors": transcripts_without_errors,
        "total_errors": total_errors,
        "api_failures": api_failures,
        "mean_errors_per_transcript": (
            round(total_errors / total_transcripts, 2)
            if total_transcripts > 0
            else 0.0
        ),
    }

    return frequency, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify communication errors in coop-comm transcripts."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input results.json path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output fig6_metrics.json path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print first transcript's prompt without making API calls",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N transcripts (0=all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output, skip already classified",
    )
    args = parser.parse_args()

    # Load input data
    if not args.input.is_file():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    with open(args.input) as f:
        records = json.load(f)

    # Filter to coop-comm records with messages
    coop_comm = [
        r for r in records
        if r["setting"] == "coop-comm" and r.get("messages_count", 0) > 0
    ]
    print(f"Found {len(coop_comm)} coop-comm transcripts with messages")

    if not coop_comm:
        print("ERROR: No coop-comm transcripts found", file=sys.stderr)
        sys.exit(1)

    # Apply limit
    if args.limit > 0:
        coop_comm = coop_comm[: args.limit]
        print(f"Limiting to first {args.limit} transcripts")

    # Dry run: print prompt for first transcript
    if args.dry_run:
        first = coop_comm[0]
        prompt = build_taxonomy_prompt(first["messages"])
        print("\n" + "=" * 60)
        print(f"DRY RUN: Prompt for task {first['task_id']} ({first['repo']})")
        print(f"Messages: {len(first['messages'])}")
        print("=" * 60)
        print(prompt)
        print("=" * 60)
        print(f"\nPrompt length: {len(prompt)} chars")
        print("Would classify with:")
        print(f"  Model: {TAXONOMY_MODEL}")
        print(f"  API: {COHERE_API_URL}")
        print(f"  Temperature: 0.0")
        print(f"  Max tokens: 1024")
        return

    # Get API key
    api_key = os.environ.get("CO_API_KEY") or os.environ.get("COHERE_API_KEY")
    if not api_key:
        print(
            "ERROR: No API key found. Set CO_API_KEY or COHERE_API_KEY environment variable.\n"
            "  export CO_API_KEY='your-key-here'",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resume support: load existing classifications
    existing_classifications = {}
    if args.resume and args.output.is_file():
        with open(args.output) as f:
            existing_data = json.load(f)
        for cls in existing_data.get("classifications", []):
            existing_classifications[cls["task_id"]] = cls
        print(f"Resuming: {len(existing_classifications)} transcripts already classified")

    # Classify all transcripts
    classifications = []
    all_warnings = []
    total = len(coop_comm)
    start_time = time.time()

    print(f"\nClassifying {total} transcripts...")
    print(f"Model: {TAXONOMY_MODEL}")
    print(f"API: {COHERE_API_URL}")
    print()

    for i, record in enumerate(coop_comm):
        task_id = record["task_id"]
        repo = record["repo"]
        messages = record["messages"]

        # Skip if already classified (resume mode)
        if task_id in existing_classifications:
            classifications.append(existing_classifications[task_id])
            print(f"[{i + 1}/{total}] task {task_id} {repo}: SKIPPED (already classified)")
            continue

        # Rate limiting between requests
        if i > 0 and task_id not in existing_classifications:
            time.sleep(BASE_RATE_LIMIT)

        # Classify
        result = classify_transcript(messages, api_key)

        # Validate
        warnings = validate_classification(result, task_id)
        all_warnings.extend(warnings)

        # Count valid errors
        valid_errors = [
            e for e in result.get("errors", [])
            if e.get("category") in VALID_CATEGORIES
        ]
        error_count = len(valid_errors)
        has_api_error = result.get("error", False) or result.get("parse_error", False)

        # Build classification record
        cls_record = {
            "task_id": task_id,
            "repo": repo,
            "messages_count": len(messages),
            "errors": result.get("errors", []),
            "summary": result.get("summary", ""),
            "api_error": has_api_error,
        }
        classifications.append(cls_record)

        # Progress
        status = "API ERROR" if has_api_error else f"{error_count} errors found"
        print(f"[{i + 1}/{total}] task {task_id} {repo}: {status}")

    elapsed = time.time() - start_time

    # Print warnings
    if all_warnings:
        print(f"\nWarnings ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"  WARNING: {w}")

    # Compute frequency aggregation
    frequency, summary_stats = compute_frequency(classifications)

    # Build output
    output = {
        "classifications": classifications,
        "frequency": frequency,
        "summary": summary_stats,
        "metadata": {
            "input_file": str(args.input),
            "model": TAXONOMY_MODEL,
            "api_url": COHERE_API_URL,
            "taxonomy": "C1a/C1b/C2/C3b/C4a/C4b (message quality errors, NOT coordination failures)",
            "classifier_note": "NEW classifier per requirements, not reusing cooperbench-eval classifiers",
            "elapsed_seconds": round(elapsed, 1),
        },
    }

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote classifications to {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("FIGURE 6 CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total transcripts: {summary_stats['total_transcripts']}")
    print(f"With errors: {summary_stats['transcripts_with_errors']}")
    print(f"Without errors: {summary_stats['transcripts_without_errors']}")
    print(f"API failures: {summary_stats['api_failures']}")
    print(f"Total errors found: {summary_stats['total_errors']}")
    print(f"Mean errors/transcript: {summary_stats['mean_errors_per_transcript']}")
    print()
    print("Frequency table:")
    print(f"  {'Category':<8} {'Count':>6} {'% of Errors':>12} {'% of Transcripts':>17}")
    print(f"  {'-' * 8} {'-' * 6} {'-' * 12} {'-' * 17}")
    for cat in sorted(VALID_CATEGORIES):
        f = frequency[cat]
        print(
            f"  {cat:<8} {f['count']:>6} {f['pct_of_errors']:>11.1f}% {f['pct_of_transcripts']:>16.1f}%"
        )
    print()
    print(f"Elapsed: {elapsed:.1f}s")
    est_cost = total * 0.015  # rough estimate: ~$0.015 per call with Command A
    print(f"Estimated API cost: ~${est_cost:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
