"""Placeholder misuse classifier: detects TODO/FIXME/stub patterns in patches.

Scans added lines in agent patches for placeholder patterns that indicate
incomplete implementation. Three severity tiers based on pattern type.

Context filtering avoids false positives on:
    - Lines that are comments explaining existing TODOs being fixed
    - Test files that intentionally use stubs
    - Documentation strings mentioning placeholders
"""

from __future__ import annotations

import re

from src.classifiers.base import BaseClassifier, ClassificationResult, Severity
from src.data_loading.schemas import PatchInfo, TaskData

# Tier 1 (HIGH): Code that will actively fail
TIER1_PATTERNS = [
    re.compile(r"\braise\s+NotImplementedError\b"),
    re.compile(r"^\s*pass\s*$"),  # bare pass in a function body
    re.compile(r"\.\.\.\s*$"),  # bare ellipsis as placeholder
]

# Tier 2 (MEDIUM): Explicit markers of incomplete work
TIER2_PATTERNS = [
    re.compile(r"\bTODO\b", re.IGNORECASE),
    re.compile(r"\bFIXME\b", re.IGNORECASE),
    re.compile(r"\bHACK\b", re.IGNORECASE),
    re.compile(r"\bXXX\b"),
    re.compile(r"\bstub\b", re.IGNORECASE),
]

# Tier 3 (LOW): Weak signals
TIER3_PATTERNS = [
    re.compile(r"\bplaceholder\b", re.IGNORECASE),
    re.compile(r"\btemporary\b", re.IGNORECASE),
    re.compile(r"\bnot\s+yet\s+implemented\b", re.IGNORECASE),
    re.compile(r"\bimplement\s+later\b", re.IGNORECASE),
    re.compile(r"\bwork\s*in\s*progress\b", re.IGNORECASE),
]

# Lines matching these patterns are filtered out (false positive reduction)
FALSE_POSITIVE_FILTERS = [
    re.compile(r"^\s*#.*(?:removed|fixed|resolved|addressed|completed)\s+(?:TODO|FIXME)", re.IGNORECASE),
    re.compile(r"^\s*(?://|#|/\*|\*)\s*NOTE:", re.IGNORECASE),
    re.compile(r"test_.*\.py|_test\.py|spec\.", re.IGNORECASE),  # test file indicator
]

# Minimum number of placeholder hits to trigger detection
MIN_TIER1_HITS = 1
MIN_TIER2_HITS = 2
MIN_TIER3_HITS = 3


class PlaceholderMisuseClassifier(BaseClassifier):
    """Detect placeholder patterns indicating incomplete implementation."""

    @property
    def name(self) -> str:
        return "placeholder_misuse"

    @property
    def description(self) -> str:
        return (
            "Detects TODO/FIXME/NotImplementedError/pass/stub patterns in agent patches "
            "that indicate submitted code is incomplete."
        )

    def classify(self, task: TaskData) -> ClassificationResult:
        # This classifier works on patches, not conversations.
        # It applies to both solo and multi-agent tasks.
        if not task.patches:
            return self._no_detection()

        all_hits: list[dict] = []

        for patch in task.patches:
            hits = _scan_patch(patch)
            all_hits.extend(hits)

        if not all_hits:
            return self._no_detection()

        # Count by tier
        tier_counts = {1: 0, 2: 0, 3: 0}
        for hit in all_hits:
            tier_counts[hit["tier"]] += 1

        # Determine if we have enough hits to trigger
        detected = (
            tier_counts[1] >= MIN_TIER1_HITS
            or tier_counts[2] >= MIN_TIER2_HITS
            or tier_counts[3] >= MIN_TIER3_HITS
        )

        if not detected:
            return self._no_detection()

        # Severity from highest tier detected
        if tier_counts[1] >= MIN_TIER1_HITS:
            severity = Severity.HIGH
        elif tier_counts[2] >= MIN_TIER2_HITS:
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW

        evidence: list[str] = []
        for hit in all_hits:
            evidence.append(
                f"[Tier {hit['tier']}] {hit['agent']}: {hit['pattern']} in "
                f"\"{hit['line_preview']}\""
            )

        confidence = min(
            1.0,
            0.3
            + tier_counts[1] * 0.25
            + tier_counts[2] * 0.12
            + tier_counts[3] * 0.06,
        )

        return ClassificationResult(
            classifier_name=self.name,
            detected=True,
            severity=severity,
            confidence=round(confidence, 3),
            evidence=evidence,
            details={
                "total_hits": len(all_hits),
                "tier_counts": tier_counts,
                "hits": all_hits,
            },
        )


def _scan_patch(patch: PatchInfo) -> list[dict]:
    """Scan a patch's added lines for placeholder patterns.

    Returns list of hit dicts with tier, pattern, line info.
    """
    hits: list[dict] = []

    for line in patch.added_lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Apply false positive filters
        if _is_false_positive(stripped, patch):
            continue

        # Check each tier
        for pattern in TIER1_PATTERNS:
            if pattern.search(stripped):
                hits.append({
                    "tier": 1,
                    "agent": patch.agent,
                    "pattern": pattern.pattern,
                    "line_preview": stripped[:100],
                })
                break  # One hit per line
        else:
            for pattern in TIER2_PATTERNS:
                if pattern.search(stripped):
                    hits.append({
                        "tier": 2,
                        "agent": patch.agent,
                        "pattern": pattern.pattern,
                        "line_preview": stripped[:100],
                    })
                    break
            else:
                for pattern in TIER3_PATTERNS:
                    if pattern.search(stripped):
                        hits.append({
                            "tier": 3,
                            "agent": patch.agent,
                            "pattern": pattern.pattern,
                            "line_preview": stripped[:100],
                        })
                        break

    return hits


def _is_false_positive(line: str, patch: PatchInfo) -> bool:
    """Check if a line is likely a false positive.

    Filters:
        - Comment lines that mention fixing/removing TODOs
        - Lines in test files (stubs are expected)
        - Pure comment/docstring lines that just mention patterns
    """
    # Check explicit false positive patterns
    for fp_pattern in FALSE_POSITIVE_FILTERS[:2]:  # First two are line-level filters
        if fp_pattern.search(line):
            return True

    # Check if all modified files are test files
    test_file_pattern = FALSE_POSITIVE_FILTERS[2]
    if patch.files_modified and all(
        test_file_pattern.search(f) for f in patch.files_modified
    ):
        return True

    # Filter docstrings (lines that are purely within triple-quote blocks)
    # Heuristic: if line starts with a docstring marker, skip
    if line.lstrip().startswith(('"""', "'''", '"""', "'''")):
        return True

    return False
