"""Work overlap classifier: detects when agents duplicate each other's work.

Computes file-path Jaccard similarity between agent patches.
Threshold > 0.3 triggers detection. Also detects function-level overlap
for Python and JavaScript files.
"""

from __future__ import annotations

from itertools import combinations

from src.classifiers.base import BaseClassifier, ClassificationResult, Severity
from src.data_loading.schemas import TaskData

FILE_OVERLAP_THRESHOLD = 0.3
FUNCTION_OVERLAP_THRESHOLD = 0.2


class WorkOverlapClassifier(BaseClassifier):
    """Detect overlapping/duplicate work between agents."""

    @property
    def name(self) -> str:
        return "work_overlap"

    @property
    def description(self) -> str:
        return (
            "Detects when multiple agents modify the same files or functions, "
            "indicating coordination failure and wasted effort."
        )

    def classify(self, task: TaskData) -> ClassificationResult:
        skip = self._skip_if_solo(task)
        if skip is not None:
            return skip

        if len(task.patches) < 2:
            return self._no_detection()

        # Compare all pairs of agent patches
        file_overlaps: list[dict] = []
        function_overlaps: list[dict] = []

        for patch_a, patch_b in combinations(task.patches, 2):
            # File-level overlap
            files_a = set(patch_a.files_modified)
            files_b = set(patch_b.files_modified)

            file_jaccard = _jaccard(files_a, files_b)
            shared_files = sorted(files_a & files_b)

            if file_jaccard > FILE_OVERLAP_THRESHOLD:
                file_overlaps.append({
                    "agents": (patch_a.agent, patch_b.agent),
                    "jaccard": round(file_jaccard, 4),
                    "shared_files": shared_files,
                    "files_a": sorted(files_a),
                    "files_b": sorted(files_b),
                })

            # Function-level overlap (more granular)
            if shared_files:
                func_overlap_info = _compute_function_overlap(
                    patch_a.functions_modified,
                    patch_b.functions_modified,
                    shared_files,
                )
                if func_overlap_info["overlap_ratio"] > FUNCTION_OVERLAP_THRESHOLD:
                    function_overlaps.append({
                        "agents": (patch_a.agent, patch_b.agent),
                        **func_overlap_info,
                    })

        if not file_overlaps and not function_overlaps:
            return self._no_detection()

        # Determine severity
        max_file_jaccard = max((o["jaccard"] for o in file_overlaps), default=0.0)

        if max_file_jaccard > 0.7 or function_overlaps:
            severity = Severity.HIGH
        elif max_file_jaccard > 0.5:
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW

        evidence: list[str] = []
        for overlap in file_overlaps:
            agents = " and ".join(overlap["agents"])
            evidence.append(
                f"File overlap (Jaccard={overlap['jaccard']:.2f}) between {agents}: "
                f"{', '.join(overlap['shared_files'][:5])}"
            )

        for overlap in function_overlaps:
            agents = " and ".join(overlap["agents"])
            evidence.append(
                f"Function-level overlap between {agents}: "
                f"{', '.join(overlap['shared_functions'][:5])}"
            )

        confidence = min(1.0, 0.5 + max_file_jaccard * 0.5 + len(function_overlaps) * 0.15)

        return ClassificationResult(
            classifier_name=self.name,
            detected=True,
            severity=severity,
            confidence=round(confidence, 3),
            evidence=evidence,
            details={
                "file_overlaps": file_overlaps,
                "function_overlaps": function_overlaps,
                "max_file_jaccard": max_file_jaccard,
            },
        )


def _jaccard(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _compute_function_overlap(
    funcs_a: dict[str, list[str]],
    funcs_b: dict[str, list[str]],
    shared_files: list[str],
) -> dict:
    """Compute function-level overlap between two patches on shared files.

    Args:
        funcs_a: Agent A's {file: [function_names]} mapping.
        funcs_b: Agent B's {file: [function_names]} mapping.
        shared_files: Files modified by both agents.

    Returns:
        Dict with overlap_ratio, shared_functions, and per-file details.
    """
    all_funcs_a: set[str] = set()
    all_funcs_b: set[str] = set()
    shared_functions: list[str] = []
    per_file: dict[str, dict] = {}

    for filepath in shared_files:
        fa = set(funcs_a.get(filepath, []))
        fb = set(funcs_b.get(filepath, []))
        all_funcs_a |= fa
        all_funcs_b |= fb

        shared = fa & fb
        if shared:
            shared_functions.extend(f"{filepath}::{f}" for f in sorted(shared))
            per_file[filepath] = {
                "functions_a": sorted(fa),
                "functions_b": sorted(fb),
                "shared": sorted(shared),
            }

    overlap_ratio = _jaccard(all_funcs_a, all_funcs_b) if (all_funcs_a or all_funcs_b) else 0.0

    return {
        "overlap_ratio": round(overlap_ratio, 4),
        "shared_functions": shared_functions,
        "per_file_details": per_file,
    }
