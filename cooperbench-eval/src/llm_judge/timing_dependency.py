"""LLM-based classifier for timing dependency failures.

CooperBench category C4b: Detects when agents agree on an order of
operations but have no enforceable plan. Prevalence: 1.1%.

Uses LLM to identify race conditions, ordering assumptions, and
temporal coordination failures between agents.
"""

from __future__ import annotations

from src.classifiers.base import ClassificationResult, Severity
from src.data_loading.schemas import TaskData
from src.llm_judge.base import LLMClassifier


class TimingDependencyClassifier(LLMClassifier):
    """Detect timing and ordering coordination failures."""

    @property
    def name(self) -> str:
        return "timing_dependency"

    @property
    def description(self) -> str:
        return (
            "Detects when agents have implicit ordering assumptions that are "
            "not enforced, leading to race conditions or premature actions. "
            "CooperBench C4b — 1.1% prevalence."
        )

    def build_prompt(self, task: TaskData) -> str:
        transcript = self.format_transcript(task.messages)
        return f"""Analyze the following multi-agent transcript for TIMING DEPENDENCY failures.

A timing dependency failure occurs when:
- An agent acts on another agent's results before they are ready
- Agents assume a specific execution order without enforcing it
- One agent's action depends on timing that isn't guaranteed
- Agents proceed in parallel when sequential execution was required
- There's an implicit "happens-before" relationship that isn't enforced

TRANSCRIPT:
{transcript}

TASK: {task.task_description}

Respond in JSON format:
{{
    "detected": true/false,
    "confidence": 0.0-1.0,
    "severity": "low"/"medium"/"high"/"critical",
    "evidence": [
        {{
            "agents_involved": ["agent1", "agent2"],
            "timing_issue": "Description of the timing problem",
            "expected_order": "What order was assumed",
            "actual_order": "What actually happened",
            "step_indices": [0, 1]
        }}
    ],
    "reasoning": "Brief explanation of your analysis"
}}

If no timing dependency failures are detected, set detected=false and leave evidence empty."""

    def parse_response(self, response_text: str, task: TaskData) -> ClassificationResult:
        data = self.parse_json_response(response_text)
        if not data or not data.get("detected", False):
            return self._no_detection()

        evidence_items = data.get("evidence", [])
        evidence: list[str] = []
        for item in evidence_items:
            agents = " and ".join(item.get("agents_involved", ["unknown"]))
            issue = item.get("timing_issue", "")
            expected = item.get("expected_order", "")
            actual = item.get("actual_order", "")
            evidence.append(f"{agents}: {issue}")
            if expected and actual:
                evidence.append(f"  Expected: {expected} -> Actual: {actual}")

        severity_map = {"low": Severity.LOW, "medium": Severity.MEDIUM, "high": Severity.HIGH, "critical": Severity.CRITICAL}
        severity = severity_map.get(data.get("severity", "medium"), Severity.MEDIUM)
        confidence = min(1.0, max(0.0, float(data.get("confidence", 0.5))))

        return ClassificationResult(
            classifier_name=self.name,
            detected=True,
            severity=severity,
            confidence=round(confidence, 3),
            evidence=evidence,
            details={
                "raw_evidence": evidence_items,
                "reasoning": data.get("reasoning", ""),
            },
        )
