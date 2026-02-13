"""LLM-based classifier for dependency access failures.

CooperBench category C4a: Detects when agents fail to communicate risks
about missing dependencies or access issues. Prevalence: 1.7%.

Uses LLM to identify cases where agents reference resources, data, or
intermediate results that other agents need but cannot access.
"""

from __future__ import annotations

from src.classifiers.base import ClassificationResult, Severity
from src.data_loading.schemas import TaskData
from src.llm_judge.base import LLMClassifier


class DependencyAccessClassifier(LLMClassifier):
    """Detect missing dependency communication between agents."""

    @property
    def name(self) -> str:
        return "dependency_access"

    @property
    def description(self) -> str:
        return (
            "Detects when agents fail to communicate about shared dependencies, "
            "missing data, or access requirements between components. "
            "CooperBench C4a — 1.7% prevalence."
        )

    def build_prompt(self, task: TaskData) -> str:
        transcript = self.format_transcript(task.messages)
        return f"""Analyze the following multi-agent transcript for DEPENDENCY ACCESS failures.

A dependency access failure occurs when:
- An agent references data/results that another agent needs but hasn't received
- Agents assume shared access to resources without verifying
- An agent's output depends on another's input that was never provided
- Risk of missing dependencies is not communicated between agents
- One agent proceeds without waiting for necessary inputs from another

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
            "dependency": "What resource/data is needed",
            "issue": "Why access is problematic",
            "step_indices": [0, 1]
        }}
    ],
    "reasoning": "Brief explanation of your analysis"
}}

If no dependency access failures are detected, set detected=false and leave evidence empty."""

    def parse_response(self, response_text: str, task: TaskData) -> ClassificationResult:
        data = self.parse_json_response(response_text)
        if not data or not data.get("detected", False):
            return self._no_detection()

        evidence_items = data.get("evidence", [])
        evidence: list[str] = []
        for item in evidence_items:
            agents = " and ".join(item.get("agents_involved", ["unknown"]))
            dep = item.get("dependency", "")
            issue = item.get("issue", "")
            evidence.append(f"{agents}: dependency \"{dep}\" — {issue}")

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
