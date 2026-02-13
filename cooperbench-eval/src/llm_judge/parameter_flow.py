"""LLM-based classifier for parameter flow failures.

CooperBench category C1b: Detects ambiguity about changing interfaces
between agents. Prevalence: 1.3%.

Uses LLM to identify cases where agents pass incorrect, mismatched,
or ambiguous parameters between their components.
"""

from __future__ import annotations

from src.classifiers.base import ClassificationResult, Severity
from src.data_loading.schemas import TaskData
from src.llm_judge.base import LLMClassifier


class ParameterFlowClassifier(LLMClassifier):
    """Detect parameter and interface mismatches between agents."""

    @property
    def name(self) -> str:
        return "parameter_flow"

    @property
    def description(self) -> str:
        return (
            "Detects ambiguity or mismatches in parameters, data formats, "
            "or interfaces passed between agents. "
            "CooperBench C1b — 1.3% prevalence."
        )

    def build_prompt(self, task: TaskData) -> str:
        transcript = self.format_transcript(task.messages)
        return f"""Analyze the following multi-agent transcript for PARAMETER FLOW failures.

A parameter flow failure occurs when:
- Agents use different formats or representations for shared data
- Output from one agent doesn't match the expected input of another
- Intermediate results are passed with wrong types, units, or structure
- Interface contracts between agents are ambiguous or contradictory
- Agents interpret shared parameters differently

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
            "parameter": "What parameter/data is mismatched",
            "agent1_format": "How agent1 represents it",
            "agent2_format": "How agent2 expects it",
            "step_indices": [0, 1]
        }}
    ],
    "reasoning": "Brief explanation of your analysis"
}}

If no parameter flow failures are detected, set detected=false and leave evidence empty."""

    def parse_response(self, response_text: str, task: TaskData) -> ClassificationResult:
        data = self.parse_json_response(response_text)
        if not data or not data.get("detected", False):
            return self._no_detection()

        evidence_items = data.get("evidence", [])
        evidence: list[str] = []
        for item in evidence_items:
            agents = " and ".join(item.get("agents_involved", ["unknown"]))
            param = item.get("parameter", "")
            fmt1 = item.get("agent1_format", "")
            fmt2 = item.get("agent2_format", "")
            evidence.append(f"{agents}: parameter \"{param}\" — {fmt1} vs {fmt2}")

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
