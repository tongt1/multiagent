"""LLM-based classifier for divergent architecture failures.

CooperBench category C1a: Detects when agents make incompatible design
decisions that prevent integration. Prevalence: 29.7%.

Uses LLM to identify semantic disagreements about approaches, frameworks,
data structures, or algorithms that go beyond simple keyword matching.
"""

from __future__ import annotations

from src.classifiers.base import ClassificationResult, Severity
from src.data_loading.schemas import TaskData
from src.llm_judge.base import LLMClassifier


class DivergentArchitectureClassifier(LLMClassifier):
    """Detect incompatible design decisions between agents."""

    @property
    def name(self) -> str:
        return "divergent_architecture"

    @property
    def description(self) -> str:
        return (
            "Detects when agents adopt incompatible approaches, frameworks, "
            "algorithms, or data structures without coordination. "
            "CooperBench C1a — 29.7% prevalence."
        )

    def build_prompt(self, task: TaskData) -> str:
        transcript = self.format_transcript(task.messages)
        return f"""Analyze the following multi-agent transcript for DIVERGENT ARCHITECTURE failures.

A divergent architecture failure occurs when agents make incompatible design decisions:
- Choosing conflicting approaches to solve the same problem
- Using incompatible data representations or formats
- Proposing contradictory algorithms or methods
- Disagreeing on solution strategy without resolving the conflict

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
            "description": "Brief description of the incompatibility",
            "agent1_approach": "What agent1 proposed/did",
            "agent2_approach": "What agent2 proposed/did",
            "step_indices": [0, 1]
        }}
    ],
    "reasoning": "Brief explanation of your analysis"
}}

If no divergent architecture is detected, set detected=false and leave evidence empty."""

    def parse_response(self, response_text: str, task: TaskData) -> ClassificationResult:
        data = self.parse_json_response(response_text)
        if not data or not data.get("detected", False):
            return self._no_detection()

        evidence_items = data.get("evidence", [])
        evidence: list[str] = []
        for item in evidence_items:
            agents = " and ".join(item.get("agents_involved", ["unknown"]))
            desc = item.get("description", "Incompatible approaches detected")
            a1 = item.get("agent1_approach", "")
            a2 = item.get("agent2_approach", "")
            evidence.append(f"{agents}: {desc}")
            if a1 and a2:
                evidence.append(f"  - {item.get('agents_involved', ['?'])[0]}: {a1}")
                evidence.append(f"  - {item.get('agents_involved', ['?', '?'])[-1]}: {a2}")

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
