"""LLM-based classifier for unverifiable claims.

CooperBench category C2: Detects when agents make assertions without
checkable evidence. Prevalence: 4.3%.

Uses LLM to identify claims about results, correctness, or completion
that lack supporting evidence in the transcript.
"""

from __future__ import annotations

from src.classifiers.base import ClassificationResult, Severity
from src.data_loading.schemas import TaskData
from src.llm_judge.base import LLMClassifier


class UnverifiableClaimsClassifier(LLMClassifier):
    """Detect assertions without checkable evidence."""

    @property
    def name(self) -> str:
        return "unverifiable_claims"

    @property
    def description(self) -> str:
        return (
            "Detects when agents make assertions about correctness, completion, "
            "or results without providing checkable evidence. "
            "CooperBench C2 — 4.3% prevalence."
        )

    def build_prompt(self, task: TaskData) -> str:
        transcript = self.format_transcript(task.messages)
        return f"""Analyze the following multi-agent transcript for UNVERIFIABLE CLAIMS.

An unverifiable claim occurs when an agent:
- Claims a solution is correct without showing verification steps
- Asserts completion without demonstrable evidence
- Makes confident statements about results that cannot be checked from the transcript
- States agreement or correctness without showing their own reasoning

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
            "agent": "agent_name",
            "claim": "What the agent claimed",
            "why_unverifiable": "Why this claim cannot be verified",
            "step_index": 0
        }}
    ],
    "reasoning": "Brief explanation of your analysis"
}}

If no unverifiable claims are detected, set detected=false and leave evidence empty."""

    def parse_response(self, response_text: str, task: TaskData) -> ClassificationResult:
        data = self.parse_json_response(response_text)
        if not data or not data.get("detected", False):
            return self._no_detection()

        evidence_items = data.get("evidence", [])
        evidence: list[str] = []
        for item in evidence_items:
            agent = item.get("agent", "unknown")
            claim = item.get("claim", "")
            why = item.get("why_unverifiable", "")
            evidence.append(f"{agent}: \"{claim}\" — {why}")

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
