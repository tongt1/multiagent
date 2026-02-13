"""LLM-based classifier for broken commitment failures.

CooperBench category C3b: Detects when agents make confident claims about
changes that were never actually made. Prevalence: 3.7%.

Uses LLM to identify cases where agents promise to do something but
fail to follow through within the transcript.
"""

from __future__ import annotations

from src.classifiers.base import ClassificationResult, Severity
from src.data_loading.schemas import TaskData
from src.llm_judge.base import LLMClassifier


class BrokenCommitmentClassifier(LLMClassifier):
    """Detect broken promises and unfulfilled commitments."""

    @property
    def name(self) -> str:
        return "broken_commitment"

    @property
    def description(self) -> str:
        return (
            "Detects when agents make commitments (promises, plans, agreements) "
            "that are not followed through in subsequent actions. "
            "CooperBench C3b — 3.7% prevalence."
        )

    def build_prompt(self, task: TaskData) -> str:
        transcript = self.format_transcript(task.messages)
        return f"""Analyze the following multi-agent transcript for BROKEN COMMITMENTS.

A broken commitment occurs when an agent:
- Promises to perform an action but never does it
- Agrees to a plan but acts differently
- Claims they will verify/check something but doesn't
- Commits to a specific approach but switches without communicating
- Says they'll handle a particular aspect but the aspect is left undone

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
            "commitment": "What the agent committed to",
            "outcome": "What actually happened (or didn't)",
            "commitment_step": 0,
            "expected_followup_step": 1
        }}
    ],
    "reasoning": "Brief explanation of your analysis"
}}

If no broken commitments are detected, set detected=false and leave evidence empty."""

    def parse_response(self, response_text: str, task: TaskData) -> ClassificationResult:
        data = self.parse_json_response(response_text)
        if not data or not data.get("detected", False):
            return self._no_detection()

        evidence_items = data.get("evidence", [])
        evidence: list[str] = []
        for item in evidence_items:
            agent = item.get("agent", "unknown")
            commitment = item.get("commitment", "")
            outcome = item.get("outcome", "")
            evidence.append(f"{agent} committed: \"{commitment}\" -> Outcome: {outcome}")

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
