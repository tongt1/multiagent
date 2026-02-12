"""Unresponsiveness classifier: detects when agents ignore questions.

Tracks questions asked by one agent and whether the other agent
responds with topically relevant content. Flags when 2+ questions
go unanswered.
"""

from __future__ import annotations

import re

from src.classifiers.base import BaseClassifier, ClassificationResult, Severity
from src.data_loading.schemas import Message, TaskData

# Minimum keyword overlap to consider a response topically relevant
TOPICAL_RELEVANCE_THRESHOLD = 0.15
MIN_UNANSWERED_QUESTIONS = 2
STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those",
    "in", "on", "at", "to", "for", "with", "from", "by", "of", "about",
    "and", "or", "but", "not", "no", "so", "if", "then", "else",
    "what", "where", "when", "why", "how", "which", "who", "whom", "whose",
    "just", "also", "very", "too", "quite", "rather",
})


class UnresponsivenessClassifier(BaseClassifier):
    """Detect when agents fail to respond to questions from other agents."""

    @property
    def name(self) -> str:
        return "unresponsiveness"

    @property
    def description(self) -> str:
        return (
            "Detects when agents ignore or fail to respond to questions asked by "
            "other agents, indicating communication breakdown."
        )

    def classify(self, task: TaskData) -> ClassificationResult:
        skip = self._skip_if_solo(task)
        if skip is not None:
            return skip

        if len(task.messages) < 2:
            return self._no_detection()

        unanswered: list[dict] = []

        # Walk through messages tracking questions and responses
        for i, msg in enumerate(task.messages):
            if not msg.is_question:
                continue

            question_keywords = _extract_keywords(msg.content)
            if len(question_keywords) < 2:
                # Very short question, skip to avoid false positives
                continue

            # Look for a response from a DIFFERENT agent in the next few messages
            responded = False
            search_window = min(i + 5, len(task.messages))  # Look ahead up to 4 messages

            for j in range(i + 1, search_window):
                response = task.messages[j]
                if response.agent == msg.agent:
                    # Same agent -- skip, could be follow-up
                    continue

                # Check topical relevance
                response_keywords = _extract_keywords(response.content)
                relevance = _keyword_overlap(question_keywords, response_keywords)

                if relevance >= TOPICAL_RELEVANCE_THRESHOLD:
                    responded = True
                    break

            if not responded:
                unanswered.append({
                    "question_index": msg.index,
                    "asker": msg.agent,
                    "question_preview": msg.content[:120].replace("\n", " "),
                    "question_keywords": sorted(question_keywords)[:10],
                })

        if len(unanswered) < MIN_UNANSWERED_QUESTIONS:
            return self._no_detection()

        # Determine severity based on count
        if len(unanswered) >= 5:
            severity = Severity.HIGH
        elif len(unanswered) >= 3:
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW

        evidence = []
        for uq in unanswered:
            evidence.append(
                f"Question at message {uq['question_index']} from {uq['asker']} "
                f"went unanswered: \"{uq['question_preview'][:80]}...\""
            )

        confidence = min(1.0, 0.4 + (len(unanswered) * 0.12))

        return ClassificationResult(
            classifier_name=self.name,
            detected=True,
            severity=severity,
            confidence=round(confidence, 3),
            evidence=evidence,
            details={
                "unanswered_questions": unanswered,
                "total_unanswered": len(unanswered),
                "total_questions": sum(1 for m in task.messages if m.is_question),
            },
        )


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from text, filtering stop words."""
    words = re.findall(r"\b[a-zA-Z_]\w{2,}\b", text.lower())
    return {w for w in words if w not in STOP_WORDS}


def _keyword_overlap(keywords_a: set[str], keywords_b: set[str]) -> float:
    """Compute keyword overlap ratio (Jaccard-like).

    Returns the fraction of question keywords that appear in the response.
    """
    if not keywords_a:
        return 0.0
    return len(keywords_a & keywords_b) / len(keywords_a)
