"""Repetition classifier: detects when agents repeat themselves in loops.

Uses n-gram Jaccard similarity between consecutive same-agent messages
to identify stuck-in-a-loop failure patterns.

Detection thresholds:
    - 2+ consecutive same-agent message pairs with Jaccard > 0.6
    - OR any single pair with Jaccard > 0.85
"""

from __future__ import annotations

from src.classifiers.base import BaseClassifier, ClassificationResult, Severity
from src.data_loading.schemas import Message, TaskData

# Thresholds
MODERATE_SIMILARITY_THRESHOLD = 0.6
HIGH_SIMILARITY_THRESHOLD = 0.85
MIN_MODERATE_PAIRS = 2
NGRAM_SIZE = 3
MIN_WORDS_FOR_NGRAMS = 5  # Skip very short messages


class RepetitionClassifier(BaseClassifier):
    """Detect repetitive messaging loops between agents."""

    @property
    def name(self) -> str:
        return "repetition"

    @property
    def description(self) -> str:
        return (
            "Detects when agents enter repetitive loops, sending near-identical "
            "messages consecutively. Uses n-gram Jaccard similarity."
        )

    def classify(self, task: TaskData) -> ClassificationResult:
        skip = self._skip_if_solo(task)
        if skip is not None:
            return skip

        if len(task.messages) < 2:
            return self._no_detection()

        # Analyze consecutive same-agent message pairs
        high_sim_pairs: list[dict] = []
        moderate_sim_pairs: list[dict] = []

        for agent, agent_msgs in task.agent_messages.items():
            for i in range(len(agent_msgs) - 1):
                msg_a = agent_msgs[i]
                msg_b = agent_msgs[i + 1]

                # Skip very short messages (greetings, acknowledgements)
                if msg_a.word_count < MIN_WORDS_FOR_NGRAMS or msg_b.word_count < MIN_WORDS_FOR_NGRAMS:
                    continue

                sim = _ngram_jaccard(msg_a.content, msg_b.content, n=NGRAM_SIZE)

                pair_info = {
                    "agent": agent,
                    "msg_indices": (msg_a.index, msg_b.index),
                    "similarity": round(sim, 4),
                    "preview_a": msg_a.content[:80].replace("\n", " "),
                    "preview_b": msg_b.content[:80].replace("\n", " "),
                }

                if sim > HIGH_SIMILARITY_THRESHOLD:
                    high_sim_pairs.append(pair_info)
                elif sim > MODERATE_SIMILARITY_THRESHOLD:
                    moderate_sim_pairs.append(pair_info)

        # Determine detection
        detected = False
        severity = Severity.LOW
        evidence: list[str] = []

        if high_sim_pairs:
            detected = True
            severity = Severity.HIGH
            for pair in high_sim_pairs:
                evidence.append(
                    f"High repetition (Jaccard={pair['similarity']:.2f}) between "
                    f"messages {pair['msg_indices'][0]} and {pair['msg_indices'][1]} "
                    f"from {pair['agent']}"
                )

        if len(moderate_sim_pairs) >= MIN_MODERATE_PAIRS:
            detected = True
            if severity == Severity.LOW:
                severity = Severity.MEDIUM
            for pair in moderate_sim_pairs:
                evidence.append(
                    f"Moderate repetition (Jaccard={pair['similarity']:.2f}) between "
                    f"messages {pair['msg_indices'][0]} and {pair['msg_indices'][1]} "
                    f"from {pair['agent']}"
                )

        if not detected:
            return self._no_detection()

        # Confidence based on number and intensity of repetitions
        all_pairs = high_sim_pairs + moderate_sim_pairs
        max_sim = max(p["similarity"] for p in all_pairs)
        confidence = min(1.0, 0.5 + (len(all_pairs) * 0.1) + (max_sim - 0.6) * 0.5)

        return ClassificationResult(
            classifier_name=self.name,
            detected=True,
            severity=severity,
            confidence=round(confidence, 3),
            evidence=evidence,
            details={
                "high_similarity_pairs": high_sim_pairs,
                "moderate_similarity_pairs": moderate_sim_pairs,
                "total_repetitive_pairs": len(all_pairs),
                "max_similarity": max_sim,
            },
        )


def _ngram_jaccard(text_a: str, text_b: str, n: int = 3) -> float:
    """Compute Jaccard similarity between n-gram sets of two texts.

    Args:
        text_a: First text.
        text_b: Second text.
        n: Size of n-grams (default trigrams).

    Returns:
        Jaccard similarity in [0.0, 1.0].
    """
    ngrams_a = _extract_ngrams(text_a, n)
    ngrams_b = _extract_ngrams(text_b, n)

    if not ngrams_a or not ngrams_b:
        return 0.0

    intersection = ngrams_a & ngrams_b
    union = ngrams_a | ngrams_b

    if not union:
        return 0.0

    return len(intersection) / len(union)


def _extract_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    """Extract word-level n-grams from text."""
    words = text.lower().split()
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}
