"""EvalSample builder for trajectory entry groups.

Converts a list of TrajectoryEntry records (all sharing one run_id)
into a single Inspect AI EvalSample.
"""

from __future__ import annotations

from inspect_ai.log import EvalSample
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ModelOutput,
)
from inspect_ai.scorer import Score

from src.models.trajectory import TrajectoryEntry


def build_baseline_sample(
    entries: list[TrajectoryEntry],
    sample_id: int | str,
    system_prompt: str = "",
) -> EvalSample:
    """Build an EvalSample from a group of trajectory entries sharing one run_id.

    Constructs messages from all solver attempts (for debugging visibility),
    scores from reward and judge entries, and metadata with token usage.

    Args:
        entries: List of TrajectoryEntry records for one run_id, in file order.
        sample_id: Unique sample identifier (integer or string).
        system_prompt: Optional system prompt to include as first message.

    Returns:
        A fully populated EvalSample.
    """
    if not entries:
        raise ValueError("Cannot build sample from empty entry list")

    # Collect solver entries (all entries where "solver" is in agent name)
    solver_entries = [e for e in entries if "solver" in e.agent]
    if not solver_entries:
        raise ValueError(f"No solver entry found for run_id {entries[0].run_id}")

    # Find reward entry (agent == "reward") if present
    reward_entry = next((e for e in entries if e.agent == "reward"), None)

    # Find judge entry (agent contains "judge") if present
    judge_entry = next((e for e in entries if "judge" in e.agent), None)

    # Determine model name from first solver entry
    model_name = solver_entries[0].metadata.get("model", "unknown")

    # --- Build messages list ---
    messages = []

    # System prompt (user decision: include full system prompt if provided)
    if system_prompt:
        messages.append(ChatMessageSystem(content=system_prompt))

    # For EACH solver entry: user message with problem, then assistant with solution
    # Per user decision: "Show all solver attempts including failed/retried ones"
    for solver in solver_entries:
        problem_text = solver.input.get("problem", "")
        solution_text = solver.output.get("solution", "")
        solver_model = solver.metadata.get("model", model_name)

        messages.append(ChatMessageUser(content=problem_text))
        messages.append(ChatMessageAssistant(content=solution_text, model=solver_model))

    # --- Build input field ---
    # EvalSample.input = string containing problem text plus metadata (Pitfall 3: use str type)
    problem_text = solver_entries[0].input.get("problem", "")
    input_parts = [problem_text]

    # Append dataset/difficulty/problem_id from metadata if available
    first_meta = solver_entries[0].metadata
    if "dataset" in first_meta:
        input_parts.append(f"Dataset: {first_meta['dataset']}")
    if "difficulty" in first_meta:
        input_parts.append(f"Difficulty: {first_meta['difficulty']}")
    if "problem_id" in first_meta:
        input_parts.append(f"Problem ID: {first_meta['problem_id']}")

    input_text = "\n".join(input_parts)

    # --- Build target ---
    # Ground truth answer from reward entry metadata, or empty string
    target = ""
    if reward_entry:
        target = str(reward_entry.metadata.get("expected_answer", ""))

    # --- Build scores dict (two scorers per user decision) ---
    scores: dict[str, Score] = {}

    # Ground truth score from reward entry
    if reward_entry:
        reward_value = reward_entry.output.get("reward", 0.0)
        method = reward_entry.output.get("method", "unknown")
        scores["ground_truth"] = Score(
            value=float(reward_value),
            explanation=f"{method} verification",
            answer=reward_entry.metadata.get("predicted_answer"),
        )
    else:
        # Fallback: if no dedicated reward agent, use the reward field from the
        # last solver entry (sample data has reward on each TrajectoryEntry)
        last_solver = solver_entries[-1]
        if last_solver.reward is not None:
            scores["ground_truth"] = Score(
                value=float(last_solver.reward),
                explanation="trajectory reward field",
            )

    # Judge score (only include if judge entry exists)
    if judge_entry:
        judge_score_val = judge_entry.output.get("score", 0.0)
        scores["judge"] = Score(
            value=float(judge_score_val),
            explanation="LLM judge assessment",
        )

    # --- Build ModelOutput ---
    last_solution = solver_entries[-1].output.get("solution", "")
    output = ModelOutput.from_content(
        model=model_name,
        content=last_solution,
        stop_reason="stop",
    )

    # --- Build metadata dict (token usage per user decision) ---
    metadata: dict[str, object] = {
        "mode": "baseline",
        "run_id": entries[0].run_id,
        "model": model_name,
    }

    # Token usage: handle both int and dict formats (Pitfall 4)
    token_info = solver_entries[0].metadata.get("tokens", {})
    if isinstance(token_info, dict):
        metadata["prompt_tokens"] = token_info.get("prompt_tokens", 0)
        metadata["completion_tokens"] = token_info.get("completion_tokens", 0)
        metadata["total_tokens"] = token_info.get("total_tokens", 0)
    elif isinstance(token_info, (int, float)):
        metadata["total_tokens"] = int(token_info)

    # Include cost if available
    cost = solver_entries[0].metadata.get("cost_usd")
    if cost is not None:
        metadata["cost_usd"] = cost

    return EvalSample(
        id=sample_id,
        epoch=1,  # Required field (Pitfall 1)
        input=input_text,
        target=target,
        messages=messages,
        output=output,
        scores=scores if scores else None,
        metadata=metadata,
    )
