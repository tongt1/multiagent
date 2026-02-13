"""Trajectory JSONL loader for math-debate multi-agent transcripts.

Loads sample_trajectories.jsonl (one JSON object per line) and converts
to TaskData objects compatible with the classifier pipeline.

Expected JSONL format per line:
    {
        "timestamp": "...",
        "run_id": "...",
        "step_id": 0,
        "agent": "solver_0",
        "action": "generate_solution",
        "input": {"problem": "...", "prompt": "..."},
        "output": {"solution": "..."} | {"feedback": "..."} | {"score": ..., "reasoning": "..."},
        "metadata": {"agent_role": "solver", ...},
        "reward": 0.0,
        "terminal": false,
        "success": null
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.data_loading.schemas import (
    AgentRole,
    EvalResult,
    Message,
    TaskData,
)

logger = logging.getLogger(__name__)


def load_trajectories(filepath: str | Path) -> list[TaskData]:
    """Load all runs from a JSONL trajectory file.

    Groups entries by run_id and converts each run into a TaskData object.

    Args:
        filepath: Path to a .jsonl file with trajectory entries.

    Returns:
        List of TaskData objects, one per unique run_id.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning("Trajectory file not found: %s", filepath)
        return []

    # Parse all entries
    entries: list[dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Malformed JSON at line %d in %s", line_num, filepath)

    if not entries:
        logger.warning("No valid entries found in %s", filepath)
        return []

    # Group by run_id
    runs: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        run_id = entry.get("run_id", "unknown")
        runs.setdefault(run_id, []).append(entry)

    # Convert each run to TaskData
    tasks: list[TaskData] = []
    for run_id, run_entries in runs.items():
        # Sort by step_id
        run_entries.sort(key=lambda e: e.get("step_id", 0))
        task = _entries_to_task(run_id, run_entries)
        tasks.append(task)

    logger.info("Loaded %d runs from %s (%d total entries)", len(tasks), filepath, len(entries))
    return tasks


def _entries_to_task(run_id: str, entries: list[dict[str, Any]]) -> TaskData:
    """Convert a list of trajectory entries for one run into a TaskData."""
    messages: list[Message] = []
    agents: set[str] = set()
    task_description = ""
    final_reward = 0.0
    final_success: bool | None = None

    for entry in entries:
        agent_name = entry.get("agent", "unknown")
        agents.add(agent_name)

        # Extract content from output
        output = entry.get("output", {})
        content = _extract_content(output, entry.get("action", ""))

        # Also include input context for richer analysis
        input_data = entry.get("input", {})
        if not task_description and "problem" in input_data:
            task_description = input_data["problem"]

        # Build message content with action context
        action = entry.get("action", "")
        full_content = f"[{action}] {content}" if action else content

        role = _infer_role_from_metadata(entry)
        msg = Message(
            agent=agent_name,
            content=full_content,
            index=entry.get("step_id", len(messages)),
            role=role,
            timestamp=entry.get("timestamp"),
            metadata={
                "action": action,
                "reward": entry.get("reward", 0.0),
                "terminal": entry.get("terminal", False),
                "raw_input": input_data,
                "raw_output": output,
                "agent_role": entry.get("metadata", {}).get("agent_role", ""),
                "round_idx": entry.get("metadata", {}).get("round_idx", 0),
                "tokens": entry.get("metadata", {}).get("tokens", 0),
            },
        )
        messages.append(msg)

        # Track final state
        if entry.get("terminal", False):
            final_reward = entry.get("reward", 0.0)
            final_success = entry.get("success")

    # Build eval result from terminal entry
    eval_result = EvalResult(
        task_id=run_id,
        passed=bool(final_success),
        score=final_reward,
        metadata={"run_id": run_id},
    )

    return TaskData(
        task_id=run_id,
        run_id=run_id,
        messages=messages,
        patches=[],  # math-debate transcripts don't have code patches
        eval_result=eval_result,
        task_description=task_description,
        agents=sorted(agents),
    )


def _extract_content(output: dict[str, Any], action: str) -> str:
    """Extract human-readable content from an output dict."""
    if not output:
        return ""

    # Priority: solution > feedback > reasoning > score > str repr
    if "solution" in output:
        return str(output["solution"])
    if "feedback" in output:
        return str(output["feedback"])
    if "reasoning" in output:
        return str(output["reasoning"])
    if "score" in output:
        reasoning = output.get("reasoning", "")
        return f"Score: {output['score']}" + (f" - {reasoning}" if reasoning else "")

    # Fallback: join all values
    parts = [f"{k}: {v}" for k, v in output.items()]
    return "; ".join(parts) if parts else ""


def _infer_role_from_metadata(entry: dict[str, Any]) -> AgentRole:
    """Infer AgentRole from entry metadata."""
    agent_role = entry.get("metadata", {}).get("agent_role", "")
    agent_name = entry.get("agent", "")

    # Map solver/verifier/judge to our agent roles
    if "solver_0" in agent_name or "agent_a" in agent_name.lower():
        return AgentRole.AGENT_A
    if "solver_1" in agent_name or "agent_b" in agent_name.lower():
        return AgentRole.AGENT_B
    return AgentRole.UNKNOWN
