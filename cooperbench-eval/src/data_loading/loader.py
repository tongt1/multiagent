"""Data loader for CooperBench log directories.

Expected directory structure:
    <run_dir>/
        <task_id>/
            conversation.jsonl   -- one JSON object per message
            patch_agent_a.diff   -- unified diff from agent A
            patch_agent_b.diff   -- unified diff from agent B
            eval_result.json     -- ground-truth evaluation
            task_description.txt -- problem statement (optional)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from src.data_loading.schemas import (
    AgentRole,
    EvalResult,
    Message,
    PatchInfo,
    TaskData,
)

logger = logging.getLogger(__name__)

# Regex patterns for extracting function definitions from diffs
PYTHON_FUNC_PATTERN = re.compile(r"^\+\s*(?:async\s+)?def\s+(\w+)\s*\(", re.MULTILINE)
JS_FUNC_PATTERN = re.compile(
    r"^\+\s*(?:export\s+)?(?:async\s+)?(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\(|function))",
    re.MULTILINE,
)

# Regex fallback for parsing unified diffs when unidiff fails
DIFF_FILE_PATTERN = re.compile(r"^(?:---|\+\+\+)\s+[ab]/(.+)$", re.MULTILINE)
DIFF_ADDED_PATTERN = re.compile(r"^\+(?!\+\+)(.*)$", re.MULTILINE)
DIFF_REMOVED_PATTERN = re.compile(r"^-(?!--)(.*)$", re.MULTILINE)


def discover_runs(base_dir: str | Path) -> list[Path]:
    """Discover all run directories under a base directory.

    A run directory is any directory containing at least one subdirectory
    with a conversation.jsonl file.

    Args:
        base_dir: Root directory to search for runs.

    Returns:
        Sorted list of run directory paths.
    """
    base = Path(base_dir)
    if not base.exists():
        logger.warning("Base directory does not exist: %s", base)
        return []

    runs: list[Path] = []
    for candidate in sorted(base.iterdir()):
        if not candidate.is_dir():
            continue
        # Check if this directory has task subdirectories with conversation logs
        has_tasks = any(
            (task_dir / "conversation.jsonl").exists()
            for task_dir in candidate.iterdir()
            if task_dir.is_dir()
        )
        if has_tasks:
            runs.append(candidate)

    logger.info("Discovered %d runs under %s", len(runs), base)
    return runs


def load_run(run_dir: str | Path) -> list[TaskData]:
    """Load all tasks from a single run directory.

    Args:
        run_dir: Path to a run directory.

    Returns:
        List of TaskData objects, one per task subdirectory.
    """
    run_path = Path(run_dir)
    if not run_path.exists():
        logger.warning("Run directory does not exist: %s", run_path)
        return []

    tasks: list[TaskData] = []
    for task_dir in sorted(run_path.iterdir()):
        if not task_dir.is_dir():
            continue
        conversation_file = task_dir / "conversation.jsonl"
        if not conversation_file.exists():
            logger.debug("Skipping %s (no conversation.jsonl)", task_dir.name)
            continue
        try:
            task = load_task(task_dir, run_id=run_path.name)
            tasks.append(task)
        except Exception:
            logger.exception("Failed to load task %s", task_dir.name)

    logger.info("Loaded %d tasks from run %s", len(tasks), run_path.name)
    return tasks


def load_task(task_dir: str | Path, run_id: str = "") -> TaskData:
    """Load a single task from its directory.

    Args:
        task_dir: Path to a task directory.
        run_id: Identifier for the parent run.

    Returns:
        Populated TaskData object.
    """
    task_path = Path(task_dir)
    task_id = task_path.name

    # Load conversation
    messages = _load_conversation(task_path / "conversation.jsonl")

    # Discover agents from messages
    agents = sorted({m.agent for m in messages})

    # Load patches for each agent
    patches: list[PatchInfo] = []
    for agent in agents:
        patch_file = task_path / f"patch_{agent}.diff"
        if patch_file.exists():
            patch = _parse_patch(patch_file, agent)
            patches.append(patch)
        else:
            # Try alternative naming conventions
            for alt_name in [f"{agent}.diff", f"{agent}_patch.diff", f"patch_{agent}.patch"]:
                alt_file = task_path / alt_name
                if alt_file.exists():
                    patch = _parse_patch(alt_file, agent)
                    patches.append(patch)
                    break

    # Load eval result
    eval_result = _load_eval_result(task_path, task_id)

    # Load task description
    task_description = _load_text_file(task_path / "task_description.txt")

    return TaskData(
        task_id=task_id,
        run_id=run_id,
        messages=messages,
        patches=patches,
        eval_result=eval_result,
        task_description=task_description,
        agents=agents,
    )


def _load_conversation(filepath: Path) -> list[Message]:
    """Parse a conversation.jsonl file into Message objects."""
    if not filepath.exists():
        logger.warning("Conversation file not found: %s", filepath)
        return []

    messages: list[Message] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                agent = data.get("agent", data.get("role", "unknown"))
                content = data.get("content", data.get("message", ""))
                role = _infer_role(agent)
                msg = Message(
                    agent=agent,
                    content=content,
                    index=idx,
                    role=role,
                    timestamp=data.get("timestamp"),
                    metadata={k: v for k, v in data.items() if k not in ("agent", "role", "content", "message", "timestamp")},
                )
                messages.append(msg)
            except json.JSONDecodeError:
                logger.warning("Malformed JSON at line %d in %s", idx + 1, filepath)

    return messages


def _infer_role(agent_name: str) -> AgentRole:
    """Infer AgentRole from agent name string."""
    name_lower = agent_name.lower()
    if "agent_a" in name_lower or name_lower == "a":
        return AgentRole.AGENT_A
    if "agent_b" in name_lower or name_lower == "b":
        return AgentRole.AGENT_B
    return AgentRole.UNKNOWN


def _parse_patch(filepath: Path, agent: str) -> PatchInfo:
    """Parse a unified diff file into a PatchInfo object.

    Uses the unidiff library for structured parsing, falling back to
    regex-based extraction if unidiff fails (e.g., non-standard diffs).
    """
    raw_diff = filepath.read_text(encoding="utf-8", errors="replace")

    try:
        return _parse_with_unidiff(raw_diff, agent)
    except Exception:
        logger.debug("unidiff parsing failed for %s, using regex fallback", filepath)
        return _parse_with_regex(raw_diff, agent)


def _parse_with_unidiff(raw_diff: str, agent: str) -> PatchInfo:
    """Parse diff using the unidiff library."""
    from unidiff import PatchSet

    patch_set = PatchSet(raw_diff)

    files_modified: list[str] = []
    added_lines: list[str] = []
    removed_lines: list[str] = []
    functions_modified: dict[str, list[str]] = {}

    for patched_file in patch_set:
        file_path = patched_file.path
        files_modified.append(file_path)

        file_funcs: list[str] = []
        for hunk in patched_file:
            for line in hunk:
                if line.is_added:
                    added_lines.append(str(line.value).rstrip("\n"))
                elif line.is_removed:
                    removed_lines.append(str(line.value).rstrip("\n"))

            # Extract function names from hunk content
            hunk_text = str(hunk)
            file_funcs.extend(_extract_functions(hunk_text, file_path))

        if file_funcs:
            functions_modified[file_path] = sorted(set(file_funcs))

    return PatchInfo(
        agent=agent,
        raw_diff=raw_diff,
        files_modified=files_modified,
        added_lines=added_lines,
        removed_lines=removed_lines,
        functions_modified=functions_modified,
    )


def _parse_with_regex(raw_diff: str, agent: str) -> PatchInfo:
    """Fallback regex-based diff parsing."""
    files_modified = sorted(set(DIFF_FILE_PATTERN.findall(raw_diff)))
    added_lines = DIFF_ADDED_PATTERN.findall(raw_diff)
    removed_lines = DIFF_REMOVED_PATTERN.findall(raw_diff)

    functions_modified: dict[str, list[str]] = {}
    for fpath in files_modified:
        funcs = _extract_functions(raw_diff, fpath)
        if funcs:
            functions_modified[fpath] = sorted(set(funcs))

    return PatchInfo(
        agent=agent,
        raw_diff=raw_diff,
        files_modified=files_modified,
        added_lines=added_lines,
        removed_lines=removed_lines,
        functions_modified=functions_modified,
    )


def _extract_functions(diff_text: str, file_path: str) -> list[str]:
    """Extract function/method names from diff text based on file type."""
    funcs: list[str] = []

    if file_path.endswith(".py"):
        for match in PYTHON_FUNC_PATTERN.finditer(diff_text):
            funcs.append(match.group(1))
    elif file_path.endswith((".js", ".ts", ".jsx", ".tsx")):
        for match in JS_FUNC_PATTERN.finditer(diff_text):
            name = match.group(1) or match.group(2)
            if name:
                funcs.append(name)

    return funcs


def _load_eval_result(task_dir: Path, task_id: str) -> EvalResult | None:
    """Load evaluation result from JSON file."""
    eval_file = task_dir / "eval_result.json"
    if not eval_file.exists():
        return None

    try:
        data: dict[str, Any] = json.loads(eval_file.read_text(encoding="utf-8"))
        return EvalResult(
            task_id=task_id,
            passed=data.get("passed", False),
            score=float(data.get("score", 0.0)),
            error_message=data.get("error_message", data.get("error", "")),
            metadata={k: v for k, v in data.items() if k not in ("passed", "score", "error_message", "error")},
        )
    except (json.JSONDecodeError, ValueError):
        logger.warning("Failed to parse eval result for task %s", task_id)
        return None


def _load_text_file(filepath: Path) -> str:
    """Load a text file, returning empty string if not found."""
    if not filepath.exists():
        return ""
    try:
        return filepath.read_text(encoding="utf-8").strip()
    except Exception:
        logger.warning("Failed to read %s", filepath)
        return ""
