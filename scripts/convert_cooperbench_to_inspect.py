#!/usr/bin/env python3
"""Convert CooperBench experiment logs to Inspect AI .eval format.

CooperBench logs are structured as individual JSON files per run:
  - *_traj.json: Agent trajectory with system/user/assistant messages
  - result.json: Run metadata, costs, agent status
  - conversation.json: Inter-agent messages (coop modes only)

Usage:
  python scripts/convert_cooperbench_to_inspect.py \
    --log-dir repos/CooperBench/logs/command-a-coop-comm \
    --output-dir /tmp/cooperbench_inspect/

  python scripts/convert_cooperbench_to_inspect.py \
    --log-dir repos/CooperBench/logs/ \
    --output-dir /tmp/cooperbench_inspect/ \
    --setting coop-comm
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalMetric,
    EvalPlan,
    EvalResults,
    EvalScore,
    EvalSpec,
    EvalStats,
    EvalSample,
    write_eval_log,
)
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ModelOutput,
)
from inspect_ai.scorer import Score


def load_run(run_dir: Path) -> dict | None:
    """Load a CooperBench run directory into a unified dict."""
    result_file = run_dir / "result.json"
    if not result_file.exists():
        return None

    with open(result_file) as f:
        result = json.load(f)

    # Load agent trajectories
    trajs = {}
    for traj_file in sorted(run_dir.glob("*_traj.json")):
        with open(traj_file) as f:
            traj = json.load(f)
        agent_id = traj.get("agent_id", traj_file.stem.replace("_traj", ""))
        trajs[agent_id] = traj

    # Load conversation if exists
    conv_file = run_dir / "conversation.json"
    conversation = []
    if conv_file.exists():
        with open(conv_file) as f:
            conversation = json.load(f)

    return {
        "result": result,
        "trajectories": trajs,
        "conversation": conversation,
        "run_dir": str(run_dir),
    }


def build_sample_from_run(run: dict, sample_id: int) -> EvalSample:
    """Convert a CooperBench run into an EvalSample."""
    result = run["result"]
    trajs = run["trajectories"]
    conversation = run["conversation"]

    repo = result.get("repo", "unknown")
    task_id = result.get("task_id", "unknown")
    setting = result.get("setting", "unknown")
    features = result.get("features", [])

    # Build input text
    input_text = f"Repository: {repo}\nTask ID: {task_id}\nFeatures: {features}\nSetting: {setting}"

    # Build messages from all agent trajectories
    messages = []
    for agent_id, traj in sorted(trajs.items()):
        agent_msgs = traj.get("messages", [])
        if not agent_msgs:
            continue

        # Add separator for each agent
        messages.append(
            ChatMessageSystem(
                content=f"--- Agent: {agent_id} (feature {traj.get('feature_id', '?')}) ---"
            )
        )

        for msg in agent_msgs:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if not content:
                continue

            if role == "system":
                messages.append(ChatMessageSystem(content=content))
            elif role == "assistant":
                messages.append(ChatMessageAssistant(content=content))
            else:
                messages.append(ChatMessageUser(content=content))

    # Add inter-agent conversation if present
    if conversation:
        messages.append(
            ChatMessageSystem(content="--- Inter-Agent Communication ---")
        )
        for msg in conversation:
            content = f"[{msg.get('from', '?')} → {msg.get('to', '?')}] {msg.get('message', '')}"
            messages.append(ChatMessageUser(content=content))

    # Build scores from result metadata
    scores = {}
    # Handle both "agents" (coop: dict of agent dicts) and "agent" (solo: single dict)
    agents_info = result.get("agents", {})
    if not agents_info and "agent" in result:
        agents_info = {"solo": result["agent"]}

    # Submission status as a score (1.0 = all submitted, 0.0 = none)
    submitted_count = sum(
        1 for a in agents_info.values() if a.get("status") == "Submitted"
    )
    total_agents = len(agents_info)
    if total_agents > 0:
        scores["submission_rate"] = Score(
            value=submitted_count / total_agents,
            explanation=f"{submitted_count}/{total_agents} agents submitted patches",
        )

    # Total cost as score (for comparison across runs)
    total_cost = result.get("total_cost", 0.0)
    scores["cost_usd"] = Score(
        value=total_cost,
        explanation=f"Total API cost: ${total_cost:.4f}",
    )

    # Total steps
    total_steps = result.get("total_steps", 0)
    scores["total_steps"] = Score(
        value=float(total_steps),
        explanation=f"Total agent steps across all agents",
    )

    # Per-agent patch lines
    total_patch_lines = sum(
        a.get("patch_lines", 0) for a in agents_info.values()
    )
    scores["patch_lines"] = Score(
        value=float(total_patch_lines),
        explanation=f"Total lines of code in patches",
    )

    # Messages sent (for coop modes)
    messages_sent = result.get("messages_sent", 0)
    if messages_sent > 0:
        scores["messages_sent"] = Score(
            value=float(messages_sent),
            explanation=f"Inter-agent messages exchanged",
        )

    # Build target (features list)
    target = f"Features: {features}"

    # Get model name from first trajectory
    model_name = "unknown"
    for traj in trajs.values():
        model_name = traj.get("model", "unknown")
        break

    # Last assistant message as output
    last_assistant = ""
    for msg in reversed(messages):
        if isinstance(msg, ChatMessageAssistant):
            last_assistant = msg.content
            break

    # Build metadata
    metadata = {
        "repo": repo,
        "task_id": task_id,
        "setting": setting,
        "features": features,
        "run_id": result.get("run_id", ""),
        "run_name": result.get("run_name", ""),
        "duration_seconds": result.get("duration_seconds", 0),
        "log_dir": result.get("log_dir", ""),
    }

    # Per-agent details
    for agent_id, agent_info in agents_info.items():
        metadata[f"{agent_id}_status"] = agent_info.get("status", "unknown")
        metadata[f"{agent_id}_cost"] = agent_info.get("cost", 0)
        metadata[f"{agent_id}_steps"] = agent_info.get("steps", 0)

    return EvalSample(
        id=sample_id,
        input=input_text,
        messages=messages,
        output=ModelOutput.from_content(
            model=model_name,
            content=last_assistant[:500] if last_assistant else "No output",
        ),
        scores=scores,
        target=target,
        metadata=metadata,
        epoch=1,
    )


def build_eval_log(
    samples: list[EvalSample],
    task_name: str,
    model_name: str,
    setting: str,
) -> EvalLog:
    """Assemble EvalLog from samples."""
    now = datetime.now(timezone.utc).isoformat()

    spec = EvalSpec(
        task=task_name,
        model=model_name,
        created=now,
        dataset=EvalDataset(name=f"cooperbench-{setting}", samples=len(samples)),
        config=EvalConfig(),
    )

    plan = EvalPlan(name=task_name)

    # Compute aggregate scores
    scorer_values: dict[str, list[float]] = {}
    for s in samples:
        if s.scores:
            for name, score in s.scores.items():
                if score.value is not None:
                    val = float(score.value) if not isinstance(score.value, str) else 0.0
                    scorer_values.setdefault(name, []).append(val)

    eval_scores = []
    for name, values in scorer_values.items():
        mean_val = sum(values) / len(values) if values else 0.0
        eval_scores.append(
            EvalScore(
                name=name,
                scorer=name,
                params={},
                metrics={
                    "mean": EvalMetric(name="mean", value=mean_val),
                    "count": EvalMetric(name="count", value=len(values)),
                },
            )
        )

    results = EvalResults(
        total_samples=len(samples),
        completed_samples=len(samples),
        scores=eval_scores,
    )

    stats = EvalStats(started_at=now, completed_at=now)

    return EvalLog(
        version=2,
        status="success",
        eval=spec,
        plan=plan,
        results=results,
        stats=stats,
        samples=samples,
    )


def discover_runs(log_dir: Path) -> list[Path]:
    """Find all run directories containing result.json."""
    runs = []
    for result_file in sorted(log_dir.rglob("result.json")):
        runs.append(result_file.parent)
    return runs


def main():
    parser = argparse.ArgumentParser(
        description="Convert CooperBench logs to Inspect AI .eval format"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="Directory containing CooperBench experiment logs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for .eval files",
    )
    parser.add_argument(
        "--setting",
        type=str,
        choices=["coop-comm", "coop-nocomm", "solo", "all"],
        default="all",
        help="Filter by experiment setting (default: all)",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="cooperbench_eval",
        help="Task name for EvalSpec",
    )
    args = parser.parse_args()

    if not args.log_dir.exists():
        print(f"Error: {args.log_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all runs
    run_dirs = discover_runs(args.log_dir)
    print(f"Found {len(run_dirs)} runs in {args.log_dir}")

    # Load and filter runs
    runs_by_setting: dict[str, list[dict]] = {}
    for run_dir in run_dirs:
        run = load_run(run_dir)
        if run is None:
            continue
        setting = run["result"].get("setting", "unknown")
        run_name = run["result"].get("run_name", "")

        # Filter by setting if specified
        if args.setting != "all":
            if args.setting not in run_name and setting != args.setting:
                continue

        key = run_name or setting
        runs_by_setting.setdefault(key, []).append(run)

    if not runs_by_setting:
        print("No matching runs found.", file=sys.stderr)
        sys.exit(1)

    # Build one .eval file per setting/run_name
    for setting_key, runs in runs_by_setting.items():
        print(f"\nProcessing {setting_key}: {len(runs)} runs")

        samples = []
        model_name = "unknown"
        for idx, run in enumerate(runs, 1):
            sample = build_sample_from_run(run, sample_id=idx)
            samples.append(sample)
            if model_name == "unknown":
                for traj in run["trajectories"].values():
                    model_name = traj.get("model", "unknown")
                    break

        log = build_eval_log(
            samples,
            task_name=args.task_name,
            model_name=model_name,
            setting=setting_key,
        )

        safe_name = setting_key.replace("/", "_").replace(" ", "_")
        output_path = args.output_dir / f"cooperbench_{safe_name}.eval"
        write_eval_log(log, location=str(output_path), format="eval")
        print(f"  Written: {output_path} ({len(samples)} samples)")

    print(f"\nDone. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
