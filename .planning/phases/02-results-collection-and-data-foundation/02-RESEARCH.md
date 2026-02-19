# Phase 2: Results Collection and Data Foundation - Research

**Researched:** 2026-02-18
**Domain:** CooperBench eval execution, log normalization, difficulty scoring
**Confidence:** HIGH

## Summary

Phase 2 transforms raw benchmark logs from Phase 1 into a unified data store suitable for downstream analysis (Figures 4, 5, 6). The work has three main subphases: (1) running `cooperbench eval` on all 300 completed runs to produce per-run `eval.json` files with test pass/fail and merge outcomes, (2) collecting all result.json + eval.json + conversation.json files into a single flat JSON array at `data/results.json`, and (3) computing per-pair difficulty scores from additional solo seeds and assigning bucket labels.

A critical blocking issue was discovered: the eval Docker backend (`eval/backends/docker.py`) does NOT clear the image entrypoint when creating sandbox containers. All CooperBench Docker images have `ENTRYPOINT=[/usr/local/bin/runner.sh]`, which causes `command="sleep infinity"` to be passed as arguments to runner.sh rather than as the container command. This makes the container exit immediately, preventing eval from running. The fix is a one-line change: add `entrypoint=""` to the `client.containers.run()` call, matching the pattern used by the agent Docker environment. This must be fixed before any eval can run.

The raw log structure is well-understood: 100 result.json files per setting (300 total), with solo logs containing `solo.patch` + `solo_traj.json`, and coop logs containing per-agent patches, trajectories, conversation.json, and result.json. No eval.json files exist yet (0 of 300). The `cooperbench eval` CLI accepts `--backend docker` and processes runs with configurable concurrency.

**Primary recommendation:** Fix the Docker eval backend entrypoint first, smoke-test one eval, then batch-eval all 300 runs, then build the collection script as pure-Python JSON processing (no pandas needed for this scale).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Evaluation execution
- Check eval backend (eval/backends/docker.py) for entrypoint compatibility BEFORE running -- don't waste time on 300 failed evals
- Concurrency: 10 (cooperbench eval default -- eval is lighter than agent runs, no LLM calls)
- Run all 3 settings (solo, coop-comm, coop-nocomm) in parallel -- 30 concurrent Docker containers
- Error handling: retry once on failure, then record as error. Errors excluded from metrics, count reported separately

#### Success & merge definitions
- Merge outcome mapping (by final outcome):
  - status=clean + strategy=naive -> `merge_clean`
  - status=clean + strategy=union -> `merge_union` (conflicts existed but resolved)
  - status=conflicts or status=error -> `merge_failed`
- Solo mode: merge_outcome = `merge_clean` (implicit -- no merge step, no conflicts by definition)
- Success metric: `both_passed` only (binary, matches paper). No partial credit tracking.
- LimitsExceeded pairs: count as failure (both_passed=false). Include in denominator -- do not exclude.

#### Difficulty score granularity
- Unit: per feature-pair (each of the 100 pairs gets its own difficulty score)
- Formula: d(pair) = 1 - Solo_pass_rate(pair), averaged across multiple solo seeds
- Run 2 additional solo seeds (3 total) to get continuous difficulty values {0, 0.33, 0.67, 1.0}
  - Additional cost: ~$92 (2 x $46)
  - This populates more difficulty buckets for meaningful Figure 4 curves
- 10 equal-width buckets over [0, 1] as specified in roadmap
- Note as known limitation: single-model reproduction yields coarser difficulty than paper's multi-model approach

#### Data store structure
- Format: single JSON file at `data/results.json` (project root)
- Flat array -- one record per feature-pair per setting
- Each record includes: repo, task_id, features, setting, run metadata (cost, steps, duration), eval results (both_passed, merge_outcome), difficulty score, bucket assignment
- Conversation messages embedded directly in coop-comm records (messages array)
- Difficulty scores and bucket assignments stored in-record (not separate file)

### Claude's Discretion
- Exact JSON schema field names and nesting
- How to handle edge cases in eval (e.g., empty patches, missing files)
- Script structure for the collection pipeline
- Whether to use pandas internally or pure JSON processing

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DATA-01 | Normalize upstream log directory structure into unified JSON results store | Log structure fully mapped (see Architecture Patterns). Collection script reads result.json, eval.json, conversation.json from 3 run directories and merges into flat array at data/results.json |
| DATA-02 | Distinguish infrastructure errors from genuine test failures in results | result.json has agent.status field (Submitted/LimitsExceeded/Error) plus infra_error tag from run_cooperbench.sh. Eval errors have separate error field in eval.json. Both propagated to unified store |
| DATA-03 | Track merge outcomes as separate dimension from test outcomes | eval.json merge field has {status, strategy} enabling merge_clean/merge_union/merge_failed classification. Feature1/feature2 test results are independent fields. Cross-product tracking enabled by schema design |
| FIG4-01 | Compute per-task difficulty score d(t) = 1 - Solo(t) in [0,1] | 3 solo seeds (existing + 2 additional runs) give per-pair pass rates. d(pair) = 1 - mean(both_passed across seeds). Stored in each record |
| FIG4-02 | Partition tasks into 10 equal-width buckets over [0,1] | bucket = min(floor(d * 10), 9). Stored as integer 0-9 in each record. With 4 possible difficulty values {0, 0.33, 0.67, 1.0}, expect 4 populated buckets |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python stdlib json | 3.12 | JSON read/write | 300 records at ~1KB each = trivial scale, no external deps needed |
| cooperbench CLI | upstream | Run eval via `cooperbench eval -n NAME --backend docker` | Already installed in venv, handles concurrency and Docker sandbox |
| docker (Python) | 7.x (in venv) | Used by eval backend internally | Already a dependency of cooperbench |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pathlib | stdlib | Path manipulation for log traversal | All file operations |
| statistics | stdlib | mean() for difficulty score averaging | Averaging solo pass rates across seeds |
| argparse | stdlib | CLI for collection/difficulty scripts | If scripts need parameters |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Pure JSON | pandas | Overkill for 300-record flat array; adds dependency; JSON is simpler for downstream consumers |
| Custom eval | cooperbench eval CLI | CLI is battle-tested, handles sandbox lifecycle, concurrent execution; no reason to rewrite |

**Installation:**
```bash
# No additional installation needed -- all dependencies exist in cooperbench venv
source repos/CooperBench/.venv/bin/activate
```

## Architecture Patterns

### Raw Log Directory Structure (Input)
```
repos/CooperBench/logs/
  command-a-solo/
    config.json              # {run_name, model, setting, concurrency}
    summary.json             # {total_tasks, completed, total_cost, results[]}
    solo/
      {repo_name}/           # e.g., dspy_task
        {task_id}/            # e.g., 8394
          f{X}_f{Y}/          # e.g., f3_f4
            result.json       # Agent run metadata
            solo.patch        # Agent's patch
            solo_traj.json    # Agent trajectory
  command-a-coop-comm/
    coop/
      {repo_name}/{task_id}/f{X}_f{Y}/
        result.json           # Agent run metadata (both agents)
        agent{X}.patch        # Agent 1's patch
        agent{Y}.patch        # Agent 2's patch
        agent{X}_traj.json    # Agent 1 trajectory
        agent{Y}_traj.json    # Agent 2 trajectory
        conversation.json     # Message array [{from, to, message, timestamp, feature_id}]
  command-a-coop-nocomm/
    coop/
      {repo_name}/{task_id}/f{X}_f{Y}/
        result.json
        agent{X}.patch
        agent{Y}.patch
        agent{X}_traj.json
        agent{Y}_traj.json
        conversation.json     # Empty array []
```

### Eval Output Structure (Generated by cooperbench eval)
```
# After eval, each f{X}_f{Y} directory gets an eval.json:
  f{X}_f{Y}/
    eval.json                 # {repo, task_id, features, setting, merge, feature1, feature2, both_passed, error, evaluated_at}
```

### Unified Data Store Structure (Output)
```
data/
  results.json               # Flat array of normalized records
```

### Pattern 1: result.json Schema (Solo)
**What:** Raw agent run metadata for solo mode
**Fields verified from actual files:**
```json
{
  "repo": "dspy_task",
  "task_id": 8394,
  "features": [3, 4],
  "setting": "solo",
  "run_id": "278cb14e",
  "run_name": "command-a-solo",
  "agent_framework": "mini_swe_agent",
  "model": "command-a-03-2025",
  "started_at": "2026-02-18T04:19:59.574560",
  "ended_at": "2026-02-18T04:21:23.389507",
  "duration_seconds": 83.814947,
  "agent": {
    "status": "Submitted",       // or "LimitsExceeded" or "Error"
    "cost": 0.1319,
    "steps": 5,
    "patch_lines": 69,
    "error": null
  },
  "total_cost": 0.1319,
  "total_steps": 5,
  "log_dir": "logs/command-a-solo/solo/dspy_task/8394/f3_f4"
}
```

### Pattern 2: result.json Schema (Coop)
**What:** Raw agent run metadata for coop mode
**Fields verified from actual files:**
```json
{
  "repo": "dspy_task",
  "task_id": 8394,
  "features": [3, 4],
  "setting": "coop",
  "run_name": "command-a-coop-comm",
  "agents": {
    "agent1": {
      "feature_id": 3,
      "status": "Submitted",
      "cost": 0.0572,
      "steps": 6,
      "patch_lines": 434
    },
    "agent2": {
      "feature_id": 4,
      "status": "Submitted",
      "cost": 0.1251,
      "steps": 10,
      "patch_lines": 60
    }
  },
  "total_cost": 0.1823,
  "total_steps": 16,
  "messages_sent": 4,
  "log_dir": "logs/command-a-coop-comm/coop/dspy_task/8394/f3_f4"
}
```

### Pattern 3: eval.json Schema (Generated by cooperbench eval)
**What:** Eval results written per-run after test execution in Docker sandbox
**Schema from evaluate.py source code:**
```json
{
  "repo": "dspy_task",
  "task_id": 8394,
  "features": [3, 4],
  "setting": "solo",
  "merge": null,                    // null for solo, object for coop
  "feature1": {
    "passed": true,
    "test_output": "..."
  },
  "feature2": {
    "passed": false,
    "test_output": "..."
  },
  "both_passed": false,
  "error": null,                    // string if eval failed
  "evaluated_at": "2026-02-18T..."
}
```

For coop mode, `merge` contains:
```json
{
  "merge": {
    "status": "clean",              // "clean" or "conflicts" or "error"
    "strategy": "naive",            // "naive" or "union" or null
    "diff": "..."                   // truncated to 5000 chars
  }
}
```

### Pattern 4: conversation.json Schema
**What:** Inter-agent messages for coop-comm mode
```json
[
  {
    "from": "agent1",
    "to": "agent2",
    "message": "I'll be modifying dspy/clients/cache.py...",
    "timestamp": 1771391479.0375905,
    "feature_id": 3
  }
]
```
For coop-nocomm: empty array `[]`.

### Pattern 5: Recommended Unified Record Schema (Claude's Discretion)
**What:** Each record in data/results.json
```json
{
  "repo": "dspy_task",
  "task_id": 8394,
  "features": [3, 4],
  "setting": "solo",
  "seed": 0,
  "run_name": "command-a-solo",
  "model": "command-a-03-2025",
  "started_at": "2026-02-18T04:19:59.574560",
  "duration_seconds": 83.814947,
  "total_cost": 0.1319,
  "total_steps": 5,
  "agent_status": "Submitted",
  "infra_error": false,
  "both_passed": false,
  "feature1_passed": true,
  "feature2_passed": false,
  "merge_outcome": "merge_clean",
  "merge_status": null,
  "merge_strategy": null,
  "eval_error": null,
  "messages": [],
  "messages_count": 0,
  "difficulty": 0.67,
  "bucket": 6
}
```

**Key design decisions for the schema:**
- `seed` field: 0 for original run, 1 and 2 for additional solo seeds. Enables filtering for difficulty computation.
- `agent_status`: Flattened from nested agent/agents objects. For coop, use worst status (if either agent is Error/LimitsExceeded, record that).
- `merge_outcome`: Derived field using the locked decision mapping. Solo always "merge_clean".
- `messages`: Full array from conversation.json for coop-comm; empty for solo/nocomm.
- `difficulty` and `bucket`: Null initially, populated after all solo seeds are collected.
- `infra_error`: Boolean from result.json post-processing tag (run_cooperbench.sh already tags these).

### Pattern 6: Difficulty Score Computation
**What:** d(pair) = 1 - mean(both_passed across solo seeds)
**With 3 seeds:** possible values are {0.0, 0.33, 0.67, 1.0}
**Bucket assignment:** `bucket = min(int(d * 10), 9)` producing bucket 0, 3, 6, or 9

```python
# Pseudocode for difficulty computation
from collections import defaultdict

solo_results = [r for r in records if r["setting"] == "solo"]
pair_key = lambda r: (r["repo"], r["task_id"], tuple(r["features"]))

# Group by pair across seeds
pair_passes = defaultdict(list)
for r in solo_results:
    pair_passes[pair_key(r)].append(r["both_passed"])

# Compute difficulty
for key, passes in pair_passes.items():
    d = 1.0 - sum(passes) / len(passes)
    bucket = min(int(d * 10), 9)
    # Update all records with this pair (all settings, all seeds)
    for r in records:
        if pair_key(r) == key:
            r["difficulty"] = round(d, 4)
            r["bucket"] = bucket
```

### Anti-Patterns to Avoid
- **Loading trajectories into the unified store:** The `*_traj.json` files are large (agent conversation history with LLM). Only `conversation.json` (inter-agent messages) belongs in the store. Trajectories are for debugging, not analysis.
- **Re-implementing eval logic:** Don't write custom merge/test logic. Use `cooperbench eval` which handles branch creation, patch application, merge strategies, and test parsing.
- **Using relative paths for eval:** The `cooperbench eval` command must be run from the CooperBench repo root because it resolves `logs/` and `dataset/` relative to cwd.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Patch merging + test execution | Custom git merge + pytest | `cooperbench eval --backend docker` | Handles 5 test frameworks (pytest, go test, cargo test, jest/vitest, pnpm test), branch setup, naive/union merge strategy, patch sanitization |
| Docker sandbox lifecycle | Custom docker-py container management | `eval/backends/docker.py` (with entrypoint fix) | Handles container creation, exec, cleanup, timeout |
| Run discovery from logs | Custom directory walker | `eval/runs.py discover_runs()` | Handles both new and legacy log structures, subset filtering |
| Test output parsing | Custom regex per language | `eval/sandbox.py _parse_results()` | Already handles pytest, go test, cargo test, jest/vitest output formats |

**Key insight:** The cooperbench eval infrastructure is comprehensive and battle-tested. The only thing that needs building is: (1) the one-line entrypoint fix, (2) the collection script that reads eval outputs, and (3) the difficulty computation.

## Common Pitfalls

### Pitfall 1: Docker Entrypoint Not Cleared (CRITICAL, BLOCKING)
**What goes wrong:** `cooperbench eval --backend docker` creates containers via `DockerBackend.create_sandbox()` which passes `command="sleep infinity"` but does NOT override the image entrypoint. All CooperBench images have `ENTRYPOINT=[/usr/local/bin/runner.sh]`, so Docker runs `runner.sh sleep infinity`. runner.sh tries to find `/patches/sleep`, fails, exits with code 1. The container dies immediately. All subsequent `exec_run` calls fail because the container is not running.
**Why it happens:** The Docker eval backend was written after the Modal backend (which has `.entrypoint([])`). The Docker backend was likely tested with images without entrypoints, or the issue was never caught because Modal was the primary eval backend.
**How to avoid:** Add `entrypoint=""` to `DockerBackend.create_sandbox()` at line 93 of `eval/backends/docker.py`, matching the pattern used in `agents/mini_swe_agent/environments/docker.py` (line 59).
**Warning signs:** Container exits immediately (status "Exited (1)"), eval errors with "container not running" or similar.
**Evidence:** Confirmed by (a) `docker image inspect` showing entrypoint is `[/usr/local/bin/runner.sh]` for all images, (b) reading runner.sh source which exits 1 if first arg is not a valid patch path, (c) the agent Docker environment already has this fix with comment "CooperBench images have ENTRYPOINT=runner.sh, must override".

### Pitfall 2: Eval Must Run from CooperBench Repo Root
**What goes wrong:** `cooperbench eval` resolves `logs/` and `dataset/` relative to the current working directory. If run from the project root (`/mnt/data/terry/home/cooperbench-repro/`), it won't find the logs.
**Why it happens:** The discover_runs function uses `Path("logs") / run_name` (relative path).
**How to avoid:** Always `cd repos/CooperBench` before running eval commands. Or create a wrapper script that handles the cd.
**Warning signs:** "no runs found to evaluate" message.

### Pitfall 3: Coop Setting Name Mismatch
**What goes wrong:** The eval `discover_runs` function looks for settings named `"solo"` and `"coop"`. Our run names are `command-a-coop-comm` and `command-a-coop-nocomm`, but both have subdirectory `coop/`. The `discover_runs` function will find them correctly because it checks for `solo/` and `coop/` subdirectories, not the run name.
**However:** The discover_runs function sets `setting="coop"` for both comm and nocomm runs, because it only checks for directory name `coop/`. The eval code then treats both as coop mode (merges patches), which is correct. But the collection script needs to distinguish comm from nocomm using the `run_name` field, not the `setting` field.
**How to avoid:** In the collection script, map run_name -> setting: `command-a-solo` -> `"solo"`, `command-a-coop-comm` -> `"coop-comm"`, `command-a-coop-nocomm` -> `"coop-nocomm"`.

### Pitfall 4: LimitsExceeded Pairs Still Produce Patches
**What goes wrong:** A pair with `agent.status="LimitsExceeded"` may still have a non-empty solo.patch (665 lines in the observed case). The eval will still test this patch and may even pass. Per user decision, these count as failure (`both_passed=false`) regardless of eval outcome.
**How to avoid:** The collection script should NOT override eval results for LimitsExceeded pairs. Instead, the eval result stands (both_passed may be true or false), but the unified record should flag `agent_status="LimitsExceeded"`. Downstream analysis can then filter or include as desired. Actually, re-reading the decision: "LimitsExceeded pairs: count as failure (both_passed=false). Include in denominator." So the collection script MUST override `both_passed=false` for LimitsExceeded regardless of eval outcome.
**Warning signs:** Difficulty scores that don't match expectations due to LimitsExceeded pairs passing eval.

### Pitfall 5: Additional Solo Seeds Need Identical Pairs
**What goes wrong:** If additional solo runs use different feature pairs than the original, difficulty scores will have missing data for some pairs.
**How to avoid:** Use the same `--subset lite` flag and same configuration. The cooperbench runner is deterministic in pair selection from the subset definition. Use `--force` to rerun even though results exist (with different run names like `command-a-solo-seed1`, `command-a-solo-seed2`).

### Pitfall 6: Empty Patches
**What goes wrong:** Some agents may produce empty patches (0 lines). The eval handles this gracefully -- `_load_patch` returns None for empty patches, and `test_solo`/`test_merged` proceed without applying agent patches (only test patches are applied). The tests will then fail because no feature code was added.
**How to avoid:** No special handling needed. The eval correctly marks these as `both_passed=false`. The collection script just reads the eval result.

### Pitfall 7: Docker Container Cleanup After Failed Evals
**What goes wrong:** If eval crashes or is interrupted, Docker containers may be left running. With 30 concurrent containers, this can exhaust resources.
**Why it happens:** `DockerSandbox.terminate()` is called in finally blocks, but process-level crashes bypass finally.
**How to avoid:** Before running eval, clean up any leftover cooperbench containers: `docker ps -a --filter ancestor=akhatua/cooperbench* -q | xargs -r docker rm -f`. After eval completes, run cleanup again.

## Code Examples

### Fix 1: Docker Eval Backend Entrypoint (CRITICAL)
```python
# File: repos/CooperBench/src/cooperbench/eval/backends/docker.py
# In DockerBackend.create_sandbox(), add entrypoint="" to containers.run():

container = client.containers.run(
    image=image,
    command="sleep infinity",
    entrypoint="",          # <-- ADD THIS LINE
    detach=True,
    working_dir=workdir,
    remove=False,
    stop_signal="SIGTERM",
)
```
Source: Pattern from `agents/mini_swe_agent/environments/docker.py` line 59.

### Running Eval for All 3 Settings
```bash
# Must run from CooperBench repo root
cd /mnt/data/terry/home/cooperbench-repro/repos/CooperBench
source .venv/bin/activate

# Eval solo (100 runs, concurrency 10)
cooperbench eval -n command-a-solo --backend docker -c 10

# Eval coop-comm (100 runs)
cooperbench eval -n command-a-coop-comm --backend docker -c 10

# Eval coop-nocomm (100 runs)
cooperbench eval -n command-a-coop-nocomm --backend docker -c 10
```

### Running Additional Solo Seeds
```bash
cd /mnt/data/terry/home/cooperbench-repro/repos/CooperBench
source .venv/bin/activate

# Seed 1 (different run name, same config)
cooperbench run -n command-a-solo-seed1 \
  -m command-a-03-2025 \
  -a mini_swe_agent \
  -s lite \
  --backend docker \
  -c 4 \
  --setting solo \
  --no-auto-eval

# Seed 2
cooperbench run -n command-a-solo-seed2 \
  -m command-a-03-2025 \
  -a mini_swe_agent \
  -s lite \
  --backend docker \
  -c 4 \
  --setting solo \
  --no-auto-eval

# Then eval both
cooperbench eval -n command-a-solo-seed1 --backend docker -c 10
cooperbench eval -n command-a-solo-seed2 --backend docker -c 10
```

### Collection Script Structure (Claude's Discretion: Script Architecture)
```python
#!/usr/bin/env python3
"""Collect benchmark results into unified data store.

Reads result.json, eval.json, and conversation.json from all run directories,
normalizes into flat records, computes difficulty scores and bucket assignments,
and writes to data/results.json.

Usage:
    python scripts/collect_results.py
    python scripts/collect_results.py --skip-difficulty  # collect without difficulty
"""

import json
from pathlib import Path
from collections import defaultdict
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COOPERBENCH_DIR = PROJECT_ROOT / "repos" / "CooperBench"
LOGS_DIR = COOPERBENCH_DIR / "logs"
OUTPUT_FILE = PROJECT_ROOT / "data" / "results.json"

# Run name -> setting mapping
RUN_SETTINGS = {
    "command-a-solo": "solo",
    "command-a-solo-seed1": "solo",
    "command-a-solo-seed2": "solo",
    "command-a-coop-comm": "coop-comm",
    "command-a-coop-nocomm": "coop-nocomm",
}

# Seed mapping
RUN_SEEDS = {
    "command-a-solo": 0,
    "command-a-solo-seed1": 1,
    "command-a-solo-seed2": 2,
}

def classify_merge_outcome(eval_data: dict, setting: str) -> str:
    """Map eval merge result to merge outcome category."""
    if setting == "solo":
        return "merge_clean"
    merge = eval_data.get("merge")
    if merge is None:
        return "merge_failed"  # No merge data = error
    status = merge.get("status", "error")
    strategy = merge.get("strategy")
    if status == "clean" and strategy == "naive":
        return "merge_clean"
    elif status == "clean" and strategy == "union":
        return "merge_union"
    else:
        return "merge_failed"

def collect_run(run_name: str, setting: str) -> list[dict]:
    """Collect all records from a single run."""
    records = []
    # ... traverse log dirs, read result.json + eval.json + conversation.json
    # ... normalize into unified schema
    return records

def compute_difficulty(records: list[dict]) -> None:
    """Compute per-pair difficulty from solo results across seeds."""
    # Group solo results by (repo, task_id, features)
    pair_key = lambda r: (r["repo"], r["task_id"], tuple(r["features"]))
    solo = [r for r in records if r["setting"] == "solo"]

    pair_passes = defaultdict(list)
    for r in solo:
        pair_passes[pair_key(r)].append(r["both_passed"])

    # Compute d(pair) = 1 - mean(both_passed)
    pair_difficulty = {}
    for key, passes in pair_passes.items():
        d = 1.0 - mean(float(p) for p in passes)
        bucket = min(int(d * 10), 9)
        pair_difficulty[key] = (round(d, 4), bucket)

    # Apply to ALL records (not just solo)
    for r in records:
        key = pair_key(r)
        if key in pair_difficulty:
            r["difficulty"], r["bucket"] = pair_difficulty[key]

def main():
    records = []
    for run_name, setting in RUN_SETTINGS.items():
        run_records = collect_run(run_name, setting)
        records.extend(run_records)

    compute_difficulty(records)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Wrote {len(records)} records to {OUTPUT_FILE}")
```

### Edge Case Handling (Claude's Discretion)
```python
def get_agent_status(result_data: dict, setting: str) -> str:
    """Extract worst agent status from result.json."""
    if setting == "solo":
        return result_data.get("agent", {}).get("status", "Unknown")
    else:
        agents = result_data.get("agents", {})
        statuses = [a.get("status", "Unknown") for a in agents.values()]
        # Worst status wins: Error > LimitsExceeded > Submitted
        priority = {"Error": 0, "LimitsExceeded": 1, "Submitted": 2, "Unknown": 3}
        return min(statuses, key=lambda s: priority.get(s, 99))

def should_override_both_passed(agent_status: str) -> bool:
    """Per user decision: LimitsExceeded -> both_passed=false."""
    return agent_status == "LimitsExceeded"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Modal-only eval | Docker + Modal + GCP backends | Current codebase | Docker backend is available but has entrypoint bug |
| Manual eval result collection | eval.json written per-run by cooperbench eval | Current codebase | Standardized output format |
| Implicit merge via runner.sh ENTRYPOINT | Explicit merge via sandbox.py branch creation | Current codebase | Eval creates branches, applies patches, merges in-container |

**Deprecated/outdated:**
- None relevant. The codebase is actively maintained and the eval infrastructure is current.

## Open Questions

1. **Will additional solo seeds produce deterministic pair assignments?**
   - What we know: The `cooperbench run` command with `--subset lite` uses `load_subset("lite")` which returns a fixed task list. Pair generation within tasks is deterministic (combinatorial from features).
   - What's unclear: Whether the same 100 pairs will be generated, or if there's randomization.
   - Recommendation: Run a dry-run of `cooperbench run -n test-seed --setting solo -s lite --backend docker` and check that it discovers the same 100 pairs. If not, the collection script needs to handle missing pairs.

2. **How long will eval take for 300 runs at concurrency 10?**
   - What we know: Each eval spins up a Docker container, applies patches, runs tests. Test timeouts are 300s. Container setup is ~5-10s.
   - What's unclear: Actual wall-clock time depends on test complexity per task.
   - Recommendation: Start with a single eval to measure per-task time, then estimate total. Budget ~2-4 hours for all 300 evals.

3. **Are there tasks where runner.sh has different behavior that could cause eval failures?**
   - What we know: runner.sh is per-task (different test commands). The eval sandbox.py calls runner.sh internally. Some tasks use Python (pytest), some Go, some TypeScript, some Rust.
   - What's unclear: Whether all 26 tasks' runner.sh scripts work correctly with the eval sandbox approach.
   - Recommendation: After fixing the entrypoint, run eval on 1 task from each repo (12 total) as a smoke test before running all 300.

## Sources

### Primary (HIGH confidence)
- `repos/CooperBench/src/cooperbench/eval/backends/docker.py` -- DockerBackend source, confirmed missing entrypoint override
- `repos/CooperBench/src/cooperbench/eval/evaluate.py` -- evaluate() function, eval.json schema
- `repos/CooperBench/src/cooperbench/eval/sandbox.py` -- test_solo(), test_merged(), merge logic, runner.sh invocation
- `repos/CooperBench/src/cooperbench/eval/runs.py` -- discover_runs() directory traversal logic
- `repos/CooperBench/src/cooperbench/cli.py` -- CLI eval command arguments
- `repos/CooperBench/logs/command-a-solo/solo/dspy_task/8394/f3_f4/result.json` -- Actual solo result.json
- `repos/CooperBench/logs/command-a-coop-comm/coop/dspy_task/8394/f3_f4/result.json` -- Actual coop result.json
- `repos/CooperBench/logs/command-a-coop-comm/coop/dspy_task/8394/f3_f4/conversation.json` -- Actual conversation format
- `repos/CooperBench/logs/command-a-solo/solo/pallets_jinja_task/1621/f5_f10/result.json` -- LimitsExceeded example
- `repos/CooperBench/logs/results-manifest.json` -- Run statistics (100 pairs per setting, 300 total)
- Docker image inspection: `docker image inspect akhatua/cooperbench-dspy:task8394 --format='{{.Config.Entrypoint}}'` -> `[/usr/local/bin/runner.sh]`
- Docker container runner.sh: confirmed via `docker run --rm --entrypoint="" ... cat /usr/local/bin/runner.sh`
- `repos/CooperBench/src/cooperbench/agents/mini_swe_agent/environments/docker.py` line 59 -- Precedent for entrypoint fix

### Secondary (MEDIUM confidence)
- `repos/CooperBench/src/cooperbench/utils.py` -- get_image_name() function for Docker image naming convention
- `scripts/run_cooperbench.sh` -- Orchestrator with infra_error tagging logic

### Tertiary (LOW confidence)
- Eval wall-clock time estimate (~2-4 hours) -- not measured, based on test timeout config and container overhead

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- All tools are already in the codebase, verified by reading source
- Architecture: HIGH -- All schemas verified from actual log files and source code
- Pitfalls: HIGH -- Entrypoint issue confirmed by Docker image inspection, source code reading, and cross-referencing with agent environment code
- Difficulty computation: HIGH -- Formula is simple (1 - mean), bucket assignment is straightforward
- Additional solo seeds: MEDIUM -- Pair determinism not yet verified; cost estimate from manifest ($46/run)

**Research date:** 2026-02-18
**Valid until:** 2026-03-18 (stable -- no external dependencies changing)
