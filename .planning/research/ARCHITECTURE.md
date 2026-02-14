# Architecture Research

**Domain:** CooperBench paper reproduction -- benchmark execution + analysis pipeline
**Researched:** 2026-02-14
**Confidence:** HIGH (based on direct codebase inspection of upstream repo, existing wrappers, and cooperbench-eval package)

## Decision: Use Upstream CLI Directly

The existing `src/evaluation/cooperbench/pipeline.py` is a custom LLM-prompt-based wrapper that generates patches via raw API calls and evaluates them with a custom Docker evaluator. This is **not** what the paper uses. The upstream `cooperbench` CLI (`repos/CooperBench/`) runs actual SWE agents (mini_swe_agent, openhands_sdk) inside sandboxed environments with real tool use (file editing, terminal commands, git operations). The upstream CLI handles:

- Agent sandboxing via Docker/Modal/GCP
- Redis-backed inter-agent messaging (coop mode)
- Git server for shared collaboration
- Per-task Docker images (`akhatua/cooperbench-*:task{id}`)
- Standardized evaluation via `runner.sh` + `tests.patch` application

**Use the upstream `cooperbench` CLI for benchmark execution.** The existing wrapper is useful for rapid prototyping but will not produce results comparable to the paper. The existing `cooperbench-eval` package is valuable for post-hoc analysis and should be integrated as the analysis backend.

## System Overview

```
                          ORCHESTRATION LAYER
 ┌──────────────────────────────────────────────────────────────┐
 │                    run_experiment.py                          │
 │  (iterates 3 settings x lite subset, invokes upstream CLI)   │
 └───────────┬──────────────────────────────────┬───────────────┘
             │ subprocess / Python API           │ on completion
             v                                   v
 ┌───────────────────────┐        ┌──────────────────────────────┐
 │   UPSTREAM COOPERBENCH │        │     RESULTS COLLECTOR         │
 │   CLI (cooperbench run)│        │  (gather eval.json, patches,  │
 │                        │        │   trajectories from logs/)     │
 │  - mini_swe_agent      │        └──────────┬───────────────────┘
 │  - Docker sandbox      │                   │
 │  - Redis messaging     │                   v
 │  - cooperbench eval    │        ┌──────────────────────────────┐
 └────────────────────────┘        │     UNIFIED RESULTS STORE     │
                                   │  results/{run_name}/          │
                                   │    summary.json               │
                                   │    per_task/ (eval.json,       │
                                   │      trajectory.json,          │
                                   │      patches)                  │
                                   └──────────┬───────────────────┘
                                              │
                          ANALYSIS LAYER      │
 ┌────────────────────────────────────────────┴─────────────────┐
 │                                                               │
 │  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
 │  │ Difficulty   │  │ Communication│  │ Failure Mode          │ │
 │  │ Scorer       │  │ Analyzer     │  │ Classifier            │ │
 │  │ (Fig 4)      │  │ (Fig 5)      │  │ (Fig 6)               │ │
 │  └──────┬──────┘  └──────┬───────┘  └──────┬───────────────┘ │
 │         │                │                  │                  │
 │         v                v                  v                  │
 │  ┌─────────────────────────────────────────────────────────┐  │
 │  │              FIGURE GENERATOR                            │  │
 │  │  matplotlib-based, one module per figure                 │  │
 │  └─────────────────────────────────────────────────────────┘  │
 │                                                               │
 └───────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | Communicates With | Location |
|-----------|---------------|-------------------|----------|
| **Experiment Orchestrator** | Runs 3 settings (solo, coop-no-comm, coop-with-comm) across lite subset; manages retries, progress, cost tracking | Upstream CLI (subprocess), Results Collector | `scripts/run_experiment.py` (new) |
| **Upstream CooperBench CLI** | Executes agents in sandboxed environments, runs evaluation, writes logs | Docker, Redis, LLM APIs | `repos/CooperBench/` (existing, unmodified) |
| **Results Collector** | Gathers eval.json + trajectory.json + patches from upstream `logs/` into unified store | Upstream log directories, Unified Results Store | `src/analysis/collector.py` (new) |
| **Unified Results Store** | Normalized JSON/JSONL storage of all results across 3 settings | All analyzers read from here | `results/` directory (new) |
| **Difficulty Scorer** | Computes per-task difficulty from gold conflict report + test complexity; buckets tasks into Easy/Medium/Hard; computes Wilson CIs and AUC | Results Store | `src/analysis/difficulty.py` (new) |
| **Communication Analyzer** | Extracts speech acts from agent transcripts; computes Plan:Question ratios; detects first-turn planning; counts specificity tokens; computes merge conflict rates per setting | Results Store | `src/analysis/communication.py` (new) |
| **Failure Mode Classifier** | Runs 10 failure classifiers (4 heuristic + 6 LLM-based) on coop transcripts using CooperBench taxonomy | Results Store, Cohere API | `cooperbench-eval/` (existing, extended) |
| **Figure Generator** | Produces publication-quality matplotlib figures matching paper style | All analyzers | `src/analysis/figures.py` (new) |

## Data Flow

### Phase 1: Benchmark Execution

```
lite.json (100 pairs, 26 tasks, 12 repos)
    |
    v
Orchestrator iterates:
    FOR setting IN [solo, coop --no-messaging, coop]:
        cooperbench run -n {run_name} -s lite \
            -m command-a-03-2025 \
            --setting {setting} \
            [--no-messaging]  \
            --backend docker
    |
    v
Upstream writes to:
    logs/{run_name}/{setting}/{repo}/task{id}/features_{i}_{j}/
        agent{i}/trajectory.json, patch.diff
        agent{j}/trajectory.json, patch.diff  (coop only)
        solo.patch                             (solo only)
        eval.json                              (auto-eval)
```

### Phase 2: Results Collection

```
logs/{run_name}/                 results/{run_name}/
    solo/                   -->      manifest.json
    coop-no-comm/           -->      solo/
    coop/                   -->          {repo}_task{id}_f{i}_f{j}.json
                            -->      coop_no_comm/
                            -->          {repo}_task{id}_f{i}_f{j}.json
                            -->      coop/
                            -->          {repo}_task{id}_f{i}_f{j}.json

Each result JSON:
{
    "repo": str,
    "task_id": int,
    "features": [int, int],
    "setting": str,
    "both_passed": bool,
    "feature1": {"passed": bool, "test_output": str},
    "feature2": {"passed": bool, "test_output": str},
    "merge": {"status": str, "strategy": str} | null,
    "trajectory": [...messages...],
    "patches": [str, ...],
    "cost": float,
    "wall_time": float
}
```

### Phase 3: Analysis

```
results/{run_name}/
    |
    +--> Difficulty Scorer
    |       reads: gold_conflict_report.json + eval results across settings
    |       outputs: difficulty_scores.json
    |           {task_key: {difficulty: float, bucket: "easy"|"medium"|"hard"}}
    |
    +--> Communication Analyzer
    |       reads: coop + coop-no-comm trajectories
    |       outputs: communication_analysis.json
    |           {task_key: {
    |               speech_acts: {plan: N, question: N, answer: N, ...},
    |               plan_question_ratio: float,
    |               first_turn_planning: bool,
    |               specificity_count: int,
    |               merge_conflict: bool
    |           }}
    |
    +--> Failure Mode Classifier
    |       reads: coop trajectories (converted to TaskData)
    |       outputs: failure_classifications.json
    |           (reuses cooperbench-eval FullReport format)
    |
    +--> Figure Generator
            reads: all analysis outputs
            outputs:
                figures/figure4_difficulty_curve.pdf
                figures/figure5_communication.pdf
                figures/figure6_failure_modes.pdf
                figures/qualitative_table.tex
```

## Recommended Project Structure

```
cooperbench-repro/
├── repos/CooperBench/              # Upstream (git submodule, unmodified)
├── cooperbench-eval/               # Existing failure mode analysis package
│   └── src/
│       ├── classifiers/            # 4 heuristic classifiers
│       ├── llm_judge/              # 6 LLM-based classifiers
│       ├── data_loading/           # Schemas + loaders
│       └── report/                 # Report generation + figure
├── src/
│   └── analysis/                   # NEW: reproduction analysis pipeline
│       ├── __init__.py
│       ├── collector.py            # Gather upstream logs -> unified store
│       ├── difficulty.py           # Task difficulty scoring (Fig 4)
│       ├── communication.py        # Speech act / comm analysis (Fig 5)
│       ├── figures.py              # All figure generation (Figs 4-6)
│       └── qualitative.py          # Qualitative analysis helpers
├── scripts/
│   ├── run_experiment.py           # NEW: 3-setting orchestrator
│   ├── collect_results.py          # NEW: post-run results gathering
│   └── generate_figures.py         # NEW: end-to-end figure generation
├── configs/
│   └── cooperbench_default.yaml    # Config (existing, updated)
├── results/                        # NEW: normalized results directory
│   └── {run_name}/
│       ├── manifest.json
│       ├── solo/
│       ├── coop_no_comm/
│       └── coop/
└── figures/                        # NEW: generated figures output
```

### Structure Rationale

- **`repos/CooperBench/`**: Kept as a git submodule. We do not modify it. All interaction is via `cooperbench run` and `cooperbench eval` CLI commands. This ensures our results use the exact same evaluation infrastructure as the paper.
- **`cooperbench-eval/`**: Already implements Figure 6 analysis (failure mode classifiers). We extend it minimally -- the existing `FullReport` + `generate_figure()` + `generate_comparison_figure()` are reusable. We add a bridge in `src/analysis/` that converts upstream log format to `TaskData` schema.
- **`src/analysis/`**: New package for reproduction-specific logic. Deliberately separate from `src/evaluation/cooperbench/` (which is the custom wrapper we are not using for execution).
- **`results/`**: Canonical normalized results. Analyzers read from here, not from raw `logs/`. This decouples analysis from execution format and supports incremental re-runs.
- **`figures/`**: Output directory for publication figures. Keeps generated artifacts out of source tree.

## Architectural Patterns

### Pattern 1: Subprocess Orchestration for Upstream CLI

**What:** The orchestrator invokes `cooperbench run` as a subprocess rather than importing Python internals.
**When to use:** When the upstream tool has complex dependency setup (Modal, Redis, Docker) and its own environment management.
**Trade-offs:** (+) No risk of import-time side effects or version conflicts. (+) Easy to reproduce exact paper methodology. (-) Less fine-grained control. (-) Error reporting is string-based (parse stdout/stderr).

```python
import subprocess
result = subprocess.run(
    ["cooperbench", "run",
     "-n", run_name,
     "-s", "lite",
     "-m", model,
     "--setting", setting,
     "--backend", "docker",
     "--no-auto-eval"],
    cwd="repos/CooperBench",
    capture_output=True, text=True
)
```

### Pattern 2: Adapter for Log Format Conversion

**What:** A thin adapter converts upstream `logs/{run}/` format to `cooperbench-eval` `TaskData` schema.
**When to use:** When two systems need to share data but have different schemas.
**Trade-offs:** (+) Both systems remain independent. (+) Schema changes in either system are isolated. (-) Adapter must be maintained when formats change.

```python
# src/analysis/collector.py
def upstream_logs_to_task_data(log_dir: Path) -> TaskData:
    """Convert upstream cooperbench log directory to cooperbench-eval TaskData."""
    trajectory = json.loads((log_dir / "agent1" / "trajectory.json").read_text())
    messages = [Message(agent=m["agent"], content=m["content"], index=i)
                for i, m in enumerate(trajectory["messages"])]
    patches = [PatchInfo(agent="agent1", raw_diff=(log_dir / "agent1" / "patch.diff").read_text())]
    eval_result = json.loads((log_dir / "eval.json").read_text())
    return TaskData(task_id=..., messages=messages, patches=patches, ...)
```

### Pattern 3: Analysis Module per Figure

**What:** Each paper figure maps to one analysis module with a clear `compute()` -> data and `plot()` -> figure separation.
**When to use:** When figures have distinct data requirements and computation logic.
**Trade-offs:** (+) Each figure can be developed and tested independently. (+) Intermediate data is inspectable. (-) Some shared code (Wilson CI, AUC) needs to live in a utils module.

```python
# src/analysis/difficulty.py
def compute_difficulty_scores(results_dir: Path, conflict_report: Path) -> dict:
    """Compute per-task difficulty. Returns {task_key: {difficulty, bucket}}."""
    ...

# src/analysis/figures.py
def plot_figure4(difficulty_data: dict, results: dict, output_path: Path):
    """Generate Figure 4: success rate by difficulty bucket with Wilson CIs."""
    ...
```

## Data Flow: Key Data Structures

### Upstream Output Format (from `cooperbench run`)

```
logs/{run_name}/{setting}/{repo}/task{id}/features_f{i}_f{j}/
    agent{i}/
        trajectory.json    # Full agent trajectory with messages + tool calls
        patch.diff         # Final generated patch
    agent{j}/              # (coop mode only)
        trajectory.json
        patch.diff
    solo.patch             # (solo mode only)
    eval.json              # Evaluation result
    result.json            # Cost + timing metadata
```

`eval.json` structure:
```json
{
    "repo": "llama_index_task",
    "task_id": 17244,
    "features": [1, 2],
    "setting": "coop",
    "merge": {"status": "clean", "strategy": "union"},
    "feature1": {"passed": true, "test_output": "..."},
    "feature2": {"passed": false, "test_output": "..."},
    "both_passed": false,
    "error": null,
    "evaluated_at": "2026-02-14T..."
}
```

### Unified Results Store Format

```json
{
    "repo": "llama_index_task",
    "task_id": 17244,
    "features": [1, 2],
    "setting": "coop",
    "both_passed": false,
    "feature1_passed": true,
    "feature2_passed": false,
    "merge_status": "clean",
    "merge_conflict": false,
    "transcript": [
        {"agent": "agent1", "content": "...", "step": 0, "action": "..."},
        ...
    ],
    "patches": {"agent1": "diff ...", "agent2": "diff ..."},
    "cost_usd": 0.42,
    "wall_time_s": 180.0
}
```

### Difficulty Score Format (for Figure 4)

```json
{
    "llama_index_task/17244/1_2": {
        "difficulty": 0.73,
        "bucket": "hard",
        "gold_conflict": true,
        "num_test_files": 3,
        "lines_changed": 247
    }
}
```

### Communication Analysis Format (for Figure 5)

```json
{
    "llama_index_task/17244/1_2": {
        "speech_acts": {"plan": 5, "question": 2, "answer": 1, "status": 3, "commit": 1},
        "plan_question_ratio": 2.5,
        "first_turn_is_planning": true,
        "specificity_tokens": 12,
        "total_messages": 14,
        "merge_conflict": false
    }
}
```

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Docker | Upstream CLI manages Docker containers for sandboxed evaluation | Must have Docker daemon running; images `akhatua/cooperbench-*:task{id}` pulled automatically |
| Redis | Upstream CLI uses Redis for inter-agent messaging in coop mode | `docker run -p 6379:6379 redis:7` before coop runs; not needed for solo or coop-no-comm |
| Cohere API | Used by LLM classifiers in `cooperbench-eval` for failure mode detection (Figure 6) | Requires `CO_API_KEY`; uses staging endpoint by default; 6 classifiers need API access |
| LLM API (for agents) | Upstream CLI uses litellm; model specified via `-m` flag | Set appropriate API key env vars (e.g., `CO_API_KEY` for Command A) |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| Orchestrator -> Upstream CLI | subprocess call | CWD must be `repos/CooperBench/`; dataset path is relative to CWD |
| Upstream logs -> Results Store | File I/O via collector | One-time post-run conversion; idempotent |
| Results Store -> Analyzers | JSON file reads | Each analyzer reads normalized JSON; no cross-analyzer dependencies |
| Analyzers -> Figure Generator | Python function calls | Each analyzer returns a dict; figure generator takes dicts and produces plots |
| cooperbench-eval -> Figure 6 | Python import | Convert upstream transcripts to `TaskData` via adapter; run classifiers; use existing `FullReport` |

## Build Order (Dependency Chain)

The components have clear dependencies that dictate implementation order:

### Phase 1: Execution Infrastructure
1. **Experiment Orchestrator** (`scripts/run_experiment.py`)
   - Depends on: upstream CLI being installed (`pip install -e repos/CooperBench`)
   - Produces: raw logs in `logs/`
   - Build first because everything downstream depends on having results

2. **Results Collector** (`src/analysis/collector.py`)
   - Depends on: Phase 1.1 (logs exist)
   - Produces: normalized results in `results/`
   - Build second because all analyzers read from this

### Phase 2: Analysis Components (parallelizable)
3. **Difficulty Scorer** (`src/analysis/difficulty.py`)
   - Depends on: Results Store + `dataset/gold_conflict_report.json`
   - Produces: difficulty_scores.json
   - Can be built independently of other analyzers

4. **Communication Analyzer** (`src/analysis/communication.py`)
   - Depends on: Results Store (specifically coop transcripts)
   - Produces: communication_analysis.json
   - Can be built independently

5. **Failure Mode Bridge** (adapter in `src/analysis/collector.py`)
   - Depends on: Results Store + cooperbench-eval package
   - Produces: failure_classifications.json via existing classifiers
   - Depends on cooperbench-eval being installable

### Phase 3: Visualization
6. **Figure Generator** (`src/analysis/figures.py`)
   - Depends on: all Phase 2 outputs
   - Produces: figures/figure4.pdf, figures/figure5.pdf, figures/figure6.pdf
   - Build last

### Phase 4: End-to-End Script
7. **Generate Figures Script** (`scripts/generate_figures.py`)
   - Depends on: all above
   - Single command to reproduce all figures from results

### Dependency Graph

```
[1] Orchestrator
        |
        v
[2] Results Collector
        |
   ┌────┼────────────┐
   v    v             v
[3]  [4]           [5]
Diff  Comm         Failure Bridge
Score Analyzer     (cooperbench-eval)
   |    |             |
   └────┼─────────────┘
        v
[6] Figure Generator
        |
        v
[7] End-to-End Script
```

## Anti-Patterns

### Anti-Pattern 1: Reimplementing Upstream Evaluation

**What people do:** Build custom Docker evaluation, custom patch merging, custom test runners.
**Why it's wrong:** The existing `src/evaluation/cooperbench/evaluator.py` duplicates what the upstream CLI already does, but with a naive merge strategy and without the runner.sh standardization. Results would not be comparable to the paper.
**Do this instead:** Use `cooperbench run` + `cooperbench eval` as-is. The upstream eval is the ground truth.

### Anti-Pattern 2: Coupling Analysis to Execution Format

**What people do:** Have analyzers directly parse raw `logs/` directory structure with hardcoded path patterns.
**Why it's wrong:** If the upstream changes log format (they already have multiple formats -- `result.json` in newer versions), all analyzers break.
**Do this instead:** Use the Results Collector as a normalization layer. Analyzers only depend on the unified results schema.

### Anti-Pattern 3: Monolithic Analysis Script

**What people do:** Put all analysis (difficulty, communication, failure modes, figure generation) in one giant script.
**Why it's wrong:** Cannot iterate on individual figures. Cannot run subset of analysis. Hard to test.
**Do this instead:** One module per analysis dimension. Each module has `compute()` (returns data) and the figure module has `plot_figureN()` (takes data, returns figure). Compose them in the end-to-end script.

### Anti-Pattern 4: Running All 300 Tasks Serially

**What people do:** Run solo, then coop-no-comm, then coop sequentially.
**Why it's wrong:** The upstream CLI already supports concurrency (`-c` flag). With 100 pairs at ~3-5 min each, serial execution would take 5-8 hours per setting.
**Do this instead:** Use `-c 20` (or higher with sufficient GPU/API quota). The three settings can also run sequentially since they share Docker images (already pulled after first run).

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 100 pairs (lite) | Default: `-c 20` concurrency, Docker backend, ~1-2 hours per setting |
| 652 tasks (full) | Increase concurrency (`-c 50`+), consider Modal/GCP backend for parallel sandboxes, budget ~$200-500 in API costs |
| Multiple models | Parameterize model in orchestrator; results store keyed by `{model}_{setting}`; figures show multi-model comparison |

### Resource Budget Estimate (lite subset, single model)

| Resource | Per Setting | Total (3 settings) |
|----------|-------------|---------------------|
| API calls (agent) | ~100-200 calls | ~400-600 calls |
| API cost (Command A) | ~$5-15 | ~$20-50 |
| API cost (LLM classifiers, Fig 6) | ~$2-5 (coop only) | ~$2-5 |
| Wall time (Docker, c=20) | ~1-2 hours | ~4-6 hours |
| Disk (logs + results) | ~500MB | ~1.5GB |

## Sources

- Upstream CooperBench repo: `/home/terry_tong_cohere_com/cooperbench-repro/repos/CooperBench/`
  - CLI: `src/cooperbench/cli.py`
  - Runner: `src/cooperbench/runner/core.py`
  - Eval: `src/cooperbench/eval/evaluate.py`
  - Dataset: `dataset/subsets/lite.json` (100 pairs, 26 tasks, 12 repos)
- Existing cooperbench-eval package: `/home/terry_tong_cohere_com/cooperbench-repro/cooperbench-eval/`
  - 10 classifiers (4 heuristic + 6 LLM): `src/classifiers/__init__.py`
  - Report generator: `src/report/generator.py`
  - Data schemas: `src/data_loading/schemas.py`
- Existing wrapper (not recommended for execution): `/home/terry_tong_cohere_com/cooperbench-repro/src/evaluation/cooperbench/`
- CooperBench paper: arXiv:2601.13295

---
*Architecture research for: CooperBench paper reproduction pipeline*
*Researched: 2026-02-14*
