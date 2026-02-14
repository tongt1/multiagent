# Phase 1: Execution Infrastructure - Research

**Researched:** 2026-02-14
**Domain:** CooperBench benchmark execution with Docker sandbox, Cohere Command A, upstream CLI
**Confidence:** MEDIUM (critical agent/backend compatibility issue found; upstream CLI well-understood)

## Summary

The upstream CooperBench CLI (`cooperbench run`) provides a mature execution harness with built-in task discovery, resume support (skip completed tasks via `result.json` detection), parallel execution via `ThreadPoolExecutor`, and inline evaluation. The lite subset defines 26 unique tasks with 100 feature pairs across 12 repositories. 18 of the 26 required Docker images are already present locally; 8 need to be pulled. Docker storage is already configured to use the mounted 1TB disk at `/mnt/data/docker`, providing 876GB of available space.

However, a critical compatibility issue exists: the user decision specifies `openhands_sdk` agent type, but this adapter is **Modal-only** -- it always creates a `ModalSandboxContext` and ignores the `backend` config parameter entirely. Only the `mini_swe_agent` adapter supports `--backend docker`. This must be resolved before execution can proceed. The recommended path: use `mini_swe_agent` agent type instead, which fully supports Docker backend and Cohere models via litellm.

**Primary recommendation:** Use `mini_swe_agent` with `--backend docker` instead of `openhands_sdk`. If `openhands_sdk` is required, a Docker-compatible adapter would need to be written first (significant effort, out of phase scope).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Use the upstream `cooperbench` CLI directly via subprocess -- NOT the custom wrapper in `src/evaluation/cooperbench/pipeline.py`
- Model string: `command-a-03-2025` with `COHERE_API_KEY` configured
- Agent type: `openhands_sdk` (matches paper methodology -- real tool use, file browsing, git)
- API endpoint: staging (`stg.api.cohere.com`) -- current codebase default
- Command pattern: `cooperbench run -m command-a-03-2025 -a openhands_sdk -s lite --setting {solo|coop} [--no-messaging]`
- Pull all missing images from Docker Hub (`docker pull akhatua/cooperbench-*:task{id}`)
- Block benchmark runs until all 26 lite subset images are available locally
- Verify each image after pulling (start container, check workspace exists)
- Use mounted disk for Docker storage -- main disk is 128GB and insufficient for all images
- Run solo setting first (needed for difficulty scores, fastest, validates pipeline)
- Then run coop-with-comm, then coop-no-comm
- Concurrency: 4 tasks in parallel (low parallelism -- manageable resource usage)
- Must support resume: skip already-completed tasks on restart (critical for 300+ task runs)
- Run naming: `command-a-solo`, `command-a-coop-comm`, `command-a-coop-nocomm`
- No budget ceiling -- track costs but don't halt
- Retry policy: 2 retries (3 total attempts) per task on infrastructure failure
- Infrastructure errors (retryable): Docker OOM/crash, API timeouts
- Genuine failures (not retried): clean test pass/fail results
- Task timeout: 10 minutes per task
- Tag results as `pass`, `fail`, or `infra_error` for downstream analysis

### Claude's Discretion
- Upstream CLI installation method (pip install, editable install, or PATH setup)
- Redis configuration for coop mode inter-agent messaging
- Exact orchestrator script structure and progress reporting format
- How to configure Docker to use mounted disk storage

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

## Standard Stack

### Core
| Component | Version/Value | Purpose | Why Standard |
|-----------|---------------|---------|--------------|
| `cooperbench` CLI | From `repos/CooperBench/` (hatch-based, Python 3.12+) | Benchmark runner | Upstream official tool, handles task discovery, execution, evaluation |
| `litellm` | >= 1.0 (dep of cooperbench) | LLM routing | Routes `command-a-03-2025` to `cohere_chat` provider, handles cost tracking |
| `docker` (Python) | >= 7.0 (dep of cooperbench) | Container management | Used by Docker backend for sandbox creation |
| `redis` (Python) | >= 7.0 (dep of cooperbench) | Inter-agent messaging | Mailbox-based messaging for coop mode |
| `rich` | >= 13.0 (dep of cooperbench) | Progress display | Built into cooperbench for output formatting |

### Supporting
| Tool | Version | Purpose | When to Use |
|------|---------|---------|-------------|
| Docker Engine | 29.2.1 (installed) | Container runtime | All benchmark execution and evaluation |
| Redis server | 7+ (via `docker run redis:alpine`) | Message broker | Coop mode only; auto-started by cooperbench |
| `uv` | 0.10.2 (installed) | Python package manager | Installing cooperbench and dependencies |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `openhands_sdk` agent | `mini_swe_agent` agent | `mini_swe_agent` supports Docker; `openhands_sdk` is Modal-only. See Critical Issue section |
| `uv pip install -e` | `pip install -e` | `uv` is faster, already available on system |

**Installation (recommended):**
```bash
cd repos/CooperBench
uv pip install -e ".[dev]" --python $(which python3)
```

## Architecture Patterns

### Upstream CLI Execution Flow

The cooperbench CLI follows this pattern:
```
cooperbench run -n NAME -m MODEL -a AGENT [-s SUBSET] [--setting SETTING] [--backend BACKEND] [-c CONCURRENCY]
    |
    v
tasks.py:discover_tasks(subset="lite")  -->  Reads dataset/subsets/lite.json
    |                                        Returns 100 task dicts [{repo, task_id, features}]
    v
core.py:run()  -->  ThreadPoolExecutor(max_workers=concurrency)
    |               For each task: execute_solo() or execute_coop()
    |               Skips if result.json exists (resume support)
    v
solo.py / coop.py  -->  get_runner(agent_name)  -->  AgentRunner.run()
    |                    Pass config={"backend": "docker"}
    v
Agent adapter  -->  Creates sandbox, runs agent, extracts patch
    |
    v
Result files  -->  logs/{run_name}/{setting}/{repo}/{task_id}/{feature_str}/
                   - result.json (status, cost, duration)
                   - *.patch (agent patches)
                   - *_traj.json (trajectory)
```

### Docker Image Naming Convention
```
akhatua/cooperbench-{repo_clean}:task{task_id}

Where repo_clean = repo_name.replace("_task", "").replace("_", "-")

Examples:
  dspy_task       -> akhatua/cooperbench-dspy:task8394
  go_chi_task     -> akhatua/cooperbench-go-chi:task26
  pillow_task     -> akhatua/cooperbench-pillow:task25
```

Source: `repos/CooperBench/src/cooperbench/utils.py` function `get_image_name()`

### Resume Logic (Built-in)

The upstream CLI already handles resume. From `solo.py` and `coop.py`:
```python
result_file = log_dir / "result.json"
if result_file.exists() and not force:
    with open(result_file) as f:
        prev_result = json.load(f)
    # Re-run if previous result was an error
    if prev_result.get("agent", {}).get("status") != "Error":
        return {"skipped": True, **prev_result}
```

This means:
- Completed tasks (status != "Error") are skipped automatically
- Error tasks are retried on next run
- The `--force` flag overrides this and reruns everything

### Orchestrator Script Structure (Recommended)

```bash
#!/usr/bin/env bash
# run_cooperbench.sh -- orchestrator for all three experimental settings
set -euo pipefail

COMMON_ARGS="-m command-a-03-2025 -a mini_swe_agent -s lite --backend docker -c 4"

# Phase 1: Solo (fastest, validates pipeline)
echo "=== Running solo setting ==="
cooperbench run -n command-a-solo $COMMON_ARGS --setting solo --no-auto-eval

# Phase 2: Coop with communication (default)
echo "=== Running coop with communication ==="
cooperbench run -n command-a-coop-comm $COMMON_ARGS --setting coop --no-auto-eval

# Phase 3: Coop without communication
echo "=== Running coop without communication ==="
cooperbench run -n command-a-coop-nocomm $COMMON_ARGS --setting coop --no-messaging --no-auto-eval

echo "=== All runs complete ==="
```

### Log Directory Structure
```
logs/
  command-a-solo/
    config.json                    # Run metadata
    summary.json                   # Aggregated results
    solo/
      dspy_task/8394/f3_f4/
        result.json                # Task-level result
        solo.patch                 # Agent patch
        solo_traj.json             # Full trajectory
  command-a-coop-comm/
    coop/
      dspy_task/8394/f3_f4/
        result.json
        agent1.patch / agent2.patch
        agent1_traj.json / agent2_traj.json
        conversation.json          # Inter-agent messages
```

### Anti-Patterns to Avoid
- **Don't write a custom task runner:** The upstream CLI handles task discovery, parallel execution, resume, and result saving. Wrapping it in another layer would duplicate all of this.
- **Don't use `src/evaluation/cooperbench/pipeline.py`:** This is the custom wrapper in this repo. The decision is to use upstream directly.
- **Don't run eval inline with generation:** Use `--no-auto-eval` during generation runs. Evaluation should be a separate phase with `cooperbench eval`.

## Critical Issue: Agent/Backend Compatibility

### openhands_sdk is Modal-only (HIGH confidence)

**What:** The `openhands_sdk` adapter (`repos/CooperBench/src/cooperbench/agents/openhands_agent_sdk/adapter.py`) is hardcoded to use Modal for sandbox execution. It always creates a `ModalSandboxContext` regardless of the `backend` config parameter. Key evidence:

1. The adapter imports and uses `modal` at module level (line 13: `import modal`)
2. The `ModalSandboxContext.__enter__` method creates `modal.Sandbox.create()` (line 679)
3. No Docker backend alternative exists anywhere in the `openhands_agent_sdk` directory
4. The `config["backend"]` parameter is never read in the adapter code

**Impact:** Running `cooperbench run --backend docker -a openhands_sdk` will create a Modal sandbox, NOT a Docker sandbox. This contradicts the user decision to use Docker backend.

**mini_swe_agent supports Docker backend (HIGH confidence):**
The `mini_swe_agent` adapter (`repos/CooperBench/src/cooperbench/agents/mini_swe_agent/adapter.py`) explicitly handles `backend == "docker"` (line 74) by creating a `DockerEnvironment` instance.

**Options:**
1. **Use `mini_swe_agent` instead of `openhands_sdk`** -- simplest, fully supported with Docker backend, uses litellm for model routing, supports messaging via Redis. Change `-a openhands_sdk` to `-a mini_swe_agent`.
2. **Use `openhands_sdk` with Modal** -- requires Modal account setup, runs sandboxes in the cloud (not local Docker). Changes the infrastructure story significantly.
3. **Write a Docker-compatible openhands_sdk adapter** -- significant development effort, would need to replace `ModalSandboxContext` with a Docker equivalent. Out of scope for this phase.

**Recommendation:** Option 1 (use `mini_swe_agent`). This is the only option that satisfies both "Docker backend" and "runs locally" constraints without new development work.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Task discovery | Custom dataset parser | `cooperbench` CLI with `-s lite` | Upstream reads `dataset/subsets/lite.json` correctly, handles all filtering |
| Resume support | Task completion tracking | Upstream `result.json` detection | Built into `solo.py`/`coop.py` -- skips completed tasks automatically |
| Docker image naming | Image name generator | `cooperbench.utils.get_image_name()` | Already handles repo_name to image_name conversion |
| Cost tracking | Custom token counter | litellm built-in `completion_cost()` | cooperbench passes cost through from litellm |
| Redis auto-start | Redis management script | `cooperbench.infra.redis.ensure_redis()` | Checks if Redis is running, auto-starts via Docker if not |
| Progress display | Custom progress bar | cooperbench `rich.Progress` | Built into `core.py:_run_with_progress()` |
| Eval sandbox | Docker eval wrapper | `cooperbench eval -n NAME --backend docker` | Upstream handles sandbox creation, patch testing, result saving |

**Key insight:** The upstream cooperbench CLI handles almost everything needed. The orchestrator script is thin -- it just calls `cooperbench run` with the right arguments for each experimental setting.

## Common Pitfalls

### Pitfall 1: openhands_sdk + Docker Backend Mismatch
**What goes wrong:** Running with `-a openhands_sdk --backend docker` silently uses Modal, not Docker
**Why it happens:** The openhands_sdk adapter ignores the backend config and always creates Modal sandboxes
**How to avoid:** Use `-a mini_swe_agent --backend docker` instead, or accept Modal execution
**Warning signs:** Errors about Modal authentication, or unexpected Modal sandbox creation

### Pitfall 2: Working Directory Must Be repos/CooperBench/
**What goes wrong:** `cooperbench run -s lite` fails with "Subset 'lite' not found at dataset/subsets/lite.json"
**Why it happens:** Task discovery uses `Path("dataset")` relative path (line 56 of `tasks.py`)
**How to avoid:** Always `cd repos/CooperBench/` before running cooperbench commands, or symlink the dataset
**Warning signs:** FileNotFoundError mentioning `dataset/subsets/`

### Pitfall 3: Python 3.12+ Requirement
**What goes wrong:** Import errors, syntax errors in cooperbench
**Why it happens:** cooperbench `requires-python = ">=3.12"`, system Python is 3.11.2
**How to avoid:** Use `uv` to create a Python 3.12+ virtual environment: `uv venv --python 3.12`
**Warning signs:** `SyntaxError` on type union syntax `X | Y`, or version mismatch errors

### Pitfall 4: Staging API Endpoint Configuration
**What goes wrong:** API calls go to production Cohere endpoint, which may not have the right permissions
**Why it happens:** litellm uses the default Cohere API URL; staging requires explicit `api_base` configuration
**How to avoid:** Set `COHERE_API_BASE=https://stg.api.cohere.com` or pass `api_base` via litellm config. The `mini_swe_agent` adapter supports `MSWEA_MODEL_API_BASE` env var to inject `api_base` into litellm calls. Alternatively, pass model string as `cohere/command-a-03-2025` and set `COHERE_API_BASE`.
**Warning signs:** 401/403 errors from Cohere API

### Pitfall 5: Docker Image Disk Space
**What goes wrong:** Docker pull fails or fills up root disk
**Why it happens:** Some images are very large (huggingface-datasets:task6252 is 16.5GB, typst:task6554 is 9.7GB)
**How to avoid:** Docker root is already configured at `/mnt/data/docker` (verified in `/etc/docker/daemon.json`). The mounted disk has 876GB free. No action needed.
**Warning signs:** "no space left on device" errors from Docker

### Pitfall 6: Redis Required for Coop Mode
**What goes wrong:** Coop runs crash with Redis connection errors
**Why it happens:** Coop mode uses Redis for inter-agent messaging via `MessagingConnector`
**How to avoid:** Ensure Redis is running before coop runs. The upstream CLI auto-starts Redis via `docker run redis:alpine` if not running (see `infra/redis.py`).
**Warning signs:** `redis.ConnectionError` during coop initialization

### Pitfall 7: Cost Tracking Warnings
**What goes wrong:** Warnings about cost calculation failure for `command-a-03-2025`
**Why it happens:** litellm may not have exact cost data for all models; cooperbench sets `MSWEA_COST_TRACKING=ignore_errors`
**How to avoid:** The env var `MSWEA_COST_TRACKING=ignore_errors` is set in `core.py` (line 37). litellm DOES recognize `command-a-03-2025` with costs: input $2.50/M tokens, output $10.00/M tokens. Should work.
**Warning signs:** RuntimeError about cost calculation (should not occur since env var suppresses it)

### Pitfall 8: result.json Error Status Re-run Behavior
**What goes wrong:** Tasks that genuinely fail (status="Error") are retried on every resume, even if the error is consistent
**Why it happens:** The resume logic intentionally re-runs Error status tasks
**How to avoid:** This is actually desired behavior for infrastructure errors. For genuine failures that should not be retried, the status will be "Submitted" (not "Error"), so they get skipped correctly.
**Warning signs:** Same task failing repeatedly on every resume run

## Code Examples

### Installing cooperbench (Recommended: editable install with uv)
```bash
# Create Python 3.12 venv (system Python is 3.11)
cd /home/terry_tong_cohere_com/cooperbench-repro/repos/CooperBench
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install cooperbench in editable mode
uv pip install -e ".[dev]"

# Verify
cooperbench --help
```
Source: `repos/CooperBench/pyproject.toml` and `README.md`

### Pulling Missing Docker Images
```bash
#!/usr/bin/env bash
# pull_missing_images.sh
set -euo pipefail

MISSING_IMAGES=(
    "akhatua/cooperbench-dottxt-ai-outlines:task1655"
    "akhatua/cooperbench-dottxt-ai-outlines:task1706"
    "akhatua/cooperbench-dspy:task8587"
    "akhatua/cooperbench-dspy:task8635"
    "akhatua/cooperbench-go-chi:task27"
    "akhatua/cooperbench-llama-index:task17244"
    "akhatua/cooperbench-react-hook-form:task85"
    "akhatua/cooperbench-react-hook-form:task153"
)

for img in "${MISSING_IMAGES[@]}"; do
    echo "Pulling $img..."
    docker pull "$img"
    # Verify: start container, check /workspace/repo exists
    container_id=$(docker run -d "$img" sleep 5)
    if docker exec "$container_id" test -d /workspace/repo; then
        echo "  OK: /workspace/repo exists"
    else
        echo "  WARNING: /workspace/repo not found in $img"
    fi
    docker rm -f "$container_id" > /dev/null
done

echo "All images pulled and verified."
```
Source: Image naming from `repos/CooperBench/src/cooperbench/utils.py:get_image_name()`

### Setting Up Environment Variables
```bash
# Required for Cohere model access
export COHERE_API_KEY="your_key_here"

# Staging endpoint (if required)
# For mini_swe_agent: litellm uses MSWEA_MODEL_API_BASE internally
export MSWEA_MODEL_API_BASE="https://stg.api.cohere.com/v2"

# Suppress cost tracking warnings
export MSWEA_COST_TRACKING="ignore_errors"

# Suppress litellm debug noise
export LITELLM_LOG="ERROR"
```

### Running the Benchmark
```bash
cd /home/terry_tong_cohere_com/cooperbench-repro/repos/CooperBench

# Solo mode (100 pairs, 4 parallel)
cooperbench run \
    -n command-a-solo \
    -m command-a-03-2025 \
    -a mini_swe_agent \
    -s lite \
    --setting solo \
    --backend docker \
    -c 4 \
    --no-auto-eval

# Coop with communication (100 pairs, 4 parallel)
cooperbench run \
    -n command-a-coop-comm \
    -m command-a-03-2025 \
    -a mini_swe_agent \
    -s lite \
    --setting coop \
    --backend docker \
    -c 4 \
    --no-auto-eval

# Coop without communication (100 pairs, 4 parallel)
cooperbench run \
    -n command-a-coop-nocomm \
    -m command-a-03-2025 \
    -a mini_swe_agent \
    -s lite \
    --setting coop \
    --no-messaging \
    --backend docker \
    -c 4 \
    --no-auto-eval
```
Source: `repos/CooperBench/src/cooperbench/cli.py` CLI argument definitions

### Evaluating Results
```bash
# Evaluate solo run
cooperbench eval -n command-a-solo --backend docker

# Evaluate coop runs
cooperbench eval -n command-a-coop-comm --backend docker
cooperbench eval -n command-a-coop-nocomm --backend docker
```
Source: `repos/CooperBench/src/cooperbench/eval/evaluate.py`

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Modal-only execution | Docker + GCP backends added | Recent (pyproject.toml has all 3) | Docker backend is fully supported for both agents and eval |
| No auto-eval | `--no-auto-eval` flag (default: auto-eval on) | Recent | Inline evaluation runs after each task |
| Manual run naming | Auto-generated names (`_generate_run_name`) | Recent | Names like `solo-msa-command-a-03-2025-lite` generated if `-n` not provided |
| No subset support | `dataset/subsets/lite.json` | Recent | 100-pair subset for quick evaluation |

**Deprecated/outdated:**
- The `src/evaluation/cooperbench/pipeline.py` custom wrapper -- decision is to NOT use this

## Environment Status (Verified)

### Docker Configuration
- Docker Engine 29.2.1 installed and running
- Storage root: `/mnt/data/docker` (configured in `/etc/docker/daemon.json`)
- Mounted disk: 1TB at `/mnt/data`, 876GB available
- Root disk: 128GB at `/`, 19GB available (insufficient for Docker images -- correctly redirected)

### Docker Images Status
- **18 of 26** lite subset images present locally
- **8 missing** images need to be pulled:
  - `akhatua/cooperbench-dottxt-ai-outlines:task1655`
  - `akhatua/cooperbench-dottxt-ai-outlines:task1706`
  - `akhatua/cooperbench-dspy:task8587`
  - `akhatua/cooperbench-dspy:task8635`
  - `akhatua/cooperbench-go-chi:task27`
  - `akhatua/cooperbench-llama-index:task17244`
  - `akhatua/cooperbench-react-hook-form:task85`
  - `akhatua/cooperbench-react-hook-form:task153`
- Largest existing image: `huggingface-datasets:task6252` at 16.5GB

### Python Environment
- System Python: 3.11.2 (INSUFFICIENT -- cooperbench requires 3.12+)
- `uv` 0.10.2 installed (can create Python 3.12 venvs)
- cooperbench NOT yet installed as CLI command

### litellm Model Support
- `command-a-03-2025` is recognized by litellm
- Provider: `cohere_chat`
- Supports function calling: yes
- Supports tool choice: yes
- Input cost: $2.50/M tokens, Output cost: $10.00/M tokens
- Max output tokens: 8,000
- Max input tokens: 256,000

## Open Questions

1. **Agent type resolution (BLOCKING)**
   - What we know: `openhands_sdk` is Modal-only; `mini_swe_agent` supports Docker
   - What's unclear: Whether the user will accept `mini_swe_agent` as a substitute
   - Recommendation: Surface this to user. If `openhands_sdk` is truly required, we need Modal setup (different infrastructure story) or a new Docker-compatible adapter (significant development effort). **Most likely the user should switch to `mini_swe_agent`**.

2. **Staging API endpoint configuration**
   - What we know: The `mini_swe_agent` adapter injects `api_base` via `MSWEA_MODEL_API_BASE` env var into litellm calls. litellm routes `command-a-03-2025` to `cohere_chat` provider.
   - What's unclear: Whether the staging endpoint URL is `https://stg.api.cohere.com/v2` or `https://stg.api.cohere.com/v2/chat` for litellm's cohere_chat provider
   - Recommendation: Test with a single task first. If staging is not needed (production API works), this is a non-issue.

3. **Retry policy implementation**
   - What we know: The upstream CLI re-runs tasks with `status="Error"` on resume. The `--force` flag forces re-run of everything.
   - What's unclear: Whether the upstream retry covers Docker OOM/crash vs API timeouts differently. The upstream does not have a built-in "2 retries per task" counter.
   - Recommendation: Rely on the upstream resume behavior (re-runs Error tasks). For the "3 total attempts" requirement, the orchestrator can re-run the `cooperbench run` command up to 3 times -- each time it skips completed tasks and retries failed ones. A task that fails 3 times stays as Error/infra_error.

4. **infra_error tagging**
   - What we know: Upstream tags tasks as `Submitted` (success), `Error` (failure), or `LimitsExceeded` (cost limit)
   - What's unclear: How to distinguish infrastructure errors from genuine test failures
   - Recommendation: Post-processing step: tasks with `status="Error"` where the error message matches Docker/API patterns can be retagged as `infra_error`. Not critical for execution, only for downstream analysis.

## Sources

### Primary (HIGH confidence)
- `repos/CooperBench/pyproject.toml` -- package definition, dependencies, entry points
- `repos/CooperBench/src/cooperbench/cli.py` -- CLI argument definitions and routing
- `repos/CooperBench/src/cooperbench/runner/core.py` -- main execution loop
- `repos/CooperBench/src/cooperbench/runner/solo.py` -- solo mode execution with resume logic
- `repos/CooperBench/src/cooperbench/runner/coop.py` -- coop mode execution with Redis messaging
- `repos/CooperBench/src/cooperbench/runner/tasks.py` -- task discovery from dataset/subsets/
- `repos/CooperBench/src/cooperbench/eval/backends/docker.py` -- Docker eval backend
- `repos/CooperBench/src/cooperbench/agents/__init__.py` -- agent registry and interface
- `repos/CooperBench/src/cooperbench/agents/mini_swe_agent/adapter.py` -- mini_swe_agent Docker support
- `repos/CooperBench/src/cooperbench/agents/openhands_agent_sdk/adapter.py` -- openhands_sdk Modal-only evidence
- `repos/CooperBench/src/cooperbench/utils.py` -- image naming convention, cost tracking
- `repos/CooperBench/src/cooperbench/infra/redis.py` -- Redis auto-start
- `repos/CooperBench/dataset/subsets/lite.json` -- lite subset definition (26 tasks, 100 pairs)
- `/etc/docker/daemon.json` -- Docker root dir configuration
- System commands: `docker images`, `df -h`, `lsblk` -- actual image and disk status

### Secondary (MEDIUM confidence)
- litellm Python API inspection -- `command-a-03-2025` model info and routing
- [litellm Cohere docs](https://docs.litellm.ai/docs/providers/cohere) -- Provider configuration
- [litellm API key docs](https://docs.litellm.ai/docs/set_keys) -- api_base configuration patterns

### Tertiary (LOW confidence)
- Staging API endpoint URL format -- inferred from `cooperbench-eval/src/llm_judge/base.py` which uses `https://stg.api.cohere.com/v2/chat`. The litellm cohere_chat provider may need a different base URL format.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all components verified in codebase, versions confirmed
- Architecture: HIGH -- all execution paths traced through source code
- Agent/backend compatibility: HIGH -- definitively confirmed openhands_sdk is Modal-only
- API endpoint config: MEDIUM -- env var mechanism verified but exact URL format needs testing
- Retry policy: MEDIUM -- upstream resume behavior understood, but custom retry counter is not built-in

**Research date:** 2026-02-14
**Valid until:** 2026-03-14 (upstream CooperBench repo may evolve)
