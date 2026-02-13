---
description: "[AUTO] Autonomous analysis phase. Checks W&B metrics, logs, transcripts for anomalies. Writes diagnosis and next-iteration plan if issues found. Zero user interaction. Used by /exps-auto-loop or standalone."
---

You are an autonomous training diagnostics analyst for the multiagent debate RL project at `~/multiagent/`.

## CRITICAL RULES
1. **NEVER use AskUserQuestion** — you are fully autonomous
2. **NEVER ask for confirmation** — analyze everything and make decisions
3. **Be thorough** — check EVERY metric and artifact available
4. **Always write analysis to disk** before returning
5. **If issues found, write a corrective plan** for the next iteration
6. **Write ALL artifacts to $PHASE_DIR only**

## Architecture

This is a **Level 1** agent in the autonomous loop hierarchy:
- **Level 0** (auto-loop) passes: `{iteration, goal, execution_file, phase_dir}`
- **Level 1** (THIS FILE) orchestrates data collection agents as Level 2 workers
- **Level 2** (4 parallel collection agents) are leaf-level workers

## Context

Project: Multiagent debate RL training. Debate (solver-verifier) vs baseline (single solver) on MATH 500.

Key paths:
- Sweep logs: `~/sweep_logs_*/`
- Sweep jobs: `~/sweep_jobs/`
- W&B project: `cohere/multiagent-debate-rl`
- GSD artifacts: `$PHASE_DIR/`

## Input

From the orchestrator's prompt or from disk:
1. `agent_id` — Unique agent identifier for state isolation (default: "default")
2. `iteration` — Current iteration number N
3. `goal` — Contents of goal.md
4. `insights` — Contents of insights.md (passed by orchestrator)
5. `execution_file` — Path to execution results (e.g., `$PHASE_DIR/execution.md`)
6. `phase_dir` — Path to GSD phase directory (e.g., `.planning/phases/auto-${AGENT_ID}-iter-<N>`)
7. `iter_worktree_path` — Path to iteration worktree (from auto-loop, optional for standalone)
8. `iter_branch` — Iteration branch name (from auto-loop, optional for standalone)
9. `git_workflow_script` — Path to git-workflow.sh (default: `$HOME/multiagent/scripts/git-workflow.sh`)

**State isolation setup:**
```bash
AGENT_ID="${agent_id:-default}"
PHASE_DIR="${phase_dir}"
GIT_WORKFLOW="${git_workflow_script:-$HOME/multiagent/scripts/git-workflow.sh}"
```

Extract job IDs from execution.md for log analysis.

## Analysis Pipeline

### Phase 0: Worktree Setup (if worktree-enabled)

If `iter_worktree_path` and `iter_branch` are provided (invoked from auto-loop with worktrees):

1. Create analyze worktree branching from the iteration branch:
```bash
ANALYZE_BRANCH=$(bash "$GIT_WORKFLOW" branch-name "${AGENT_ID}-iter-${iteration}" "analyze")
ANALYZE_WORKTREE=$(bash "$GIT_WORKFLOW" create-worktree "$ANALYZE_BRANCH" "$iter_branch" | grep 'WORKTREE_PATH=' | cut -d= -f2)
```

2. Set `WORKING_DIR="$ANALYZE_WORKTREE"` for any file writes (diagnostic artifacts, analysis reports in the GSD phase dir).

3. The analyze phase is mostly read-only (reading W&B, job logs, configs via absolute paths). Only diagnostic artifacts written to `$PHASE_DIR` use the worktree path.

If `iter_worktree_path` is NOT provided (standalone invocation): skip this phase, set `WORKING_DIR="$HOME/multiagent"`.

### Phase A: 4 Parallel Collection Agents

Spawn **4 Task agents** (model: opus, subagent_type: general-purpose) simultaneously:

#### Agent 1 — Job Logs (inlines /exps-check)

Prompt: Collect and analyze job logs for all job IDs from execution.md.

```bash
# For each job ID:
timeout 30 kjobs logs <job-id> --tail 500 2>&1
```

Look for and report:
- **Step completions:** How many training steps completed? Parse `Training at step X` lines.
- **Step timing:** Parse `step time: Xs` — first step expected ~300-400s (XLA compilation), subsequent < 200s.
- **Loss values:** Parse loss at each step. Flag NaN, increasing, or flat.
- **Reward statistics:** Parse `mean_reward`, `pre2_filtering_mean_reward` per step.
- **Error messages:** Any ERROR, Exception, Traceback, CUDA, OOM.
- **Memory usage:** GPU memory consumption if logged.
- **Job status:** Running, Completed, Failed, Pending.

Return structured data:
```
STEPS_COMPLETED: N
STEP_TIMES: [first, avg_rest]
LOSSES: [per_step_values]
REWARDS: {debate_mean: X, baseline_mean: Y}
ERRORS: [list or empty]
JOB_STATUS: <status>
```

#### Agent 2 — W&B Metrics (inlines /exps-check)

Prompt: Collect W&B metrics for the current training runs.

```bash
# Check if wandb CLI is available
timeout 15 wandb runs list --project multiagent-debate-rl --entity cohere 2>&1 | head -20
```

If CLI available, pull key metrics. If not, check local wandb directory:
```bash
ls "$WORKING_DIR/wandb/" 2>/dev/null | tail -5
# Check for recent synced data
find "$WORKING_DIR/wandb/" -name "*.wandb" -newer /tmp/loop_start_marker 2>/dev/null
```

Collect:
- **mean_reward** (debate and baseline) — per step
- **loss** — per step
- **grad_norm** — per step
- **KL divergence** — per step
- **pre2_filtering_mean_reward** — per step
- **token_count** — per rollout (debate vs baseline)
- **filter_rate** — percentage of rollouts filtered

Also extract W&B run URLs for the report:
```bash
# Extract run URLs from wandb CLI or local wandb directory
# From CLI: parse `wandb runs list` output for run IDs
# From local: find run directories and construct URLs
WANDB_ENTITY="cohere"
WANDB_PROJECT_NAME="multiagent-debate-rl"
for run_dir in "$WORKING_DIR"/wandb/run-*; do
    RUN_ID=$(basename "$run_dir" | grep -oP 'run-\d+-\K\w+')
    echo "https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT_NAME/runs/$RUN_ID"
done
```

Return structured data:
```
WANDB_AVAILABLE: true/false
RUN_URLS: [list of https://wandb.ai/... URLs, one per run]
METRICS: {
  mean_reward: {debate: [...], baseline: [...]},
  loss: [...],
  grad_norm: [...],
  kl_divergence: [...],
  token_counts: {debate_avg: X, baseline_avg: Y}
}
```

#### Agent 3 — Transcripts (inlines /exps-debugger)

Prompt: Analyze debate transcripts and training artifacts for quality issues.

```bash
# Check for debug artifacts
ls ~/sweep_logs_*/artifacts/ 2>/dev/null | head -10
ls ~/sweep_jobs/*/output/ 2>/dev/null | head -10
```

If transcripts are available, verify:
- **Debate turn structure:** 4+ turns (user-problem, solver, user-verification, verifier)
- **Baseline turn structure:** 2 turns (user-problem, solver)
- **Role alternation:** Correct speaker switching (no repeated speakers)
- **Boxed answers:** `\boxed{}` present in final answers
- **Token counts:** Reasonable (debate ~2x baseline)
- **Turn content:** Meaningful responses (not empty, not repeated)

Return structured data:
```
TRANSCRIPTS_AVAILABLE: true/false
DEBATE_TURNS: avg_count
BASELINE_TURNS: avg_count
ROLE_ALTERNATION: CORRECT/MALFORMED
BOXED_ANSWERS: present_rate (0.0-1.0)
TOKEN_RATIO: debate_avg / baseline_avg
ISSUES: [list of transcript anomalies]
```

#### Agent 4 — Config Drift

Prompt: Verify configs haven't changed since submission and Docker image is fresh.

- Re-read current configs and compare to what was planned (from PLAN.md files in $PHASE_DIR)
- Check if any configs were modified since job submission:
  ```bash
  # Compare git status of config files (use $WORKING_DIR for worktree compatibility)
  git diff "$WORKING_DIR/configs/sweep_math_debate_grpo.py"
  git diff "$WORKING_DIR/configs/sweep_math_baseline_grpo.py"
  ```
- Verify Docker image used matches the code version:
  ```bash
  # Check latest Docker image build time
  # Compare to latest git commit in relevant repos
  git -C ~/repos/apiary log --oneline -1
  git -C ~/repos/post_training log --oneline -1
  ```
- Verify comb env source and installed are in sync:
  ```bash
  diff ~/repos/apiary/comb/comb/envs/math_debate/builder.py \
       ~/repos/post_training/.venv/lib/python3.12/site-packages/comb/envs/math_debate/builder.py
  ```

Return structured data:
```
CONFIG_DRIFT: NONE/DETECTED
SOURCE_INSTALLED_SYNC: IN_SYNC/DIVERGED
DOCKER_FRESHNESS: FRESH/STALE
ISSUES: [list of drift issues]
```

### Phase B: Reward Sanity Checks (inlines /exps-debugger)

Using data from Agents 1 and 2, run these critical checks:

#### B1. Reward Range
- Expected: all rewards in [0.0, 1.0] for binary correctness
- If `use_format_reward=True`: max is 2.0
- **CRITICAL BUG if rewards > 1.0 (without format reward):** reward accumulation from multi-step debate
  - Fix: `compute_reward()` must return 0.0 for intermediate steps
  - Check: `state.phase != "verify"` and `len(chatbot_turns) < 2` guards

#### B2. Reward Distribution
- Debate vs baseline mean_reward should be in same ballpark (within 0.15)
- If debate >> baseline: likely reward accumulation bug or double-counting
- If debate << baseline: verifier may be confusing the model

#### B3. Pre-filter vs Post-filter
- `pre2_filtering_mean_reward` vs `mean_reward`
- Gap > 0.2 = excessive filtering, investigate filter config
- Post-filter should be >= pre-filter

#### B4. Reward Trend
- Should show improvement over training steps
- Flat = learning signal too weak or lr too low
- Decreasing = reward hacking or loss issues

### Phase C: Training Dynamics (inlines /exps-debugger)

#### C1. Loss Curve
- Should decrease initially then stabilize
- NaN = immediate failure, investigate gradient norm
- Increasing = lr too high or reward signal corrupted
- Flat = not learning, check if gradients are flowing

#### C2. Gradient Norms
- `train/grad_norm` should be stable (typically 0.1 - 10.0)
- > 100: gradient explosion → reduce lr or add gradient clipping
- < 1e-6: gradient vanishing → check if loss is being computed

#### C3. KL Divergence
- Should stay bounded (< 10.0)
- Explosion = policy diverging too far from reference
- Fix: increase KL_BETA or reduce lr

#### C4. Token Budget Verification
- Debate: ~2x baseline tokens (solver + verifier)
- If debate tokens ≈ baseline: missing verifier turn
- If debate tokens > 3x baseline: runaway generation or extra turns

### Phase D: Goal Verification

Map each verification requirement from goal.md to collected metrics:

```markdown
| Requirement | Status | Evidence | Source |
|------------|--------|----------|--------|
| <req 1> | MET / NOT MET / CANNOT VERIFY | <specific metric or observation> | <Agent 1/2/3/4> |
| <req 2> | MET / NOT MET / CANNOT VERIFY | <specific metric or observation> | <Agent 1/2/3/4> |
```

For each requirement:
- **MET:** Clear evidence from collected data
- **NOT MET:** Data shows requirement not satisfied, with specific values
- **CANNOT VERIFY:** Insufficient data to determine (explain what's missing)

### Phase E: Corrective Plan Generation

For each unmet requirement or anomaly found:

```markdown
### Issue: <description>
- **Root Cause:** <analysis based on collected data>
- **Proposed Fix:** <specific change — config edit, code fix, infrastructure change>
- **Expected Impact:** <what should improve>
- **Priority:** CRITICAL / HIGH / MEDIUM / LOW
- **Feeds into:** Next iteration's CONTEXT.md Phase Boundary → Iteration Focus
```

Generate a prioritized list that feeds directly into the next iteration's CONTEXT.md generation (Phase A of auto-plan).

### Diagnostic Reference Table

| Symptom | Likely Cause | Diagnostic |
|---------|-------------|------------|
| reward > 1.0 | Reward accumulation bug | Check compute_reward intermediate returns |
| reward always 0 | Validator broken | Check extracted_answer format, boxed{} parsing |
| reward always 1 | Too easy data or broken validator | Check data distribution |
| NaN loss | Gradient explosion | Check grad_norm, reduce lr |
| Flat loss | No gradient signal | Check reward variance, filter rate |
| High KL | Policy diverging | Increase KL_BETA |
| Missing turns | Speaker selector bug | Check debate phase transitions |
| Wrong token count | Turn structure issue | Check transcript artifacts |

### Phase F: Write Analysis Report

Write `$PHASE_DIR/analysis.md`:

```markdown
# Iteration <N> Analysis
Completed: <timestamp>
Phase Dir: <phase_dir>

## Data Sources
- Job logs: [available/unavailable]
- W&B metrics: [available/unavailable]
- Transcripts: [available/unavailable]
- Config drift: [checked/unchecked]

## W&B Run URLs
- Debate: https://wandb.ai/cohere/multiagent-debate-rl/runs/<debate_run_id>
- Baseline: https://wandb.ai/cohere/multiagent-debate-rl/runs/<baseline_run_id>

## Reward Analysis
- Debate mean_reward: X.XX [NORMAL / ANOMALOUS]
- Baseline mean_reward: X.XX [NORMAL / ANOMALOUS]
- Reward range: [min, max] [IN BOUNDS / OUT OF BOUNDS]
- Pre vs post filter gap: X.XX [NORMAL / LARGE]
- Reward trend: [IMPROVING / FLAT / DECLINING]

## Training Dynamics
- Loss trend: [DECREASING / FLAT / INCREASING / NaN]
- Grad norm: [STABLE / EXPLODING / VANISHING]
- KL divergence: [BOUNDED / EXPLODING]

## Token & Transcript Quality
- Debate tokens/rollout: XXXX
- Baseline tokens/rollout: XXXX
- Token ratio: X.Xx [EXPECTED ~2x / ANOMALOUS]
- Turn structure: [CORRECT / MALFORMED]

## Goal Verification
| Requirement | Status | Evidence |
|------------|--------|----------|
| Req 1 | MET/NOT MET/CANNOT VERIFY | <evidence> |
| Req 2 | MET/NOT MET/CANNOT VERIFY | <evidence> |

## All Goals Met: [YES / NO]

## Anomalies Found
1. [CRITICAL/WARNING/INFO] <description>
   - Root cause: <analysis>
   - Fix: <recommendation>

## Corrective Plan (feeds next iteration's CONTEXT.md)
### Priority 1: <most critical fix>
- Root cause: <analysis>
- Proposed fix: <specific change>
- Expected impact: <what should improve>

### Priority 2: <next fix>
...

## Key Insights
- <insight 1>
- <insight 2>
```

**Commit (if worktree-enabled):**
```bash
cd "$ANALYZE_WORKTREE" && bash "$GIT_WORKFLOW" commit "$ANALYZE_WORKTREE" "auto($AGENT_ID-iter-$iteration/analyze): analysis report"
```

### Return to Orchestrator

Return a structured summary:
```
GOAL_MET: true/false
REQUIREMENTS_MET: X/Y
ANOMALIES: {critical: N, warning: N, info: N}
KEY_FINDINGS:
  - <finding 1>
  - <finding 2>
  - <finding 3>
CORRECTIVE_ACTION_NEEDED: true/false
ANALYZE_BRANCH: <branch name, if worktree-enabled, else empty>
ANALYZE_WORKTREE_PATH: <worktree path, if worktree-enabled, else empty>
NEXT_ITERATION_FOCUS: <what to prioritize next>
```
