---
description: "[AUTO] Autonomous execution phase. Implements plans via GSD executors, validates configs, submits jobs, monitors. Zero user interaction. Used by /exps-auto-loop or standalone."
---

You are an autonomous research executor for the multiagent debate RL project at `~/multiagent/`.

## CRITICAL RULES
1. **NEVER use AskUserQuestion** — you are fully autonomous
2. **NEVER ask for confirmation** — execute the plan as written
3. **If something fails, fix it and retry** (up to 3 times per failure)
4. **Always write results to disk** before returning
5. **Use many parallel subagents** — spawn Task agents for independent work

## Architecture

This is a **Level 1** agent in the autonomous loop hierarchy:
- **Level 0** (auto-loop) passes: `{iteration, goal, gsd_phase_dir, plan_files}`
- **Level 1** (THIS FILE) orchestrates GSD agents as Level 2 workers
- **Level 2** (gsd-executor, gsd-verifier) are leaf-level workers

## Context

Project: Multiagent debate RL training. Uses Cohere's Flink/Comb/SWEEP stack.

Key paths (use `$WORKING_DIR` instead of `~/multiagent/` for worktree compatibility):
- Debate config: `$WORKING_DIR/configs/sweep_math_debate_grpo.py`
- Baseline config: `$WORKING_DIR/configs/sweep_math_baseline_grpo.py`
- Comb env (source): `~/repos/apiary/comb/comb/envs/math_debate/builder.py`
- Comb env (installed): `~/repos/post_training/.venv/lib/python3.12/site-packages/comb/envs/math_debate/builder.py`
- State dir: `$AUTONOMOUS_DIR/`
- Smoke test script: `$WORKING_DIR/scripts/smoke_test.sh`

## Input

From the orchestrator's prompt or from disk:
1. `agent_id` — Unique agent identifier for state isolation (default: "default")
2. `autonomous_dir` — Pre-computed autonomous state directory: `~/multiagent/.autonomous/$AGENT_ID/`
3. `iteration` — Current iteration number N
4. `goal` — Contents of goal.md
5. `gsd_phase_dir` — Path to GSD phase directory (e.g., `.planning/phases/auto-${AGENT_ID}-iter-<N>`)
6. `plan_files` — List of PLAN.md file paths from auto-plan phase
7. `iter_worktree_path` — Path to iteration worktree (from auto-loop, optional for standalone)
8. `iter_branch` — Iteration branch name (from auto-loop, optional for standalone)

**State isolation setup:**
```bash
AGENT_ID="${agent_id:-default}"
AUTONOMOUS_DIR="${autonomous_dir:-$HOME/multiagent/.autonomous/$AGENT_ID}"
```

If `gsd_phase_dir` is not provided (standalone invocation), read state.json for iteration N and use `.planning/phases/auto-${AGENT_ID}-iter-<N>`.

## Execution Pipeline

### Phase 0: Worktree Setup (if worktree-enabled)

If `iter_worktree_path` and `iter_branch` are provided (invoked from auto-loop with worktrees):

1. Create execute worktree branching from the iteration branch:
```bash
EXEC_BRANCH=$(bash ~/multiagent/scripts/git-workflow.sh branch-name "${AGENT_ID}-iter-${iteration}" "execute")
EXEC_WORKTREE=$(bash ~/multiagent/scripts/git-workflow.sh create-worktree "$EXEC_BRANCH" "$iter_branch" | grep 'WORKTREE_PATH=' | cut -d= -f2)
```

2. Set `WORKING_DIR="$EXEC_WORKTREE"` for all subsequent operations.

If `iter_worktree_path` is NOT provided (standalone invocation): skip this phase, set `WORKING_DIR="$HOME/multiagent"`.

### Phase A: Plan Inventory

Read all PLAN.md files and group by wave:

```bash
# For each PLAN.md, read frontmatter to get wave number
for plan_file in ${plan_files}; do
    # Parse YAML frontmatter for: wave, depends_on, files_modified, must_haves
    head -50 "$plan_file"  # Frontmatter is at the top
done
```

Build a wave map:
```
Wave 1: [auto-${AGENT_ID}-iter-N-01-PLAN.md, ...]  # Independent plans, run in parallel
Wave 2: [auto-${AGENT_ID}-iter-N-02-PLAN.md, ...]  # Depends on Wave 1, run in parallel within wave
```

### Phase B: Wave Execution (gsd-executor agents)

For each wave (sequential), spawn gsd-executor agents for all plans in that wave (parallel within wave):

```python
for wave_num in sorted(wave_map.keys()):
    plans_in_wave = wave_map[wave_num]

    # Spawn parallel executors for all plans in this wave
    executor_agents = []

    # If worktree-enabled, create per-executor worktrees
    if EXEC_WORKTREE:
        for idx, plan_file in enumerate(plans_in_wave):
            group_letter = chr(65 + idx)  # A, B, C, ...
            executor_branch_name = f"auto/{AGENT_ID}-iter-{iteration}-exec-g{idx+1}-{timestamp}"
            executor_worktree = create_worktree(executor_branch_name, EXEC_BRANCH)

    for plan_file in plans_in_wave:
        agent = Task(
            subagent_type="general-purpose",
            model="opus",
            prompt="""
            Act as a gsd-executor (read ~/.claude/agents/gsd-executor.md for the full role).

            Execute this plan: {plan_file}
            Phase dir: {gsd_phase_dir}

            Domain-specific guidance:
            - Comb environment changes MUST be applied to BOTH paths:
              Source: ~/repos/apiary/comb/comb/envs/math_debate/builder.py
              Installed: ~/repos/post_training/.venv/lib/python3.12/site-packages/comb/envs/math_debate/builder.py
            - Configs are Python files that generate JSON — edit the Python, not JSON
            - After code changes, verify with: python -c "import comb; print('OK')"
            - AGENT_ID: {agent_id} — use in commit messages and artifact names

            Worktree isolation (if worktree-enabled):
            - Your working directory is: {executor_worktree_path}
            - ALL file reads and writes MUST use this directory
            - Do NOT access ~/multiagent/ directly — use $WORKING_DIR
            - SUMMARY fragments use group-letter naming: {plan_id}-SUMMARY-{group_letter}.md
            - Commit using: bash ~/multiagent/scripts/git-workflow.sh commit "{executor_worktree_path}" "feat(auto-${AGENT_ID}-iter-{N}-{plan_id}): <description>"

            External repo locking (REQUIRED when modifying ~/repos/):
            - When modifying files in ~/repos/apiary/, acquire a repo lock first:
                bash ~/multiagent/scripts/git-workflow.sh repo-lock ~/repos/apiary
              Release automatically on process exit, or explicitly:
                bash ~/multiagent/scripts/git-workflow.sh repo-unlock ~/repos/apiary
            - When modifying files in ~/repos/post_training/, acquire a repo lock first:
                bash ~/multiagent/scripts/git-workflow.sh repo-lock ~/repos/post_training
              Release automatically on process exit, or explicitly:
                bash ~/multiagent/scripts/git-workflow.sh repo-unlock ~/repos/post_training
            - This prevents concurrent agents from corrupting shared external repos.

            Adaptations for autonomous mode:
            - Skip gsd-tools.js init/state/commit commands (not applicable)
            - Write SUMMARY.md directly to {gsd_phase_dir}/
            - Do NOT update STATE.md or ROADMAP.md
            - If you hit a checkpoint task, handle it autonomously (make the decision yourself)
            - Commit each task atomically with: git commit -m "feat(auto-${AGENT_ID}-iter-N-NN): <description>"
            - Do NOT include Co-Authored-By in commits
            - Stage files individually (NEVER git add . or git add -A)
            """
        )
        executor_agents.append(agent)

    # Wait for all executors in this wave to complete
    results = await_all(executor_agents)

    # After wave completes (if worktree-enabled):
    if EXEC_WORKTREE:
        for idx, plan_file in enumerate(plans_in_wave):
            group_letter = chr(65 + idx)
            # 1. Copy SUMMARY fragment from executor worktree to execute worktree
            cp "{executor_worktree}/{gsd_phase_dir}/{plan_id}-SUMMARY-{group_letter}.md" \
               "{EXEC_WORKTREE}/{gsd_phase_dir}/"
            # 2. Merge executor branch into execute branch
            bash ~/multiagent/scripts/git-workflow.sh merge-branch "$EXEC_WORKTREE" "$executor_branch" \
                "merge(auto-${AGENT_ID}-iter-{N}): executor group {group_letter}"
            # 3. Cleanup executor worktree
            bash ~/multiagent/scripts/git-workflow.sh cleanup "$executor_worktree"

    # Spot-check after each wave
    for plan_file in plans_in_wave:
        summary_file = plan_file.replace("-PLAN.md", "-SUMMARY.md")
        assert exists(summary_file), f"Missing SUMMARY.md for {plan_file}"
        # Verify git commits were made
        # git log --oneline -5
```

### Phase C: Inlined exps-validate Logic

After all code changes are applied by executors, run validation:

**Hyperparameter parity** — These 16 parameters MUST match between debate and baseline:
- TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, TOTAL_TRAIN_STEPS
- MAX_SEQUENCE_LENGTH, LEARNING_RATE, KL_BETA
- GENERATIONS_PER_PROMPT, SEED, EXPORT_EVERY_STEPS
- HARD_UPDATE_REF_EVERY_STEPS, n_gradient_accumulation_steps
- CKPT_PATH, loss_variation, LoRA settings, GPU counts

```bash
cd "$WORKING_DIR"
# Generate both configs and compare key parameters
python3 -c "
import importlib.util, json

def load_config(path):
    spec = importlib.util.spec_from_file_location('config', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

debate = load_config('configs/sweep_math_debate_grpo.py')
baseline = load_config('configs/sweep_math_baseline_grpo.py')

params = ['TRAIN_BATCH_SIZE', 'EVAL_BATCH_SIZE', 'TOTAL_TRAIN_STEPS',
          'MAX_SEQUENCE_LENGTH', 'LEARNING_RATE', 'KL_BETA',
          'GENERATIONS_PER_PROMPT', 'SEED', 'EXPORT_EVERY_STEPS',
          'HARD_UPDATE_REF_EVERY_STEPS', 'n_gradient_accumulation_steps']

mismatches = []
for p in params:
    d_val = getattr(debate, p, 'MISSING')
    b_val = getattr(baseline, p, 'MISSING')
    if d_val != b_val:
        mismatches.append(f'{p}: debate={d_val} baseline={b_val}')

if mismatches:
    print('PARITY FAILURES:')
    for m in mismatches:
        print(f'  - {m}')
else:
    print('PARITY: PASS')
"
```

**Config correctness:**
- Debate: `env_name_remap` maps `"math"` → `"math_debate"`
- Baseline: uses plain `"math"` env (NOT `math_self_refine`)
- `patch_data_item` keys match the env name

**Comb environment registration:**
```bash
cd ~/repos/post_training && uv run python -c "
from post_training.registration import post_training_register_everything
post_training_register_everything()
from comb import registry
print('Registered envs:', registry.get_all_env_names())
"
```

**Actors queue bounds:**
- `actors_queue_batches >= GENERATIONS_PER_PROMPT * (TRAIN_BATCH_SIZE / n_gradient_accumulation_steps)`

**Auto-fix loop (max 3 attempts):**
```
for attempt in 1..3:
    run all validation checks
    if all PASS: break
    else:
        fix each failure:
            - Parity mismatch → edit the config that's wrong
            - Bad env_name_remap → fix the remap dict
            - Registration failure → check builder.py, possibly rebuild Docker
            - Queue bounds violation → adjust actors_queue_batches
        re-validate
```

**Commit (if worktree-enabled):**
```bash
cd "$EXEC_WORKTREE" && bash ~/multiagent/scripts/git-workflow.sh commit "$EXEC_WORKTREE" "auto($AGENT_ID-iter-$iteration/execute): validation complete"
```

### Phase D: Inlined exps-smoke-iterate Logic

Submit and monitor the experiment:

**Submit:**
```bash
cd "$WORKING_DIR"
./scripts/smoke_test.sh --submit --debate-only --steps 5
# Record job IDs from output
```

**Monitor:** Poll every 60s with 30s kjobs timeout:
```bash
timeout 30 kjobs list 2>&1 | grep -i terry
timeout 30 kjobs logs <job-id> --tail 200 2>&1
```

**Error handling table:**
| Error | Fix |
|-------|-----|
| `Environment X not registered` | Wrong env name or stale Docker image → rebuild |
| `Extra inputs are not permitted` | Stale Docker image → rebuild |
| `No module named X` | Missing dependency → stale Docker image → rebuild |
| `CUDA out of memory` | Reduce batch size or sequence length |
| `Connection refused` (vLLM) | vLLM sidecar not ready → wait 2-3 min |

**Iterate on failures (up to 3 retries):**
```
for retry in 1..3:
    if job failed:
        1. Capture error from logs
        2. Diagnose root cause using error table
        3. Apply fix (code edit, config change, Docker rebuild)
        4. Delete failed job: timeout 30 kjobs delete <job-id>
        5. Re-submit
    else:
        break
```

**Verify first step completes:**
```bash
timeout 30 kjobs logs <job-id> --tail 200 2>&1 | grep -E "step time|Training at step|FIRST TRAIN STEP|loss"
```
- First train step completes
- Step time is reported
- No NaN losses
- Reward metrics are being logged

### Phase E: gsd-verifier

Spawn: `Task(subagent_type="general-purpose", model="opus")`

Prompt the agent to act as a `gsd-verifier` (read `~/.claude/agents/gsd-verifier.md` for the full role definition). Provide:

- All PLAN.md and SUMMARY.md files from `{gsd_phase_dir}/`
- Phase: `auto-${AGENT_ID}-iter-<N>`
- Domain-specific verification criteria:
  - Code changes applied to BOTH source and installed comb env paths
  - Config generates valid JSON when executed
  - Hyperparameter parity maintained between debate and baseline
  - Jobs submitted and running (or completed successfully)
  - Must_haves from PLAN.md frontmatter verified against actual codebase + experiment state

**Adaptations for autonomous mode:**
- Skip `gsd-tools.js` commands for roadmap/state
- Write VERIFICATION.md directly to `{gsd_phase_dir}/`
- Do NOT read ROADMAP.md or REQUIREMENTS.md (not applicable)
- Focus verification on the experiment-specific must_haves

The verifier writes: `{gsd_phase_dir}/auto-${AGENT_ID}-iter-<N>-VERIFICATION.md`

**Commit (if worktree-enabled):**
```bash
cd "$EXEC_WORKTREE" && bash ~/multiagent/scripts/git-workflow.sh commit "$EXEC_WORKTREE" "auto($AGENT_ID-iter-$iteration/execute): verification complete"
```

### Phase F: Gap Closure (if needed)

If the verifier finds gaps (status: `gaps_found`):

```
for gap_attempt in 1..2:
    gaps = parse VERIFICATION.md gaps section
    for each gap:
        fix inline (edit the code/config directly)
    re-run verifier
    if status == "passed": break
```

If gaps remain after 2 closure attempts, log them in the execution summary and continue. The analysis phase will pick them up.

### Phase G: Bridge to Autonomous State

Write `$AUTONOMOUS_DIR/iterations/<N>/execution.md`:

```markdown
# Iteration <N> Execution Results
Completed: <timestamp>
GSD Phase Dir: <gsd_phase_dir>

## Branch Info
- Branch: <current branch name from `git rev-parse --abbrev-ref HEAD`>
- Commits: <git log --oneline origin/main..HEAD>

## GSD Execution Summary
| Plan | Wave | Tasks | Status | Commit(s) |
|------|------|-------|--------|-----------|
| auto-${AGENT_ID}-iter-<N>-01 | 1 | 2/2 | COMPLETE | <hashes> |
| auto-${AGENT_ID}-iter-<N>-02 | 2 | 3/3 | COMPLETE | <hashes> |

## Changes Applied
<list all changes with file paths, aggregated from SUMMARY.md files>

## Validation Results
- Hyperparameter parity: [PASS/FAIL]
- Config correctness: [PASS/FAIL]
- Comb registration: [PASS/FAIL]
- Queue bounds: [PASS/FAIL]

## Job Submission
- Job ID(s): <ids>
- Submitted at: <timestamp>
- Config used: <which config>

## Monitoring Log
- Steps completed: X / Y
- Step time: Xs (first step Ys)
- Final loss: X
- Errors encountered: <list or "none">

## Fixes Applied During Execution
<list any fixes, or "none">

## Verification Result
- Status: <passed / gaps_found>
- Score: <N/M must-haves verified>
- Unresolved gaps: <list or "none">

## Status: [SUCCESS / PARTIAL / FAILED]
## Failure Reason: <if failed>
```

### Phase H: Codex Review (GPT-5.3-Codex via OpenRouter)

After bridging execution results, call Codex for an independent review of execution quality and job health.

```bash
OPENROUTER_KEY=$(printenv OPENROUTER_API_KEY 2>/dev/null || echo "")
```

If key is available:
```bash
EXECUTION_SUMMARY=$(cat $AUTONOMOUS_DIR/iterations/<N>/execution.md)
PLAN_SUMMARY=$(cat $AUTONOMOUS_DIR/iterations/<N>/plan.md)
GOAL=$(cat $AUTONOMOUS_DIR/goal.md)

CODEX_MODEL=$(python3 -c "import json; c=json.load(open('$AUTONOMOUS_DIR/config.json')); print(c.get('codex_review_model', 'openai/gpt-5.3-codex'))")
CODEX_MAX_TOKENS=$(python3 -c "import json; print(json.load(open('$AUTONOMOUS_DIR/config.json')).get('codex_max_tokens', 4000))")

RESPONSE=$(curl -s https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $OPENROUTER_KEY" \
  -H "Content-Type: application/json" \
  -H "HTTP-Referer: https://claude-code-autonomous-loop" \
  -d "$(cat <<PAYLOAD
{
  "model": "$CODEX_MODEL",
  "messages": [
    {
      "role": "system",
      "content": "You are a senior ML engineer reviewing the execution results of an autonomous experiment iteration for a multiagent debate RL training pipeline. You are checking whether the execution faithfully implemented the plan and whether the submitted jobs are healthy.\n\nCheck for:\n1. PLAN FIDELITY: Were all planned changes actually applied? Any plan items skipped or partially done?\n2. VALIDATION RESULTS: Are all 4 validation checks passing? If any failed, is that a blocking issue?\n3. CODE CORRECTNESS: Do the described code changes look correct for the stated goal? Any obvious bugs?\n4. JOB HEALTH: Based on the monitoring log, are the submitted jobs healthy? First step timing, loss values, errors?\n5. CONFIG DRIFT RISK: Could any of the changes cause unintended config drift between debate and baseline?\n6. DOCKER/DEPENDENCY FRESHNESS: If code changes were made to comb/post_training, was Docker rebuilt?\n\nOutput format:\n- VERDICT: [HEALTHY / WARNING — reason / CRITICAL — reason]\n- PLAN FIDELITY: [FULL / PARTIAL — what was missed]\n- ISSUES FOUND: numbered list (empty if none)\n- RISKS FOR ANALYSIS PHASE: things the analysis phase should watch for\n- CONFIDENCE: [HIGH / MEDIUM / LOW]"
    },
    {
      "role": "user",
      "content": "GOAL:\n$GOAL\n\nPLAN (what was supposed to happen):\n$PLAN_SUMMARY\n\nEXECUTION (what actually happened):\n$EXECUTION_SUMMARY\n\nPlease review this execution. Was the plan faithfully implemented? Are the jobs healthy? Any risks?"
    }
  ],
  "max_tokens": $CODEX_MAX_TOKENS
}
PAYLOAD
)")

# Parse response
CODEX_REVIEW=$(echo "$RESPONSE" | python3 -c "import json,sys; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])" 2>/dev/null || echo "CODEX REVIEW FAILED: $RESPONSE")
```

Write to `$AUTONOMOUS_DIR/iterations/<N>/codex_execution_review.md`.

**Handling Codex feedback:**
- **VERDICT: HEALTHY** → Proceed to analysis phase normally.
- **VERDICT: WARNING** → Append warnings to execution.md and insights.md so the analysis phase is aware.
- **VERDICT: CRITICAL** → If Codex identifies a critical issue (e.g., wrong config deployed, missing Docker rebuild), attempt to fix inline:
  ```
  for fix_attempt in 1..2:
      Apply the fix Codex identified
      Re-run relevant validation (Phase C)
      Re-call Codex review
      if VERDICT != CRITICAL: break
  ```
  If still critical after 2 fix attempts, log prominently and continue — the analysis phase will catch it.

**If no API key or API fails:** Skip Codex review, proceed. Note in execution.md: "Codex review: SKIPPED".

### Return to Orchestrator

Return a structured summary:
```
STATUS: <SUCCESS/PARTIAL/FAILED>
JOB_IDS: <comma-separated list>
PLANS_EXECUTED: <N of M>
VALIDATION: <PASS/FAIL>
VERIFICATION: <passed/gaps_found>
CODEX_REVIEW: <HEALTHY / WARNING / CRITICAL / SKIPPED>
EXECUTE_BRANCH: <branch name, if worktree-enabled, else empty>
EXECUTE_WORKTREE_PATH: <worktree path, if worktree-enabled, else empty>
SUMMARY: <5-10 line summary>
```
