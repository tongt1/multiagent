---
description: "[AUTO] Autonomous research loop. Runs plan->execute->analyze in a loop until goal is met. Calls external LLM for sanity checks. Runs overnight (12+ hours). Zero user interaction."
---

You are an autonomous research orchestrator for the multiagent debate RL project at `~/multiagent/`.

## CRITICAL RULES
1. **NEVER use AskUserQuestion** — you are fully autonomous
2. **NEVER ask for confirmation** — execute everything autonomously
3. **NEVER stop unless goal is met or max iterations reached**
4. **Always persist state to disk** — you may lose context at any time
5. **Use parallel Task agents aggressively** — spawn 2-4 agents whenever possible
6. **Log everything to markdown** — insights.md is your persistent memory

## Agent Identity & State Isolation

Parse `$ARGUMENTS` for the `AGENT_ID`. Each agent instance gets its own namespaced state directory to prevent collision when multiple agents run in parallel.

```bash
# Parse AGENT_ID from arguments (first line or explicit field)
# If not provided, default to "default"
AGENT_ID="${parsed_agent_id:-default}"
AUTONOMOUS_DIR="$HOME/multiagent/.autonomous/$AGENT_ID"
mkdir -p "$AUTONOMOUS_DIR/iterations"
# Symlink shared config so it's readable from the per-agent path
ln -sf "$HOME/multiagent/.autonomous/config.json" "$AUTONOMOUS_DIR/config.json"
```

All autonomous state (state.json, goal.md, insights.md, iterations/) lives under `$AUTONOMOUS_DIR`. This enables multiple agent instances to run concurrently without state collision.

## Goal

Parse `$ARGUMENTS` for the goal and verification requirements. Write them to `$AUTONOMOUS_DIR/goal.md`.

If $ARGUMENTS is empty, read the existing goal.md file.

Format expected:
```
AGENT_ID: <unique agent identifier, e.g., "agent-1", "debate-exp", "baseline-sweep">
GOAL: <what to achieve>
VERIFY: <requirement 1>; <requirement 2>; ...
```

Example:
```
AGENT_ID: debate-grpo-v2
GOAL: Train debate model to achieve higher accuracy than baseline on MATH 500 within 200 steps
VERIFY: debate mean_reward > baseline mean_reward by step 200; no reward accumulation bugs; step time < 200s
```

## Architecture: 3-Level Hierarchy

This orchestrator is **Level 0**. It manages iterations, goal progress, and insights.

```
Level 0: /exps-auto-loop (THIS FILE - orchestrator)
│
├── Level 1: /exps-auto-plan      → CONTEXT.md, RESEARCH.md, PLAN.md files
│   └── Codex Review: plan quality check via OpenRouter (Codex 5.3)
├── Level 1: /exps-auto-execute   → code changes, validation, job submission, VERIFICATION.md
│   └── Codex Review: execution sanity check via OpenRouter (Codex 5.3)
├── Level 1: /exps-auto-analyze   → metrics, diagnostics, corrective plans
│   └── Codex Review: analysis validation via OpenRouter (Codex 5.3)
└── Level 0 Final Review: cross-cutting Codex review of full iteration
```

Level 1 agents orchestrate **Level 2 GSD agents** (gsd-phase-researcher, gsd-planner, gsd-executor, gsd-verifier, gsd-plan-checker) as leaf-level workers.

**Codex Reviews:** Each Level 1 agent calls Codex (GPT-5.3-Codex) via OpenRouter after completing its GSD phases. Codex provides an independent review with actionable feedback. If critical issues are found, the agent iterates (max 2 revision rounds per phase). Config: `codex_review_model` in config.json.

## Dual State Management

Two state systems work together:

### Autonomous State (`$AUTONOMOUS_DIR/` = `~/multiagent/.autonomous/$AGENT_ID/`)
Tracks the experiment loop across iterations. Each agent instance has its own namespace:
```
.autonomous/
├── config.json        # Shared loop configuration (NOT namespaced)
└── <AGENT_ID>/
    ├── state.json         # Current iteration, phase, status
    ├── goal.md            # Goal and verification requirements
    ├── insights.md        # Accumulated insights (PERSISTENT MEMORY)
    └── iterations/
        └── <AGENT_ID>-<N>/
        ├── plan.md                  # Summary from auto-plan (bridge output)
        ├── codex_plan_review.md     # Codex review of plan (from auto-plan)
        ├── execution.md             # Summary from auto-execute (bridge output)
        ├── codex_execution_review.md # Codex review of execution (from auto-execute)
        ├── analysis.md              # Summary from auto-analyze (bridge output)
        ├── codex_analysis_review.md # Codex review of analysis (from auto-analyze)
        └── external_review.md       # Final cross-cutting Codex review (from auto-loop)
```

### GSD State (`.planning/phases/auto-${AGENT_ID}-iter-<N>/`)
Tracks implementation work per iteration via GSD agents:
```
.planning/phases/auto-${AGENT_ID}-iter-<N>/
├── auto-${AGENT_ID}-iter-<N>-CONTEXT.md      # Autonomous context (replaces discuss-phase)
├── auto-${AGENT_ID}-iter-<N>-RESEARCH.md     # From gsd-phase-researcher
├── auto-${AGENT_ID}-iter-<N>-01-PLAN.md      # From gsd-planner (wave 1: code changes)
├── auto-${AGENT_ID}-iter-<N>-02-PLAN.md      # From gsd-planner (wave 2: validation/submission)
├── auto-${AGENT_ID}-iter-<N>-01-SUMMARY.md   # From gsd-executor
├── auto-${AGENT_ID}-iter-<N>-02-SUMMARY.md   # From gsd-executor
└── auto-${AGENT_ID}-iter-<N>-VERIFICATION.md # From gsd-verifier
```

Level 1 agents **bridge** both systems: they orchestrate GSD agents writing to `.planning/` and write summaries to `.autonomous/iterations/<N>/`.

## Orchestration Loop

### 0. Initialize
```python
# Pseudocode for the loop
read config.json for settings
read/write goal.md from $ARGUMENTS
read state.json for current iteration
if state.status == "completed": exit with success
iteration = state.iteration + 1
create iterations/<iteration>/ directory
```

Update state.json:
```json
{
  "iteration": N,
  "phase": "planning",
  "status": "running",
  "branch": "",
  "worktree_path": "",
  "pr_url": "",
  "started_at": "<timestamp>",
  "last_updated": "<timestamp>"
}
```

### 0.5. Create GSD Phase Directory
```bash
GSD_PHASE_DIR=".planning/phases/auto-${AGENT_ID}-iter-${iteration}"
mkdir -p "$GSD_PHASE_DIR"
```

This directory is where GSD agents will write their artifacts (CONTEXT.md, RESEARCH.md, PLAN.md, SUMMARY.md, VERIFICATION.md).

### 0.7. Create Iteration Branch
Create an isolated branch for this iteration to avoid conflicts with parallel agents:
```bash
ITER_BRANCH=$(bash ~/multiagent/scripts/git-workflow.sh branch-name "${AGENT_ID}-iter" "${iteration}")
cd ~/multiagent && git fetch origin main && git checkout -b "$ITER_BRANCH" origin/main
```

Record in state.json:
```json
{
  "iteration": N,
  "branch": "$ITER_BRANCH",
  "phase": "planning",
  "status": "running"
}
```

### 1. PLANNING PHASE
Spawn a **Task agent** (model: opus, subagent_type: general-purpose) with the full instructions from `/exps-auto-plan`. Pass it:

```
{
  agent_id: $AGENT_ID,
  autonomous_dir: $AUTONOMOUS_DIR,
  iteration: N,
  goal: <contents of goal.md>,
  insights: <contents of insights.md>,
  prev_analysis: <contents of iterations/<N-1>/analysis.md or "First iteration">,
  gsd_phase_dir: ".planning/phases/auto-${AGENT_ID}-iter-<N>"
}
```

The agent must:
- Generate autonomous CONTEXT.md (replaces interactive discuss-phase)
- Spawn gsd-phase-researcher → RESEARCH.md
- Spawn gsd-planner → PLAN.md files (with frontmatter, waves, must_haves)
- Spawn gsd-plan-checker → revision loop (max 3)
- Bridge: Write summary to `iterations/<N>/plan.md`
- Return: plan count, plan file paths, wave count, Codex review status, summary string

Update state.json phase to "executing".

### 2. EXECUTION PHASE
Spawn a **Task agent** (model: opus, subagent_type: general-purpose) with the full instructions from `/exps-auto-execute`. Pass it:

```
{
  agent_id: $AGENT_ID,
  autonomous_dir: $AUTONOMOUS_DIR,
  iteration: N,
  goal: <contents of goal.md>,
  gsd_phase_dir: ".planning/phases/auto-${AGENT_ID}-iter-<N>",
  plan_files: <list of PLAN.md paths from step 1>
}
```

The agent must:
- Group PLAN.md files by wave from frontmatter
- Spawn gsd-executor agents (parallel within wave, sequential across waves)
- Inline exps-validate logic (16-parameter parity, config correctness, comb registration, queue bounds)
- Inline exps-smoke-iterate logic (submit, monitor, error handling, retry)
- Spawn gsd-verifier → VERIFICATION.md
- Bridge: Write summary to `iterations/<N>/execution.md`
- Return: status, job_ids, validation results, plan execution counts, Codex review status

Update state.json phase to "analyzing".

### 3. ANALYSIS PHASE
Spawn a **Task agent** (model: opus, subagent_type: general-purpose) with the full instructions from `/exps-auto-analyze`. Pass it:

```
{
  agent_id: $AGENT_ID,
  autonomous_dir: $AUTONOMOUS_DIR,
  iteration: N,
  goal: <contents of goal.md>,
  insights: <contents of insights.md>,
  execution_file: "iterations/<N>/execution.md",
  gsd_phase_dir: ".planning/phases/auto-${AGENT_ID}-iter-<N>"
}
```

The agent must:
- Spawn 4 parallel collection agents (logs, W&B, transcripts, config drift)
- Inline exps-check logic (health dashboard)
- Inline exps-debugger logic (reward/training diagnostics)
- Verify against goal requirements
- Generate corrective plan for next iteration
- Bridge: Write analysis to `iterations/<N>/analysis.md`, update insights.md
- Return: GOAL_MET boolean, requirements met/total, anomalies, corrective plan summary, Codex review status

Update state.json phase to "reviewing".

### 4. FINAL CODEX REVIEW (Cross-Cutting)

**Note:** Each Level 1 agent (auto-plan, auto-execute, auto-analyze) already performs its own Codex review after completing its GSD phases. This Step 4 is a **cross-cutting final review** that looks at the full iteration holistically.

Call Codex (GPT-5.3-Codex) via OpenRouter for an independent sanity check of the entire iteration.

```bash
# Check for API key
OPENROUTER_KEY=$(printenv OPENROUTER_API_KEY 2>/dev/null || echo "")
```

If key is available, make the API call:
```bash
# Read all artifacts for this iteration
PLAN=$(cat $AUTONOMOUS_DIR/iterations/<N>/plan.md)
EXECUTION=$(cat $AUTONOMOUS_DIR/iterations/<N>/execution.md)
ANALYSIS=$(cat $AUTONOMOUS_DIR/iterations/<N>/analysis.md)
CODEX_PLAN_REVIEW=$(cat $AUTONOMOUS_DIR/iterations/<N>/codex_plan_review.md 2>/dev/null || echo "N/A")
CODEX_EXEC_REVIEW=$(cat $AUTONOMOUS_DIR/iterations/<N>/codex_execution_review.md 2>/dev/null || echo "N/A")
CODEX_ANALYSIS_REVIEW=$(cat $AUTONOMOUS_DIR/iterations/<N>/codex_analysis_review.md 2>/dev/null || echo "N/A")
GOAL=$(cat $AUTONOMOUS_DIR/goal.md)
INSIGHTS=$(tail -80 $AUTONOMOUS_DIR/insights.md)

CODEX_MODEL=$(python3 -c "import json; c=json.load(open('$AUTONOMOUS_DIR/config.json')); print(c.get('codex_review_model', c.get('external_review_model', 'openai/gpt-5.3-codex')))")
CODEX_MAX_TOKENS=$(python3 -c "import json; print(json.load(open('$AUTONOMOUS_DIR/config.json')).get('codex_max_tokens', 4000))")

curl -s https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $OPENROUTER_KEY" \
  -H "Content-Type: application/json" \
  -H "HTTP-Referer: https://claude-code-autonomous-loop" \
  -d "$(cat <<PAYLOAD
{
  "model": "$CODEX_MODEL",
  "messages": [
    {
      "role": "system",
      "content": "You are a senior ML research advisor performing a cross-cutting review of a full autonomous research iteration. You are reviewing the PLAN, EXECUTION, and ANALYSIS phases together to identify:\n1. Logical inconsistencies between phases (e.g., plan said X but execution did Y)\n2. Missed opportunities or blind spots across the iteration\n3. Whether the corrective plan for the next iteration is well-targeted\n4. Statistical errors, misinterpreted metrics, or overlooked failure modes\n5. Whether previous Codex per-phase reviews were adequately addressed\n\nBe skeptical and thorough. Output a structured review with:\n- CROSS-PHASE CONSISTENCY: [CONSISTENT / ISSUES FOUND]\n- ISSUES FOUND: numbered list\n- RECOMMENDATIONS FOR NEXT ITERATION: numbered list\n- CONFIDENCE ASSESSMENT: [HIGH / MEDIUM / LOW] with reasoning"
    },
    {
      "role": "user",
      "content": "GOAL:\n$GOAL\n\nPLAN:\n$PLAN\n\nEXECUTION:\n$EXECUTION\n\nANALYSIS:\n$ANALYSIS\n\nPER-PHASE CODEX REVIEWS:\n--- Plan Review ---\n$CODEX_PLAN_REVIEW\n--- Execution Review ---\n$CODEX_EXEC_REVIEW\n--- Analysis Review ---\n$CODEX_ANALYSIS_REVIEW\n\nACCUMULATED INSIGHTS:\n$INSIGHTS\n\nPlease review this full iteration. Are the phases consistent? Were the per-phase Codex reviews addressed? Are there any cross-cutting issues?"
    }
  ],
  "max_tokens": $CODEX_MAX_TOKENS
}
PAYLOAD
)"
```

Parse the response (extract `.choices[0].message.content` from JSON) and write to `iterations/<N>/external_review.md`.

**If Codex returns critical cross-phase inconsistencies:** Log them prominently in insights.md so the next iteration's planning phase addresses them.

**If the API call fails** (model not available): Try the fallback model from `config.json.codex_review_model_fallback`. If that also fails or no API key: skip this step and note it in insights.md.

### 5. CHECK GOAL COMPLETION
Read the analysis summary. If ALL verification requirements are MET:
- Update state.json: `status: "completed"`, `goal_met: true`
- Write final report to `$AUTONOMOUS_DIR/FINAL_REPORT.md`
- Append completion note to insights.md
- STOP the loop

If NOT all met:
- Increment iteration
- Continue to next loop cycle

### 5.5. Create PR for Iteration
After analysis completes, if there are commits on the iteration branch:

```bash
# Check if we have commits beyond origin/main
COMMIT_COUNT=$(git -C "$ITER_WORKTREE_PATH" log --oneline origin/main..HEAD | wc -l)
if [ "$COMMIT_COUNT" -gt 0 ]; then
    COMMIT_LOG=$(git -C "$ITER_WORKTREE_PATH" log --oneline origin/main..HEAD)
    PR_TITLE="auto/${AGENT_ID}/iter-${iteration}: ${goal_summary}"
    PR_BODY="## Summary
- Agent: ${AGENT_ID}
- Autonomous iteration ${iteration}
- Goal: $(head -1 $AUTONOMOUS_DIR/goal.md)

## Changes
${COMMIT_LOG}

## Verification
- $(grep -m1 'Status:' $AUTONOMOUS_DIR/iterations/${iteration}/analysis.md 2>/dev/null || echo 'Status: pending')

## Artifacts
- GSD phase: .planning/phases/auto-${AGENT_ID}-iter-${iteration}/
- Analysis: .autonomous/${AGENT_ID}/iterations/${iteration}/

> Auto-generated by exps-auto-loop agent ${AGENT_ID} iteration ${iteration}. Targets multiagent repo (staging).
> Production deployment requires user verification."

    PR_URL=$(bash ~/multiagent/scripts/git-workflow.sh create-pr "$ITER_WORKTREE_PATH" "$PR_TITLE" "$PR_BODY")
fi
```

Record `pr_url` in state.json. If PR creation fails, log warning and continue (non-blocking).

**Note:** Do NOT cleanup the worktree here — step 5.7 needs access to config files in the worktree. Cleanup happens in step 5.8 after the report is generated.

### 5.7. Generate Iteration Report

Generate `$AUTONOMOUS_DIR/iterations/${iteration}/REPORT.md` — the primary user-readable output for this iteration. This report must contain **exact values** from collected data, never summaries or approximations.

**Data sources to read:**
```bash
PLAN_SUMMARY=$(cat $AUTONOMOUS_DIR/iterations/${iteration}/plan.md 2>/dev/null)
EXECUTION=$(cat $AUTONOMOUS_DIR/iterations/${iteration}/execution.md 2>/dev/null)
ANALYSIS=$(cat $AUTONOMOUS_DIR/iterations/${iteration}/analysis.md 2>/dev/null)
GOAL=$(cat $AUTONOMOUS_DIR/goal.md)

# Set WORKING_DIR — prefer iteration worktree (if still exists), fallback to main repo
WORKING_DIR="${ITER_WORKTREE_PATH:-$HOME/multiagent}"
if [ ! -d "$WORKING_DIR" ]; then
    WORKING_DIR="$HOME/multiagent"
fi

# Extract config parameters from the Python config files
DEBATE_CONFIG="$WORKING_DIR/configs/sweep_math_debate_grpo.py"
BASELINE_CONFIG="$WORKING_DIR/configs/sweep_math_baseline_grpo.py"

# Extract W&B run URLs and metrics from analysis.md
WANDB_PROJECT="cohere/multiagent-debate-rl"

# Extract PR URL from state.json
PR_URL=$(python3 -c "import json; print(json.load(open('$AUTONOMOUS_DIR/state.json')).get('pr_url', 'pending'))" 2>/dev/null || echo "pending")
PR_NUM=$(echo "$PR_URL" | grep -oE '[0-9]+$' || echo "N/A")

# Calculate elapsed time
START_TIME=$(python3 -c "import json; print(json.load(open('$AUTONOMOUS_DIR/state.json')).get('started_at', ''))" 2>/dev/null)
ELAPSED=$(python3 -c "
from datetime import datetime
try:
    start = datetime.fromisoformat('$START_TIME'.replace('Z','+00:00'))
    elapsed = datetime.now(start.tzinfo) - start
    hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
    minutes = remainder // 60
    print(f'{hours}h {minutes}m')
except: print('unknown')
" 2>/dev/null || echo "unknown")
```

**Extract exact config parameters:**
```bash
CONFIG_TABLE=$(python3 -c "
import importlib.util, json

def load_config(path):
    spec = importlib.util.spec_from_file_location('config', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

try:
    debate = load_config('$DEBATE_CONFIG')
    baseline = load_config('$BASELINE_CONFIG')
    params = [
        ('Model', getattr(debate, 'CKPT_PATH', 'N/A'), getattr(baseline, 'CKPT_PATH', 'N/A')),
        ('Learning rate', getattr(debate, 'LEARNING_RATE', 'N/A'), getattr(baseline, 'LEARNING_RATE', 'N/A')),
        ('Batch size', getattr(debate, 'TRAIN_BATCH_SIZE', 'N/A'), getattr(baseline, 'TRAIN_BATCH_SIZE', 'N/A')),
        ('Steps', getattr(debate, 'TOTAL_TRAIN_STEPS', 'N/A'), getattr(baseline, 'TOTAL_TRAIN_STEPS', 'N/A')),
        ('KL beta', getattr(debate, 'KL_BETA', 'N/A'), getattr(baseline, 'KL_BETA', 'N/A')),
        ('Generations/prompt', getattr(debate, 'GENERATIONS_PER_PROMPT', 'N/A'), getattr(baseline, 'GENERATIONS_PER_PROMPT', 'N/A')),
        ('Max seq length', getattr(debate, 'MAX_SEQUENCE_LENGTH', 'N/A'), getattr(baseline, 'MAX_SEQUENCE_LENGTH', 'N/A')),
        ('Seed', getattr(debate, 'SEED', 'N/A'), getattr(baseline, 'SEED', 'N/A')),
    ]
    for name, d, b in params:
        print(f'| {name} | {d} | {b} |')
except Exception as e:
    print(f'| Error loading configs | {e} | |')
" 2>/dev/null)
```

**Extract metrics from analysis.md:**
```bash
# Parse exact metric values from analysis.md sections
DEBATE_REWARD=$(grep -oP 'Debate mean_reward:\s*\K[\d.]+' $AUTONOMOUS_DIR/iterations/${iteration}/analysis.md 2>/dev/null || echo "N/A")
BASELINE_REWARD=$(grep -oP 'Baseline mean_reward:\s*\K[\d.]+' $AUTONOMOUS_DIR/iterations/${iteration}/analysis.md 2>/dev/null || echo "N/A")
REWARD_DELTA=$(python3 -c "
d,b = '$DEBATE_REWARD', '$BASELINE_REWARD'
if d != 'N/A' and b != 'N/A': print(f'{float(d)-float(b):+.4f}')
else: print('N/A')
" 2>/dev/null || echo "N/A")

# Extract all metrics block from analysis.md for the full table
LOSS_TREND=$(grep -oP 'Loss trend:\s*\K\S+' $AUTONOMOUS_DIR/iterations/${iteration}/analysis.md 2>/dev/null || echo "N/A")
GRAD_NORM=$(grep -oP 'Grad norm:\s*\K\S+' $AUTONOMOUS_DIR/iterations/${iteration}/analysis.md 2>/dev/null || echo "N/A")
KL_DIV=$(grep -oP 'KL divergence:\s*\K\S+' $AUTONOMOUS_DIR/iterations/${iteration}/analysis.md 2>/dev/null || echo "N/A")
REWARD_TREND=$(grep -oP 'Reward trend:\s*\K\S+' $AUTONOMOUS_DIR/iterations/${iteration}/analysis.md 2>/dev/null || echo "N/A")
```

**Extract W&B run info:**
```bash
# Get W&B run URLs from analysis.md "W&B Run URLs" section (populated by auto-analyze Agent 2)
WANDB_DEBATE_URL=$(grep -oP 'Debate:\s*\Khttps://wandb\.ai/\S+' $AUTONOMOUS_DIR/iterations/${iteration}/analysis.md 2>/dev/null || echo "")
WANDB_BASELINE_URL=$(grep -oP 'Baseline:\s*\Khttps://wandb\.ai/\S+' $AUTONOMOUS_DIR/iterations/${iteration}/analysis.md 2>/dev/null || echo "")

# Fallback: try extracting any W&B URLs from analysis.md or execution.md
if [ -z "$WANDB_DEBATE_URL" ]; then
    WANDB_DEBATE_URL=$(grep -oP 'https://wandb\.ai/\S+' $AUTONOMOUS_DIR/iterations/${iteration}/analysis.md 2>/dev/null | head -1 || echo "")
fi
if [ -z "$WANDB_DEBATE_URL" ]; then
    WANDB_DEBATE_URL=$(grep -oP 'https://wandb\.ai/\S+' $AUTONOMOUS_DIR/iterations/${iteration}/execution.md 2>/dev/null | head -1 || echo "")
fi
# Final fallback: construct dashboard URL
WANDB_DEBATE_URL="${WANDB_DEBATE_URL:-https://wandb.ai/${WANDB_PROJECT}}"
WANDB_BASELINE_URL="${WANDB_BASELINE_URL:-https://wandb.ai/${WANDB_PROJECT}}"

# Extract W&B chart images — check wandb local media directory, then sweep artifacts
REWARD_CHART=$(find "$WORKING_DIR/wandb/" -path "*/media/images/*reward*" -name "*.png" 2>/dev/null | sort | tail -1 || echo "")
if [ -z "$REWARD_CHART" ]; then
    REWARD_CHART=$(find ~/sweep_logs_*/ -name "*reward*.png" 2>/dev/null | sort | tail -1 || echo "")
fi
LOSS_CHART=$(find "$WORKING_DIR/wandb/" -path "*/media/images/*loss*" -name "*.png" 2>/dev/null | sort | tail -1 || echo "")
if [ -z "$LOSS_CHART" ]; then
    LOSS_CHART=$(find ~/sweep_logs_*/ -name "*loss*.png" 2>/dev/null | sort | tail -1 || echo "")
fi
```

**Extract transcript excerpts from analysis.md or debug artifacts:**
```bash
# Look for transcript sections in analysis or sweep job outputs
TRANSCRIPT_DIR=$(ls -d ~/sweep_jobs/*/output/ 2>/dev/null | tail -1)
if [ -n "$TRANSCRIPT_DIR" ]; then
    # Extract a correct debate transcript example
    DEBATE_CORRECT=$(python3 -c "
import json, glob
files = sorted(glob.glob('$TRANSCRIPT_DIR/*.json'))[:5]
for f in files:
    try:
        data = json.load(open(f))
        # Look for debate transcripts with correct answers
        if 'debate' in str(data).lower() and data.get('reward', 0) > 0.5:
            turns = data.get('turns', data.get('messages', []))
            for t in turns[:6]:
                role = t.get('role', t.get('speaker', 'unknown'))
                content = t.get('content', '')[:300]
                print(f'> **{role}:** {content}')
            break
    except: continue
else:
    print('> No correct debate transcripts available in this iteration.')
" 2>/dev/null)

    # Extract an incorrect debate transcript (failure mode)
    DEBATE_INCORRECT=$(python3 -c "
import json, glob
files = sorted(glob.glob('$TRANSCRIPT_DIR/*.json'))[:10]
for f in files:
    try:
        data = json.load(open(f))
        if 'debate' in str(data).lower() and data.get('reward', 0) < 0.5:
            turns = data.get('turns', data.get('messages', []))
            for t in turns[:6]:
                role = t.get('role', t.get('speaker', 'unknown'))
                content = t.get('content', '')[:300]
                print(f'> **{role}:** {content}')
            break
    except: continue
else:
    print('> No incorrect debate transcripts available in this iteration.')
" 2>/dev/null)

    # Extract a baseline transcript for the same problem
    BASELINE_TRANSCRIPT=$(python3 -c "
import json, glob
files = sorted(glob.glob('$TRANSCRIPT_DIR/*.json'))[:10]
for f in files:
    try:
        data = json.load(open(f))
        if 'baseline' in str(data).lower() or ('debate' not in str(data).lower()):
            turns = data.get('turns', data.get('messages', []))
            for t in turns[:4]:
                role = t.get('role', t.get('speaker', 'unknown'))
                content = t.get('content', '')[:300]
                print(f'> **{role}:** {content}')
            break
    except: continue
else:
    print('> No baseline transcripts available in this iteration.')
" 2>/dev/null)
else
    DEBATE_CORRECT="> No transcript artifacts found for this iteration."
    DEBATE_INCORRECT="> No transcript artifacts found for this iteration."
    BASELINE_TRANSCRIPT="> No transcript artifacts found for this iteration."
fi
```

**Extract changes from previous iteration:**
```bash
PREV_DELTA="First iteration — no previous comparison"
if [ -f "$AUTONOMOUS_DIR/iterations/$((iteration-1))/analysis.md" ]; then
    PREV_DELTA=$(grep -A 5 "Corrective Plan\|Changes from Previous\|Priority 1:" $AUTONOMOUS_DIR/iterations/$((iteration-1))/analysis.md 2>/dev/null | head -10 || echo "See previous analysis")
fi
```

**Write the report:**

Write `$AUTONOMOUS_DIR/iterations/${iteration}/REPORT.md`:

```markdown
## Experiment Report: ${AGENT_ID} — Iteration ${iteration}
**Branch:** auto/${AGENT_ID}-iter-${iteration} | **W&B:** [${WANDB_PROJECT}](${WANDB_DEBATE_URL}) | **Duration:** ${ELAPSED}

### Research Question
${GOAL}

### Experimental Design
| Parameter | Debate | Baseline |
|-----------|--------|----------|
${CONFIG_TABLE}

**Config files:** `configs/sweep_math_debate_grpo.py` vs `configs/sweep_math_baseline_grpo.py`
**Changes from previous iteration:** ${PREV_DELTA}

### Quantitative + Qualitative Results

#### Reward Curves
${REWARD_CHART:+!​[Reward over training steps](${REWARD_CHART})}
${REWARD_CHART:-*No reward chart available — check W&B dashboard*}

| Metric | Debate | Baseline | Delta |
|--------|--------|----------|-------|
| Final reward (mean) | **${DEBATE_REWARD}** | **${BASELINE_REWARD}** | ${REWARD_DELTA} |
| Reward trend | ${REWARD_TREND} | — | |
| Loss trend | ${LOSS_TREND} | — | |
| KL divergence | ${KL_DIV} | — | |
| Grad norm | ${GRAD_NORM} | — | |

#### Loss Curves
${LOSS_CHART:+!​[Training loss](${LOSS_CHART})}
${LOSS_CHART:-*No loss chart available — check W&B dashboard*}

#### Sample Transcripts
**Debate (correct):**
${DEBATE_CORRECT}

**Debate (incorrect — failure mode):**
${DEBATE_INCORRECT}

**Baseline (same problem):**
${BASELINE_TRANSCRIPT}

### Interpretation
$(grep -A 20 "## Corrective Plan\|## Goal Verification\|## All Goals Met\|## Anomalies Found" $AUTONOMOUS_DIR/iterations/${iteration}/analysis.md 2>/dev/null | head -30 || echo "See analysis.md for full interpretation")

### Goal Progress
$(grep -A 10 "## Goal Verification" $AUTONOMOUS_DIR/iterations/${iteration}/analysis.md 2>/dev/null | head -15 || echo "See analysis.md")

### W&B Links
- Debate: [${AGENT_ID}-debate](${WANDB_DEBATE_URL})
- Baseline: [${AGENT_ID}-baseline](${WANDB_BASELINE_URL})
- Dashboard: [${WANDB_PROJECT}](https://wandb.ai/${WANDB_PROJECT})

### PR Link
[PR #${PR_NUM}: auto/${AGENT_ID}/iter-${iteration}](${PR_URL})

### Artifacts
- Plan: `$AUTONOMOUS_DIR/iterations/${iteration}/plan.md`
- Execution: `$AUTONOMOUS_DIR/iterations/${iteration}/execution.md`
- Analysis: `$AUTONOMOUS_DIR/iterations/${iteration}/analysis.md`
- GSD Phase: `.planning/phases/auto-${AGENT_ID}-iter-${iteration}/`
- Codex Reviews: `$AUTONOMOUS_DIR/iterations/${iteration}/codex_*_review.md`
- External Review: `$AUTONOMOUS_DIR/iterations/${iteration}/external_review.md`
```

**Key rules for report generation:**
- ALWAYS use exact numeric values extracted from analysis.md and W&B data — never write "improved significantly"
- ALWAYS include figure paths (W&B chart URLs or local screenshot paths) — use placeholders with instructions if unavailable
- ALWAYS include verbatim transcript excerpts from job output artifacts — never paraphrase or summarize transcript content
- If any data source is unavailable, note it explicitly (e.g., "N/A — W&B data not yet available") rather than omitting the section

### 5.8. Cleanup Iteration Worktree

After the report is generated and all data extracted, cleanup the iteration worktree:
```bash
if [ -n "$ITER_WORKTREE_PATH" ] && [ -d "$ITER_WORKTREE_PATH" ]; then
    bash ~/multiagent/scripts/git-workflow.sh cleanup "$ITER_WORKTREE_PATH"
fi
```

### 6. CONTEXT MANAGEMENT
After every iteration, write a checkpoint:

**Update insights.md** — append a summary of this iteration:
```markdown
### Iteration <N> — <timestamp>
- Status: <SUCCESS/PARTIAL/FAILED>
- GSD phase: `.planning/phases/auto-${AGENT_ID}-iter-<N>/`
- Plans executed: <count>
- Changes made: <brief list>
- Key finding: <most important insight>
- Goal progress: <X of Y requirements met>
- Branch: <branch name>
- PR: <PR URL or "pending">
- Next action: <what the next iteration should focus on>
```

**Update state.json** with current iteration and timestamp.

Also update `iterations_summary` array in state.json to track per-iteration branch/PR info:
```json
{
  "iterations_summary": [
    {
      "iteration": N,
      "branch": "$ITER_BRANCH",
      "pr_url": "$PR_URL",
      "status": "SUCCESS/PARTIAL/FAILED",
      "goal_progress": "X of Y"
    }
  ]
}
```

This ensures that if context is lost (context window limit, crash, overnight restart), the next invocation can read these files and pick up exactly where it left off.

### 7. LOOP CONTROL
```
max_iterations = config.json.max_iterations (default: 20)
if iteration >= max_iterations:
    write "MAX ITERATIONS REACHED" to insights.md
    update state.json status to "max_iterations_reached"
    write partial report to FINAL_REPORT.md
    STOP
else:
    go to step 1 with iteration += 1
```

## Final Report Format

When the goal is met (or max iterations reached), write `$AUTONOMOUS_DIR/FINAL_REPORT.md`:

```markdown
# Autonomous Loop Final Report
Completed: <timestamp>

## Goal
<the goal>

## Result: [GOAL MET / PARTIALLY MET / NOT MET]

## Iterations: <N> of <max>

## Summary
<2-3 paragraph summary of what happened across all iterations>

## Verification Requirements
| Requirement | Status | Evidence | Iteration Achieved |
|------------|--------|----------|-------------------|
| Req 1 | MET/NOT MET | <evidence> | <N> |
| Req 2 | MET/NOT MET | <evidence> | <N> |

## Key Insights
<bulleted list of the most important things learned>

## Changes Applied
<cumulative list of all config/code changes across all iterations>

## GSD Phase Directories
<list of all .planning/phases/auto-${AGENT_ID}-iter-<N>/ directories with their artifacts>

## Jobs Submitted
<list of all job IDs with outcomes>

## Errors Encountered and Resolved
<list of all errors and their fixes>

## Recommendations
<what to do next, if anything>
```

## Recovery from Context Loss (Dual-Level)

If invoked and state.json shows `status: "running"`:

### Level 0 Recovery (Autonomous State)
1. Read the last iteration number from state.json
2. Check which phase was last completed by looking for files:
   - plan.md exists but not execution.md → resume from execution
   - execution.md exists but not analysis.md → resume from analysis
   - analysis.md exists → start next iteration

### Level 1 Recovery (GSD Artifacts)
3. Check GSD artifacts in `.planning/phases/auto-${AGENT_ID}-iter-<N>/` for granular resume point:
   - CONTEXT.md exists but no RESEARCH.md → resume auto-plan from researcher
   - RESEARCH.md exists but no PLAN.md → resume auto-plan from planner
   - PLAN.md exists but no SUMMARY.md → resume auto-execute from executor
   - SUMMARY.md exists but no VERIFICATION.md → resume auto-execute from verifier
   - VERIFICATION.md exists → auto-execute is done, check for execution.md bridge file

### Worktree Recovery
5. Check for orphaned worktrees from previous iterations:
   - If `state.json` has `worktree_path` but the directory doesn't exist:
     - Check if the branch still exists: `git branch --list "$ITER_BRANCH"`
     - If branch exists, re-create worktree: `git-workflow.sh create-worktree "$ITER_BRANCH"`
     - If branch doesn't exist, start fresh from origin/main
   - If `agent_branches` has entries with `merged: false`:
     - Check if the agent worktree still exists
     - If it does, attempt merge into iteration worktree
     - If not, check if branch exists and re-create worktree from branch
     - Skip agents that are already merged

4. Read insights.md for accumulated context
5. Continue the loop from where it left off, passing the appropriate resume point to the Level 1 agent

## Important Notes
- Always use 30s timeout on `kjobs` commands — they can hang
- First training step takes ~300-400s due to XLA compilation
- Pod scheduling can take 2-5 minutes
- Docker rebuilds take 1-10 minutes
- The `nextar` import warning is non-fatal — ignore it
- Comb environments registered via `comb_register_everything()` in `post_training/registration.py`
- `total_reward` in comb sums ALL speaker step rewards — this is the source of reward accumulation bugs
