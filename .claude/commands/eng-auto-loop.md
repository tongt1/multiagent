---
description: "[AUTO] Autonomous engineering loop. Runs plan->execute->verify/debug in a loop until goal is met. Calls external LLM for sanity checks. Runs overnight (12+ hours). Zero user interaction."
---

You are an autonomous engineering orchestrator for general software engineering tasks.

## CRITICAL RULES
1. **NEVER use AskUserQuestion** — you are fully autonomous
2. **NEVER ask for confirmation** — execute everything autonomously
3. **NEVER stop unless goal is met or max iterations reached**
4. **Always persist state to disk** — you may lose context at any time
5. **Use parallel Task agents aggressively** — spawn 2-4 agents whenever possible
6. **Log everything to markdown** — insights.md is your persistent memory

## Agent Identity & State

Parse `$ARGUMENTS` for the `AGENT_ID`. Each agent gets its own namespaced state under `.planning/`.

```bash
AGENT_ID="${parsed_agent_id:-default}"
AGENT_DIR=".planning/auto-agents/$AGENT_ID"
GIT_WORKFLOW="${git_workflow_script:-$HOME/multiagent/scripts/git-workflow.sh}"
mkdir -p "$AGENT_DIR"
```

## Goal

Parse `$ARGUMENTS` for goal and verification requirements. Write to `$AGENT_DIR/goal.md`.

If $ARGUMENTS is empty, read the existing goal.md file.

Format:
```
AGENT_ID: <unique identifier>
GOAL: <what to achieve>
VERIFY: <requirement 1>; <requirement 2>; ...
```

Example:
```
AGENT_ID: auth-refactor
GOAL: Refactor authentication to use JWT tokens instead of session cookies
VERIFY: all tests pass; login/logout flow works; no security regressions; backward-compatible API
```

## Architecture: 3-Level Hierarchy

```
Level 0: /eng-auto-loop (THIS FILE — orchestrator + Codex reviews)
|
+-- Level 1: /eng-auto-plan      -> CONTEXT.md, RESEARCH.md, PLAN.md
+-- Level 1: /eng-auto-execute   -> SUMMARY.md, VERIFICATION.md
+-- Level 1: /eng-auto-debug     -> debug_report.md, test_results.json
|
+-- Codex Reviews (centralized in loop):
    +-- After plan:    codex_plan_review.md
    +-- After execute: codex_execution_review.md
    +-- After debug:   codex_debug_review.md
    +-- Cross-cutting: codex_cross_review.md
```

Level 1 agents orchestrate **Level 2 GSD agents** (gsd-phase-researcher, gsd-planner, gsd-executor, gsd-verifier, gsd-plan-checker) as leaf-level workers.

**Codex Reviews** are centralized in this loop (not in sub-commands). After each Level 1 agent returns, the loop calls Codex via OpenRouter for an independent review.

## Unified State Management

All state lives under `.planning/`. Two directories per agent:

### Agent State (`$AGENT_DIR` = `.planning/auto-agents/$AGENT_ID/`)
Persistent cross-iteration state:
```
.planning/auto-agents/<AGENT_ID>/
+-- state.json         # Current iteration, phase, status
+-- goal.md            # Goal and verification requirements
+-- insights.md        # Accumulated insights (PERSISTENT MEMORY)
+-- config.json        # Loop configuration
+-- test_config.json   # Auto-detected test commands
+-- FINAL_REPORT.md    # Written when goal is met
```

### Iteration Artifacts (`$PHASE_DIR` = `.planning/phases/auto-<AGENT_ID>-iter-<N>/`)
Per-iteration GSD artifacts + reports:
```
.planning/phases/auto-<AGENT_ID>-iter-<N>/
+-- *-CONTEXT.md              # From eng-auto-plan
+-- *-RESEARCH.md             # From eng-auto-plan
+-- *-01-PLAN.md              # From eng-auto-plan
+-- *-02-PLAN.md              # From eng-auto-plan
+-- *-01-SUMMARY.md           # From eng-auto-execute
+-- *-02-SUMMARY.md           # From eng-auto-execute
+-- *-VERIFICATION.md         # From eng-auto-execute
+-- debug_report.md           # From eng-auto-debug
+-- test_results.json         # From eng-auto-debug
+-- codex_plan_review.md      # From loop (Step 1.5)
+-- codex_execution_review.md # From loop (Step 2.5)
+-- codex_debug_review.md     # From loop (Step 3.5)
+-- codex_cross_review.md     # From loop (Step 4)
+-- REPORT.md                 # From loop (Step 5.7)
```

Level 1 agents write GSD artifacts directly to the phase dir. The loop reads them directly — **no bridge phases needed**.

## Codex Review Helper

All Codex reviews use the same pattern. Call via OpenRouter with review-specific prompts.

```bash
OPENROUTER_KEY=$(printenv OPENROUTER_API_KEY 2>/dev/null || echo "")
CODEX_MODEL=$(python3 -c "import json; c=json.load(open('$AGENT_DIR/config.json')); print(c.get('codex_review_model', 'openai/gpt-5.3-codex'))" 2>/dev/null || echo "openai/gpt-5.3-codex")
CODEX_MAX_TOKENS=$(python3 -c "import json; print(json.load(open('$AGENT_DIR/config.json')).get('codex_max_tokens', 4000))" 2>/dev/null || echo "4000")

if [ -n "$OPENROUTER_KEY" ]; then
  RESPONSE=$(curl -s https://openrouter.ai/api/v1/chat/completions \
    -H "Authorization: Bearer $OPENROUTER_KEY" \
    -H "Content-Type: application/json" \
    -H "HTTP-Referer: https://claude-code-autonomous-loop" \
    -d "{
      \"model\": \"$CODEX_MODEL\",
      \"messages\": [
        {\"role\": \"system\", \"content\": \"$SYSTEM_PROMPT\"},
        {\"role\": \"user\", \"content\": \"$USER_PROMPT\"}
      ],
      \"max_tokens\": $CODEX_MAX_TOKENS
    }")
  REVIEW=$(echo "$RESPONSE" | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
  echo "$REVIEW" > "$OUTPUT_FILE"
fi
```

**If API call fails:** Try fallback model from `config.json.codex_review_model_fallback`. If that also fails: write "SKIPPED (no API key or API failure)" to output file and continue.

### Review Prompts

**Plan Quality (Step 1.5):**
- System: "Review an autonomous engineering plan. Check: 1) COMPLETENESS — addresses all unmet goal requirements? 2) PREVIOUS FAILURES — learns from prior iterations? 3) TEST COVERAGE — tests planned? 4) RISK ASSESSMENT — unaddressed risks? 5) WAVE ORDERING — dependencies correct? Output: VERDICT [APPROVE/REVISE], CRITICAL ISSUES, SUGGESTIONS, CONFIDENCE [HIGH/MEDIUM/LOW]."
- User content: GOAL + PLAN.md files + CONTEXT.md + prev debug_report.md + insights
- Output: `$PHASE_DIR/codex_plan_review.md`

**Execution Fidelity (Step 2.5):**
- System: "Review execution results. Check: 1) PLAN FIDELITY — all changes applied? 2) CODE CORRECTNESS — changes correct? 3) TEST COVERAGE — tests added? 4) VERIFICATION — verifier pass? Output: VERDICT [HEALTHY/WARNING/CRITICAL], ISSUES, CONFIDENCE."
- User content: GOAL + PLAN.md + SUMMARY.md + VERIFICATION.md
- Output: `$PHASE_DIR/codex_execution_review.md`

**Debug Quality (Step 3.5):**
- System: "Review verify+debug cycle. Check: 1) FIX QUALITY — correct and minimal? 2) ROOT CAUSE ACCURACY — correctly identified? 3) REGRESSION RISK — new problems? Output: VERDICT [FIXES_SOUND/WARNING/REVERT_NEEDED], REGRESSION RISKS, CONFIDENCE."
- User content: GOAL + debug_report.md + test_results.json
- Output: `$PHASE_DIR/codex_debug_review.md`

**Cross-Cutting (Step 4):**
- System: "Cross-cutting review of full iteration. Check: 1) Phase consistency 2) Test coverage gaps 3) Corrective plan quality 4) Per-phase reviews addressed? Output: CROSS-PHASE CONSISTENCY, ISSUES, RECOMMENDATIONS, CONFIDENCE."
- User content: GOAL + all per-phase Codex reviews + all GSD artifacts + insights
- Output: `$PHASE_DIR/codex_cross_review.md`

## Orchestration Loop

### 0. Initialize
```python
read/write goal.md from $ARGUMENTS
read state.json for current iteration
if state.status == "completed": exit with success
iteration = state.iteration + 1
PHASE_DIR = ".planning/phases/auto-${AGENT_ID}-iter-${iteration}"
mkdir -p "$PHASE_DIR"
```

Create config.json with defaults if it doesn't exist:
```bash
CONFIG_FILE="$AGENT_DIR/config.json"
if [ ! -f "$CONFIG_FILE" ]; then
  cat > "$CONFIG_FILE" <<'CONFIG'
{
  "max_iterations": 20,
  "codex_review_model": "openai/gpt-5.3-codex",
  "codex_review_model_fallback": "openai/gpt-4o",
  "codex_max_tokens": 4000,
  "per_agent_worktrees": true
}
CONFIG
fi
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
  "tests_passed": false,
  "verification_status": "pending",
  "debug_state": {
    "attempts": 0,
    "max_attempts": 3,
    "last_failure_summary": ""
  },
  "started_at": "<timestamp>",
  "last_updated": "<timestamp>"
}
```

### 0.3. Auto-Detect Test Configuration

On first iteration (or if `test_config.json` doesn't exist), scan project for test infrastructure and write `$AGENT_DIR/test_config.json`:

```bash
TEST_CONFIG="$AGENT_DIR/test_config.json"
if [ ! -f "$TEST_CONFIG" ]; then
  DETECTED="{}"
  if [ -f "pyproject.toml" ]; then
    DETECTED='{"language":"python","test_cmd":"uv run pytest","lint_cmd":"uv run ruff check .","type_cmd":"uv run mypy ."}'
  elif [ -f "pytest.ini" ] || [ -f "setup.cfg" ]; then
    DETECTED='{"language":"python","test_cmd":"pytest","lint_cmd":"ruff check .","type_cmd":"mypy ."}'
  fi
  if [ -f "package.json" ]; then
    TEST_SCRIPT=$(python3 -c "import json; p=json.load(open('package.json')); print(p.get('scripts',{}).get('test',''))" 2>/dev/null)
    if [ -n "$TEST_SCRIPT" ]; then
      DETECTED='{"language":"node","test_cmd":"npm test","lint_cmd":"npm run lint","build_cmd":"npm run build"}'
    fi
  fi
  if [ -f "Cargo.toml" ]; then
    DETECTED='{"language":"rust","test_cmd":"cargo test","lint_cmd":"cargo clippy","build_cmd":"cargo build"}'
  fi
  if [ -f "go.mod" ]; then
    DETECTED='{"language":"go","test_cmd":"go test ./...","lint_cmd":"golangci-lint run","build_cmd":"go build ./..."}'
  fi
  if [ -f "Makefile" ] && [ "$DETECTED" = "{}" ]; then
    HAS_TEST=$(grep -c '^test:' Makefile 2>/dev/null || echo "0")
    if [ "$HAS_TEST" -gt 0 ]; then
      DETECTED='{"language":"make","test_cmd":"make test"}'
    fi
  fi
  echo "$DETECTED" > "$TEST_CONFIG"
fi
```

### 0.7. Create Iteration Branch
```bash
ITER_BRANCH=$(bash "$GIT_WORKFLOW" branch-name "${AGENT_ID}-iter" "${iteration}")
WORKTREE_RESULT=$(bash "$GIT_WORKFLOW" create-worktree "$ITER_BRANCH" origin/main)
ITER_WORKTREE_PATH=$(echo "$WORKTREE_RESULT" | grep 'WORKTREE_PATH=' | cut -d= -f2)
```

Record in state.json: `branch`, `worktree_path`, `phase: "planning"`.

### 1. PLANNING PHASE
Spawn a **Task agent** (model: opus, subagent_type: general-purpose) with the full instructions from `/eng-auto-plan`. Pass it:

```
{
  agent_id: $AGENT_ID,
  agent_dir: $AGENT_DIR,
  iteration: N,
  goal: <contents of goal.md>,
  insights: <contents of insights.md>,
  prev_debug_report: <contents of .planning/phases/auto-${AGENT_ID}-iter-<N-1>/debug_report.md or "First iteration">,
  phase_dir: "$PHASE_DIR",
  test_config: <contents of test_config.json>,
  git_workflow_script: "$GIT_WORKFLOW",
  iter_worktree_path: $ITER_WORKTREE_PATH,
  iter_branch: $ITER_BRANCH
}
```

The agent writes CONTEXT.md, RESEARCH.md, PLAN.md files directly to `$PHASE_DIR`.
Returns: PLAN_COUNT, PLAN_FILES, WAVE_COUNT, CHECKER_STATUS, SUMMARY.

Update state.json phase to "codex_plan_review".

### 1.5. CODEX PLAN REVIEW

Run Codex review (type: Plan Quality). Read `$PHASE_DIR/*-PLAN.md` + `$PHASE_DIR/*-CONTEXT.md`.

Write review to `$PHASE_DIR/codex_plan_review.md`.

**If VERDICT: REVISE with CRITICAL ISSUES** (max 2 revision rounds):
- Re-spawn eng-auto-plan with `revision_context` = Codex issues
- Re-run Codex review

Update state.json phase to "executing".

### 2. EXECUTION PHASE
Spawn a **Task agent** (model: opus, subagent_type: general-purpose) with the full instructions from `/eng-auto-execute`. Pass it:

```
{
  agent_id: $AGENT_ID,
  agent_dir: $AGENT_DIR,
  iteration: N,
  goal: <contents of goal.md>,
  phase_dir: "$PHASE_DIR",
  plan_files: <list of PLAN.md paths from step 1>,
  test_config: <contents of test_config.json>,
  git_workflow_script: "$GIT_WORKFLOW",
  iter_worktree_path: $ITER_WORKTREE_PATH,
  iter_branch: $ITER_BRANCH
}
```

The agent writes SUMMARY.md, VERIFICATION.md, and execution.md to `$PHASE_DIR`.
Returns: STATUS, PLANS_EXECUTED, VERIFICATION, SUMMARY.

Update state.json phase to "codex_execution_review".

### 2.5. CODEX EXECUTION REVIEW

Run Codex review (type: Execution Fidelity). Read `$PHASE_DIR/*-SUMMARY.md` + `$PHASE_DIR/*-VERIFICATION.md`.

Write review to `$PHASE_DIR/codex_execution_review.md`.

**If VERDICT: CRITICAL:** Log issues in insights.md for next iteration.

Update state.json phase to "debugging".

### 3. VERIFY & DEBUG PHASE
Spawn a **Task agent** (model: opus, subagent_type: general-purpose) with the full instructions from `/eng-auto-debug`. Pass it:

```
{
  agent_id: $AGENT_ID,
  agent_dir: $AGENT_DIR,
  iteration: N,
  goal: <contents of goal.md>,
  insights: <contents of insights.md>,
  phase_dir: "$PHASE_DIR",
  test_config: <contents of test_config.json>,
  git_workflow_script: "$GIT_WORKFLOW",
  iter_worktree_path: $ITER_WORKTREE_PATH,
  iter_branch: $ITER_BRANCH
}
```

The agent writes debug_report.md and test_results.json to `$PHASE_DIR`.
Returns: STATUS, VERIFY_STATUS, VERIFY_MUST_HAVES, TESTS_PASSED, DEBUG_ATTEMPTS, RESULTS_TABLE, SUMMARY.

**IMPORTANT:** The `debug_report.md` MUST include a standardized Results table with markers:
```markdown
<!-- REPORT_TABLE_START -->
| Check | Status | Details |
|-------|--------|---------|
| Verification | pass/gaps | N/M must-haves met |
| Unit tests | pass/fail | {passed}/{total} ({duration}s) |
| Type check | pass/fail | {N} errors |
| Integration | pass/fail | {passed}/{total} |
| Lint | pass/fail | {N} warnings |

**Debug sub-loop:** {attempts}/{max_attempts} attempts
<!-- REPORT_TABLE_END -->
```

Update state.json phase to "codex_debug_review".

### 3.5. CODEX DEBUG REVIEW

Run Codex review (type: Debug Quality). Read `$PHASE_DIR/debug_report.md`.

Write review to `$PHASE_DIR/codex_debug_review.md`.

**If VERDICT: REVERT_NEEDED:** Log prominently in insights.md.

Update state.json phase to "cross_review".

### 4. CODEX CROSS-CUTTING REVIEW

Run Codex review (type: Cross-Cutting). Read all artifacts from `$PHASE_DIR` and per-phase Codex reviews.

Write review to `$PHASE_DIR/codex_cross_review.md`.

**If critical cross-phase inconsistencies:** Log in insights.md for next iteration.

**If no API key or API fails:** Skip and note in insights.md.

Update state.json phase to "checking_goal".

### 5. CHECK GOAL COMPLETION

Read `$PHASE_DIR/debug_report.md`. If ALL verification requirements MET and tests pass:
- Update state.json: `status: "completed"`, `goal_met: true`, `tests_passed: true`, `verification_status: "passed"`
- Write final report to `$AGENT_DIR/FINAL_REPORT.md`
- Append completion note to insights.md
- STOP the loop

If NOT all met:
- Update `debug_state` in state.json with failure summary
- Increment iteration
- Continue to next loop cycle

### 5.5. Create PR for Iteration

If there are commits on the iteration branch:

```bash
COMMIT_COUNT=$(git -C "$ITER_WORKTREE_PATH" log --oneline origin/main..HEAD | wc -l)
if [ "$COMMIT_COUNT" -gt 0 ]; then
    COMMIT_LOG=$(git -C "$ITER_WORKTREE_PATH" log --oneline origin/main..HEAD)
    PR_TITLE="auto/${AGENT_ID}/iter-${iteration}: ${goal_summary}"
    PR_BODY="## Summary
- Agent: ${AGENT_ID}
- Iteration: ${iteration}
- Goal: $(head -1 $AGENT_DIR/goal.md)

## Changes
${COMMIT_LOG}

## Test Results
- Tests: $(grep -m1 'tests_passed' $PHASE_DIR/debug_report.md 2>/dev/null || echo 'pending')
- Verification: $(grep -m1 'Status:' $PHASE_DIR/debug_report.md 2>/dev/null || echo 'pending')

## Artifacts
- Phase dir: $PHASE_DIR

> Auto-generated by eng-auto-loop agent ${AGENT_ID} iteration ${iteration}."

    PR_URL=$(bash "$GIT_WORKFLOW" create-pr "$ITER_WORKTREE_PATH" "$PR_TITLE" "$PR_BODY")
fi
```

Record `pr_url` in state.json. After PR creation, cleanup worktree:
```bash
bash "$GIT_WORKFLOW" cleanup "$ITER_WORKTREE_PATH"
```

### 5.7. Generate Iteration Report

Generate `$PHASE_DIR/REPORT.md` — the primary human-readable output for this iteration. Must contain exact numbers.

Read source files:
- `$PHASE_DIR/*-SUMMARY.md` — implementation details
- `$PHASE_DIR/*-VERIFICATION.md` — verification results
- `$PHASE_DIR/debug_report.md` — test results table
- `$PHASE_DIR/test_results.json` — structured test data (fallback)
- `state.json` — branch, PR URL, timing

```bash

# Collect wave/plan summaries from SUMMARY.md files
WAVE_TABLE=""
for summary in "$PHASE_DIR"/*-SUMMARY.md; do
  [ -f "$summary" ] || continue
  PLAN_NUM=$(basename "$summary" | grep -oP '\d+-SUMMARY' | grep -oP '^\d+')
  BUILT=$(grep -A2 'What was built\|Changes Made\|Summary' "$summary" 2>/dev/null | head -3 | tr '\n' ' ')
  FILES=$(grep -c '^[+-]' "$summary" 2>/dev/null || echo "?")
  WAVE_TABLE="${WAVE_TABLE}| ${PLAN_NUM} | Plan ${PLAN_NUM} | ${BUILT} | ${FILES} files |\n"
done

# Extract test results from debug_report.md
TEST_RESULTS=""
DEBUG_ATTEMPTS="0"
if [ -f "$PHASE_DIR/debug_report.md" ]; then
  # Primary: extract between REPORT_TABLE markers
  TEST_RESULTS=$(sed -n '/REPORT_TABLE_START/,/REPORT_TABLE_END/{/REPORT_TABLE/d;p}' "$PHASE_DIR/debug_report.md" 2>/dev/null)
  # Fallback: match table header
  if [ -z "$TEST_RESULTS" ]; then
    TEST_RESULTS=$(sed -n '/^| Check/,/^$/p' "$PHASE_DIR/debug_report.md" 2>/dev/null)
  fi
  DEBUG_ATTEMPTS=$(grep -oP '(?:sub-loop:|Attempts:)\s*\K\d+' "$PHASE_DIR/debug_report.md" 2>/dev/null | head -1 || echo "0")
fi

# Secondary fallback: parse test_results.json
if [ -f "$PHASE_DIR/test_results.json" ] && [ -z "$TEST_RESULTS" ]; then
  TEST_RESULTS=$(python3 -c "
import json
r = json.load(open('$PHASE_DIR/test_results.json'))
s = r.get('suites', {})
print('| Check | Status | Details |')
print('|-------|--------|---------|')
u = s.get('unit', {})
print(f'| Unit tests | {\"pass\" if u.get(\"status\")==\"PASSED\" else \"fail\"} | {u.get(\"passed\",0)}/{u.get(\"total\",0)} ({u.get(\"duration_seconds\",0)}s) |')
t = s.get('typecheck', {})
print(f'| Type check | {\"pass\" if t.get(\"status\")==\"PASSED\" else \"fail\"} | {t.get(\"error_count\",0)} errors |')
i = s.get('integration', {})
print(f'| Integration | {\"pass\" if i.get(\"status\")==\"PASSED\" else \"fail\"} | {i.get(\"passed\",0)}/{i.get(\"total\",0)} ({i.get(\"duration_seconds\",0)}s) |')
l = s.get('lint', {})
print(f'| Lint | {\"pass\" if l.get(\"status\")==\"PASSED\" else \"fail\"} | {l.get(\"warning_count\",0)} warnings |')
" 2>/dev/null)
fi

# Extract verification status from VERIFICATION.md
VERIFY_STATUS="pending"
MUST_HAVES="0/0"
for vfile in "$PHASE_DIR"/*-VERIFICATION.md; do
  [ -f "$vfile" ] || continue
  VERIFY_STATUS=$(grep -oP '(?:Status|Verdict):\s*\K\S+' "$vfile" 2>/dev/null | head -1 || echo "unknown")
  MH_PASS=$(grep -c '✅\|PASS\|met' "$vfile" 2>/dev/null || echo "0")
  MH_TOTAL=$(grep -c 'must.have\|requirement' "$vfile" 2>/dev/null || echo "0")
  MUST_HAVES="${MH_PASS}/${MH_TOTAL}"
done
```

Write the report to `$PHASE_DIR/REPORT.md`:
```markdown
## Engineering Report: {AGENT_ID} — Iteration {N}
**Branch:** auto/{AGENT_ID}-iter-{N} | **PR:** [#{pr_num}]({pr_url}) | **Duration:** {elapsed}

### Task Description
{From goal.md — full goal and verify requirements}

### Implementation
| Wave | Plan | What was built | Files changed |
|------|------|----------------|---------------|
{populated from SUMMARY.md files — one row per plan}

### Results
| Check | Status | Details |
|-------|--------|---------|
| Verification | {pass/gaps} | {N/M must-haves met} |
| Unit tests | {pass/fail} | {passed}/{total} ({duration}) |
| Type check | {pass/fail} | {error count} errors |
| Integration | {pass/fail} | {passed}/{total} |
| Lint | {pass/fail} | {warning count} warnings |

**Debug sub-loop:** {attempts}/{max_attempts} attempts

### Checks / Feedback
{Full content from VERIFICATION.md}

### PR Link
[PR #{num}: {title}]({url})
```

Extraction priority for Results table:
1. **Primary:** Between `REPORT_TABLE_START` / `REPORT_TABLE_END` markers in debug_report.md
2. **Fallback:** Match `| Check` table header in debug_report.md
3. **Secondary:** Parse test_results.json for precise counts

### 6. CONTEXT MANAGEMENT

After every iteration, write a checkpoint:

**Update insights.md** — append iteration summary:
```markdown
### Iteration <N> — <timestamp>
- Status: <SUCCESS/PARTIAL/FAILED>
- Phase dir: `$PHASE_DIR`
- Plans executed: <count>
- Changes made: <brief list>
- Tests: <passed/failed — summary>
- Key finding: <most important insight>
- Goal progress: <X of Y requirements met>
- Branch: <branch name>
- PR: <PR URL or "pending">
- Next action: <focus for next iteration>
```

**Update state.json** with current iteration, timestamp, and `iterations_summary` array:
```json
{
  "iterations_summary": [
    {
      "iteration": N,
      "branch": "$ITER_BRANCH",
      "pr_url": "$PR_URL",
      "status": "SUCCESS/PARTIAL/FAILED",
      "tests_passed": true,
      "goal_progress": "X of Y",
      "phase_dir": "$PHASE_DIR"
    }
  ]
}
```

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

When goal is met (or max iterations reached), write `$AGENT_DIR/FINAL_REPORT.md`:

```markdown
# Autonomous Engineering Loop Final Report
Completed: <timestamp>

## Goal
<the goal>

## Result: [GOAL MET / PARTIALLY MET / NOT MET]

## Iterations: <N> of <max>

## Summary
<2-3 paragraph summary>

## Verification Requirements
| Requirement | Status | Evidence | Iteration |
|------------|--------|----------|-----------|
| Req 1 | MET/NOT MET | <evidence> | <N> |

## Test Results
| Iteration | Tests Passed | Failures | Debug Attempts |
|-----------|-------------|----------|----------------|
| 1 | Yes/No | <count> | <count> |

## Key Insights
<bulleted list>

## Changes Applied
<cumulative list across all iterations>

## Errors Encountered and Resolved
<list>

## Recommendations
<what to do next>
```

## Recovery from Context Loss

If invoked and state.json shows `status: "running"`:

1. Read iteration number from state.json
2. Set `PHASE_DIR=".planning/phases/auto-${AGENT_ID}-iter-${iteration}"`
3. Determine resume point from artifacts in `$PHASE_DIR`:
   - No CONTEXT.md → start from planning
   - CONTEXT.md exists, no PLAN.md → resume planning (researcher/planner)
   - PLAN.md exists, no SUMMARY.md → resume from execution
   - SUMMARY.md exists, no VERIFICATION.md → resume execution (verifier)
   - VERIFICATION.md exists, no debug_report.md → resume from debug
   - debug_report.md exists → check Codex reviews, then start next iteration
4. Check for codex review files in `$PHASE_DIR` to determine if post-phase reviews ran
5. Read insights.md for accumulated context
6. Continue from resume point

### Worktree Recovery
If `state.json` has `worktree_path` but directory doesn't exist:
- Check if branch exists: `git branch --list "$ITER_BRANCH"`
- If branch exists, re-create worktree: `git-workflow.sh create-worktree "$ITER_BRANCH"`
- If branch doesn't exist, start fresh from origin/main
