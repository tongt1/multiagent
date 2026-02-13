---
phase: 02-fix-data-flow
plan: 02
type: implementation
wave: 1
depends_on: []
files_modified:
  - ~/.claude/commands/exps-auto-loop.md
  - ~/.claude/commands/exps-auto-analyze.md
autonomous: true
must_haves:
  - $WORKING_DIR is defined before use in step 5.7
  - Worktree cleanup happens AFTER report generation (step 5.8, not 5.5)
  - WANDB_DEBATE_URL has 3-tier fallback (analysis→execution→dashboard)
  - WANDB_BASELINE_URL has same fallback
  - REWARD_CHART uses find in wandb media dir, not ls with wrong glob
  - LOSS_CHART same
  - analysis.md bridge format includes W&B Run URLs section
---

# Phase 02: Fix Data Flow Bugs

## Tasks

### Task 1: Define $WORKING_DIR in step 5.7
Add `WORKING_DIR="${ITER_WORKTREE_PATH:-$HOME/multiagent}"` with directory existence fallback before config path definitions.

### Task 2: Move worktree cleanup after report
Remove `git-workflow.sh cleanup` from step 5.5. Add new step 5.8 with cleanup after report generation.

### Task 3: Fix W&B URL extraction
Replace single grep against analysis.md with 3-tier fallback:
1. Grep `Debate:\s*\Khttps://wandb\.ai/\S+` from analysis.md W&B Run URLs section
2. Fallback: any `https://wandb.ai/` URL from analysis.md or execution.md
3. Final fallback: dashboard URL `https://wandb.ai/${WANDB_PROJECT}`

### Task 4: Fix chart PNG paths
Replace `ls ~/sweep_logs_*/wandb/*.png` with:
1. `find "$WORKING_DIR/wandb/" -path "*/media/images/*reward*" -name "*.png"`
2. Fallback: `find ~/sweep_logs_*/ -name "*reward*.png"`

### Task 5: Add W&B URLs to analysis.md bridge format
In exps-auto-analyze.md:
- Add URL extraction instructions to Agent 2
- Add `## W&B Run URLs` section to Phase F bridge template

### Task 6: Add step 5.8 worktree cleanup
New section after step 5.7 with conditional cleanup.
