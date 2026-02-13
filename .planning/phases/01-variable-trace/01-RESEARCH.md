# Phase 01: Variable Trace Audit — RESEARCH

## Methodology
Traced every `${VARIABLE}` in the exps-auto-loop.md step 5.7 REPORT.md template back to its extraction code (grep/python) and then to the source data format in exps-auto-analyze.md (analysis.md bridge) or exps-auto-execute.md (execution.md bridge).

## Variable-by-Variable Trace

### Working Variables (10 of 21)
| Variable | Extraction | Source Format | Status |
|----------|-----------|---------------|--------|
| `AGENT_ID` | Loop args | Parsed from `$ARGUMENTS` | OK |
| `iteration` | Loop state | `state.json` | OK |
| `WANDB_PROJECT` | Hardcoded | `"cohere/multiagent-debate-rl"` | OK |
| `ELAPSED` | Python datetime | `state.json.started_at` | OK |
| `GOAL` | `cat goal.md` | `$AUTONOMOUS_DIR/goal.md` | OK |
| `PR_URL` | Python json | `state.json.pr_url` | OK |
| `PR_NUM` | `grep -oE '[0-9]+$'` | Trailing digits of PR_URL | OK |
| `DEBATE_REWARD` | `grep -oP 'Debate mean_reward:\s*\K[\d.]+'` | analysis.md `- Debate mean_reward: X.XX [NORMAL/ANOMALOUS]` | OK |
| `BASELINE_REWARD` | Same pattern for Baseline | analysis.md `- Baseline mean_reward: X.XX` | OK |
| `PREV_DELTA` | `grep -A 5 "Corrective Plan\|..."` | Previous analysis.md sections | OK |

### Metrics with Format-Sensitive Grep (4 of 21)
| Variable | Grep Pattern | Source Format | Status |
|----------|-------------|---------------|--------|
| `LOSS_TREND` | `grep -oP 'Loss trend:\s*\K\S+'` | `- Loss trend: DECREASING` | OK (captures first word) |
| `GRAD_NORM` | `grep -oP 'Grad norm:\s*\K\S+'` | `- Grad norm: STABLE` | OK |
| `KL_DIV` | `grep -oP 'KL divergence:\s*\K\S+'` | `- KL divergence: BOUNDED` | OK |
| `REWARD_TREND` | `grep -oP 'Reward trend:\s*\K\S+'` | `- Reward trend: IMPROVING` | OK |

### BROKEN Variables (7 of 21)
| # | Variable | Bug | Severity |
|---|----------|-----|----------|
| 1 | `CONFIG_TABLE` | `$WORKING_DIR` undefined in step 5.7 — importlib.util fails | CRITICAL |
| 2 | `CONFIG_TABLE` | Worktree cleaned up in 5.5 before report reads configs in 5.7 | HIGH |
| 3 | `WANDB_DEBATE_URL` | Grep for `https://wandb.ai/\S+` in analysis.md — analysis.md never writes W&B URLs | HIGH |
| 4 | `WANDB_BASELINE_URL` | Same — no data source | HIGH |
| 5 | `REWARD_CHART` | `ls ~/sweep_logs_*/wandb/*reward*.png` — W&B doesn't save PNGs locally | MEDIUM |
| 6 | `LOSS_CHART` | Same — no PNGs exist at that path | MEDIUM |
| 7 | Transcripts | JSON schema assumptions (key names) may not match SWEEP output | LOW |
