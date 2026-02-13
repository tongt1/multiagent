# Project: exps-auto-loop Report Template Audit

## Goal
Audit and fix exps-auto-loop.md report template data flow. Verify all grep patterns match actual analysis.md output. Fix figure embedding paths. Ensure transcript extraction works.

## Scope
- `~/.claude/commands/exps-auto-loop.md` — step 5.7 report template
- `~/.claude/commands/exps-auto-analyze.md` — analysis bridge format (data source)
- `~/.claude/commands/exps-auto-execute.md` — execution bridge format (data source)

## Agent
- AGENT_ID: bravo
- Worktree: `/mnt/data/terry/worktrees/auto/bravo-audit`
- Branch: `auto/bravo-audit`
