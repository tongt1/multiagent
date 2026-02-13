# Project: Fix eng-auto Issues 5-7 + Warning 2

## Goal
Fix consistency issues across eng-auto-execute.md, eng-auto-loop.md, and eng-auto-debug.md to ensure the autonomous engineering loop pipeline is coherent.

## Fixes Applied
1. **Issue 5 (FIX 1):** Remove duplicate REPORT.md generation from eng-auto-execute.md — Phase F "Generate Iteration Report" removed, REPORT_PATH removed from Return section. Report generation is solely in eng-auto-loop.md Step 5.7.
2. **Issue 6 (FIX 2):** Fix Results table extraction in eng-auto-loop.md Step 5.7 — use REPORT_TABLE_START/END markers written by eng-auto-debug, with fallback to `| Check` header match.
3. **Issue 7 (FIX 3):** Align debug spawn parameters in eng-auto-loop.md Step 3 with eng-auto-debug.md's Input section — removed `insights`, `execution_file`, `test_config`; added `plan_files`.
4. **Warning 2 (FIX 4):** Add config.json creation with defaults in eng-auto-loop.md Step 0 initialization — max_iterations=20, codex_review_model, codex_review_model_fallback, codex_max_tokens, per_agent_worktrees.

## Status
All fixes applied and verified.
