# Roadmap: Fix eng-auto Issues 5-7 + Warning 2

## Phase 1: Remove duplicate report generation (Issue 5) - DONE
- Remove Phase F "Generate Iteration Report" from eng-auto-execute.md
- Remove REPORT_PATH from Return section
- Renumber Phase G → Phase F

## Phase 2: Fix REPORT_TABLE extraction (Issue 6) - DONE
- Update eng-auto-loop.md Step 5.7 to use REPORT_TABLE_START/END markers
- Add fallback to `| Check` header match
- Add secondary fallback to test_results.json parsing

## Phase 3: Align debug spawn params (Issue 7) - DONE
- Remove `insights`, `execution_file`, `test_config` from Step 3 spawn block
- Add `plan_files` parameter
- Verify alignment with eng-auto-debug.md Input section

## Phase 4: Add config.json creation (Warning 2) - DONE
- Add config.json creation block in Step 0 initialization
- Include defaults: max_iterations, codex_review_model, codex_review_model_fallback, codex_max_tokens, per_agent_worktrees
