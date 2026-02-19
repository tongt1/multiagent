# Codebase Concerns

**Analysis Date:** 2026-02-14

## Tech Debt

**Kjobs/Apiary Integration Stub:**
- Issue: Complete executor implementation stubbed out with NotImplementedError
- Files: `/home/terry_tong_cohere_com/reward-training/src/infrastructure/distributed_executor.py` (lines 571-616, 619-704)
- Impact: Cannot submit distributed training jobs to internal Cohere infrastructure; limits deployment options to local/Kubernetes/Ray only
- Fix approach: Wait for kjobs/apiary public API documentation and implement all methods (submit_job, get_status, cancel_job, wait_for_completion, list_jobs)

**BEE Evaluation Graceful Degradation:**
- Issue: BEE/Hive integration returns stub results when unavailable instead of failing fast
- Files: `/home/terry_tong_cohere_com/reward-training/src/evaluation/bee_eval.py` (lines 35-84, 238-251)
- Impact: Silent failures in evaluation pipeline; results contain zeros but appear successful (status: "bee_not_available")
- Fix approach: Either make BEE a required dependency with hard failure, or add explicit validation that prevents downstream processing of stub results

**W&B Data Source Not Implemented (Streamlit Viewer):**
- Issue: Overview page hardcoded to return empty results for W&B source type
- Files: `/home/terry_tong_cohere_com/reward-training/streamlit_viewer/pages/overview.py` (line 66)
- Impact: Cannot view training metrics directly from W&B API; forces users to export Parquet files manually
- Fix approach: Implement W&B API queries similar to data_loader.py patterns (lines 123-165)

**Silent Exception Handling:**
- Issue: Multiple locations catch broad exceptions and pass/return empty without logging
- Files:
  - `/home/terry_tong_cohere_com/reward-training/src/data/training_export.py` (lines 56-58, 291)
  - `/home/terry_tong_cohere_com/reward-training/streamlit_viewer/lib/data_loader.py` (lines 97, 179-180)
  - `/home/terry_tong_cohere_com/reward-training/src/evaluation/code_executor.py` (lines 42-44)
  - `/home/terry_tong_cohere_com/reward-training/src/evaluation/math_verifier.py` (lines 80, 87-88, 132)
- Impact: Debugging failures is difficult; errors swallowed without trace
- Fix approach: Add logger.exception() or logger.warning() in all exception handlers; only catch specific exceptions

**Large Monolithic Files:**
- Issue: Several files exceed 500 lines with complex logic
- Files:
  - `/home/terry_tong_cohere_com/reward-training/src/cli/main.py` (916 lines)
  - `/home/terry_tong_cohere_com/reward-training/src/infrastructure/distributed_executor.py` (707 lines)
  - `/home/terry_tong_cohere_com/reward-training/src/training/train_marti.py` (620 lines)
  - `/home/terry_tong_cohere_com/reward-training/src/orchestration/experiment_runner.py` (569 lines)
  - `/home/terry_tong_cohere_com/reward-training/src/cli/runner.py` (566 lines)
- Impact: Hard to navigate, test, and maintain; multiple responsibilities in single files
- Fix approach: Extract subcommand handlers from CLI files into separate modules; split executor classes by backend type; separate training loop from config/logging setup

**OpenRLHF Tight Coupling:**
- Issue: Training infrastructure assumes OpenRLHF framework but implementation is MARTI-only
- Files:
  - `/home/terry_tong_cohere_com/reward-training/src/training/training_config.py` (OpenRLHFConfig class)
  - `/home/terry_tong_cohere_com/reward-training/src/training/train_marti.py` (references openrlhf_config but doesn't use the framework)
  - `/home/terry_tong_cohere_com/reward-training/src/infrastructure/ray_training_executor.py` (line 96: container includes "openrlhf")
- Impact: Misleading config names; future OpenRLHF integration may conflict with current MARTI implementation
- Fix approach: Rename OpenRLHFConfig to MARTITrainingConfig or GenericRLConfig; decouple from external framework assumptions

## Known Bugs

**Hardcoded Sleep Durations:**
- Symptoms: Fixed 10-second polling intervals regardless of operation type
- Files:
  - `/home/terry_tong_cohere_com/reward-training/src/cli/main.py` (lines 811, 874)
  - `/home/terry_tong_cohere_com/reward-training/src/cli/runner.py` (line 518)
- Trigger: Any training job status polling loop
- Workaround: None; users must wait full interval even for fast operations
- Fix: Add configurable poll_interval parameter; use exponential backoff

**Exponential Backoff Capped Incorrectly:**
- Symptoms: Retry backoff formula uses min() which caps but doesn't handle boundary correctly
- Files: `/home/terry_tong_cohere_com/reward-training/src/orchestration/experiment_runner.py` (line 272)
- Trigger: Stage failures triggering retry logic
- Workaround: Works as intended but formula is unnecessarily complex
- Fix: Simplify to: backoff = min(30 * (2 ** attempt), 600)

## Security Considerations

**.env File Present in Repository:**
- Risk: Environment file exists in working directory; could leak secrets if accidentally committed
- Files: `/home/terry_tong_cohere_com/reward-training/.env` (exists but in .gitignore)
- Current mitigation: .gitignore includes `.env` pattern (line 24)
- Recommendations: Add pre-commit hook to block commits containing API keys; document required env vars in .env.example file instead

**Code Execution Without Sandboxing:**
- Risk: Code executor runs untrusted Python with resource limits but no containerization
- Files: `/home/terry_tong_cohere_com/reward-training/src/evaluation/code_executor.py` (lines 35-44)
- Current mitigation: Memory limit (256MB) and CPU time limit (5s) via resource module
- Recommendations: Use Docker/gVisor for execution isolation; validate code against blocklists for dangerous imports (os, subprocess, socket)

**No API Key Rotation Mechanism:**
- Risk: LiteLLM/Cohere API keys loaded from environment but never refreshed
- Files: All files importing from `src.infrastructure.llm_client`
- Current mitigation: None
- Recommendations: Implement key rotation support; add expiration monitoring; use secret management service (AWS Secrets Manager, Vault)

## Performance Bottlenecks

**Synchronous File I/O in Training Loop:**
- Problem: JSONL trajectory loading blocks training initialization
- Files: `/home/terry_tong_cohere_com/reward-training/src/data/training_export.py` (read_trajectories_from_jsonl)
- Cause: Single-threaded file read for potentially large trajectory files
- Improvement path: Use asyncio file I/O or memory-mapped files; stream trajectories lazily instead of loading all at once

**Polling-Based Job Status Checks:**
- Problem: Busy-wait loops with fixed sleep intervals waste resources
- Files:
  - `/home/terry_tong_cohere_com/reward-training/src/cli/main.py` (training status loops)
  - `/home/terry_tong_cohere_com/reward-training/src/infrastructure/distributed_executor.py` (line 464)
- Cause: No event-driven job status notifications
- Improvement path: Use Kubernetes watch API for job events; implement webhooks for Ray job state changes

**String-Based Log Parsing:**
- Problem: Extracting metrics from logs instead of structured data
- Files: `/home/terry_tong_cohere_com/reward-training/streamlit_viewer/lib/debate_parser.py` (regex-based parsing)
- Cause: Legacy log format from original implementation
- Improvement path: Switch to structured JSON logging; parse from JSON instead of regex

## Fragile Areas

**Async/Await Inconsistency:**
- Files: 15 files mix async and sync code without clear boundaries
  - `/home/terry_tong_cohere_com/reward-training/src/orchestration/batch_executor.py`
  - `/home/terry_tong_cohere_com/reward-training/src/infrastructure/distributed_executor.py`
  - `/home/terry_tong_cohere_com/reward-training/src/cli/runner.py`
  - (112 async occurrences across 15 files)
- Why fragile: Event loop management errors; easy to accidentally call async functions synchronously
- Safe modification: Always use asyncio.run() for top-level async entry points; never mix asyncio.run() in nested calls
- Test coverage: 30 out of 52 test files use mocks/patches; async tests may not catch runtime loop issues

**Role Mask Computation:**
- Files: `/home/terry_tong_cohere_com/reward-training/src/training/wandb_enrichment/role_mask_computer.py` (449 lines)
- Why fragile: Complex regex-based parsing of debate transcripts; returns None on parse failure (lines 311-334 in tests)
- Safe modification: Add extensive logging at each parse step; validate input format before processing
- Test coverage: Test file is 535 lines with edge cases but still has TODO for handling malformed inputs

**Multi-Model Configuration:**
- Files: `/home/terry_tong_cohere_com/reward-training/src/training/training_config.py` (multi-model sweep logic)
- Why fragile: Dynamic config generation for multiple models; easy to create invalid combinations
- Safe modification: Use Pydantic validators for all config fields; test all combinations with dry-run mode
- Test coverage: test_multi_model_config.py and test_multi_model_sweep.py exist but only 36+32 assertions total

## Scaling Limits

**Single-Machine Training Assumption:**
- Current capacity: MARTI training runs on single GPU (train_marti.py uses torch.device("cuda"))
- Limit: Cannot utilize multi-node clusters without OpenRLHF/Ray integration
- Scaling path: Implement distributed data-parallel training; integrate with Ray Train or Deepspeed

**In-Memory Trajectory Storage:**
- Current capacity: All trajectories loaded into memory at once
- Limit: Large-scale experiments (>100K trajectories) will OOM
- Scaling path: Implement streaming dataset with lazy loading; use PyTorch IterableDataset instead of list

**Synchronous Batch Processing:**
- Current capacity: Batch executor processes problems sequentially
- Limit: ~10-50 problems per hour depending on model latency
- Scaling path: Already has async infrastructure; increase concurrency limits in batch_executor.py

## Dependencies at Risk

**Poetry vs UV Migration:**
- Risk: poetry.lock exists but pyproject.toml uses [dependency-groups] which is UV-specific syntax
- Impact: Unclear which package manager is canonical; may cause dependency resolution conflicts
- Migration plan: Choose one package manager; remove the other's lockfile; update CI/CD scripts

**Pinned Major Versions Only:**
- Risk: Dependencies use >= constraints without upper bounds (e.g., "pydantic>=2.12.5")
- Impact: Breaking changes in minor/patch releases could break builds
- Migration plan: Use poetry's caret syntax (^2.12.5) or explicit ranges; enable Dependabot for automated updates

**Optional Dependencies Not Isolated:**
- Risk: training/viewer/distributed extras may have version conflicts
- Impact: Installing all extras simultaneously might fail
- Migration plan: Test each optional dependency group in isolation; add constraint pins for overlapping deps

## Missing Critical Features

**No Checkpoint Validation:**
- Problem: Training saves checkpoints but never validates they load correctly
- Blocks: Cannot detect corrupted checkpoints until inference time
- Priority: High — silent failures waste GPU hours

**No Experiment Reproducibility Manifest:**
- Problem: Config saved to W&B but no standalone reproducibility file
- Blocks: Cannot re-run experiments without W&B access; no git hash tracking
- Priority: Medium — impacts scientific rigor and debugging

**No Training Job Cancellation UI:**
- Problem: CLI can cancel jobs but no bulk cancellation or dependency-aware cancellation
- Blocks: Cannot easily stop multi-stage experiments; must cancel each job individually
- Priority: Low — workaround is manual scripting

## Test Coverage Gaps

**Integration Test Coverage Low:**
- What's not tested: End-to-end pipeline from data generation → training → evaluation
- Files: No test file for complete workflow; only unit tests and component integration tests
- Risk: Component tests pass but full pipeline fails on real data
- Priority: High

**Distributed Executor Only Mocked:**
- What's not tested: Real Kubernetes/Ray job submission
- Files: `/home/terry_tong_cohere_com/reward-training/tests/test_distributed_executor.py` (485 lines but all mocked)
- Risk: Real deployments may fail due to API mismatches or authentication issues
- Priority: High — blocks production deployment confidence

**W&B Logging Never Validated:**
- What's not tested: Actual W&B metric upload and schema validation
- Files: wandb_enrichment module tests mock wandb.log calls
- Risk: Metrics logged with wrong schema; W&B dashboards break silently
- Priority: Medium — workaround is manual dashboard verification

**LLM Client Error Handling:**
- What's not tested: API rate limits, timeouts, authentication failures
- Files: `/home/terry_tong_cohere_com/reward-training/src/infrastructure/llm_client.py` (no dedicated test file)
- Risk: Production failures on rate limits or quota exhaustion
- Priority: Medium — tenacity should handle this but untested

**CooperBench Classifier Edge Cases:**
- What's not tested: Malformed patches, unicode handling, extremely long diffs
- Files: `/home/terry_tong_cohere_com/reward-training/cooperbench-eval/src/classifiers/` (31 Python files)
- Risk: Classifier failures on real agent outputs with unexpected formatting
- Priority: Low — current tests cover main patterns

---

*Concerns audit: 2026-02-14*
