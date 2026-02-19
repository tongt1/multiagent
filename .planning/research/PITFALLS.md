# Pitfalls Research

**Domain:** CooperBench paper reproduction -- LLM agent benchmark evaluation with Docker sandboxes, statistical analysis, LLM-as-judge classification, single-model vs multi-model comparability
**Researched:** 2026-02-14
**Confidence:** HIGH (verified against paper methodology via arXiv:2601.13295, existing codebase analysis, and domain literature)

## Critical Pitfalls

### Pitfall 1: Difficulty Score Mismatch Against Paper Methodology

**What goes wrong:**
The paper defines task difficulty as `d(t) = 1 - (1/|M|) * sum(Solo_m(t))`, averaging solo success rates across ALL models in their evaluation set. When reproducing with a single model (Command A), the difficulty score is computed from one model's solo performance only. This produces a different difficulty ranking than the paper's -- tasks that are hard for Command A may be easy for GPT-5 or Claude Sonnet 4.5, and vice versa. Figure 6 (success rate vs. difficulty) becomes non-comparable because the x-axis itself differs.

**Why it happens:**
The difficulty formula averages over the model set M. With |M|=1, d(t) degenerates to `1 - Solo_CommandA(t)`, which is a model-specific difficulty measure rather than a task-intrinsic measure. The paper's difficulty reflects consensus difficulty across multiple frontier models.

**How to avoid:**
1. Compute two separate difficulty axes: (a) paper-reported difficulty if the authors release per-pair scores, and (b) Command A-specific difficulty from solo runs.
2. For Figure 6 reproduction, use the paper's difficulty bucketing if available; for our own analysis, clearly label the x-axis as "Difficulty (Command A solo)" not "Difficulty (CooperBench)."
3. Check `gold_conflict_report.json` -- the gold merge conflict rate per pair may serve as a model-independent difficulty proxy.
4. Consider using gold patch complexity (lines changed, files touched, conflict rate) as an alternative difficulty measure that does not depend on model performance.

**Warning signs:**
- Difficulty distribution looks markedly different from paper's Figure 6 x-axis spread.
- Buckets have very uneven sample sizes (some empty, some over-represented).
- Correlation between our difficulty and paper difficulty is low.

**Phase to address:**
Data preparation and analysis pipeline phase -- must decide difficulty methodology before running analysis.

---

### Pitfall 2: Docker Container Failures Silently Counted as Test Failures

**What goes wrong:**
When running 300+ evaluations (100 pairs x 3 settings: solo, coop-with-comm, coop-no-comm) in Docker sandboxes, container-level failures (OOM kills, Docker daemon errors, network timeouts, disk space exhaustion, image pull failures) get classified as "both features failed" rather than "infrastructure error." This inflates the failure rate and deflates the success rate. The existing evaluator in `src/evaluation/cooperbench/evaluator.py` catches exceptions and returns `both_passed=False` with an error string, but downstream analysis may not distinguish infrastructure errors from genuine test failures.

**Why it happens:**
The evaluator's exception handler returns a `CooperBenchEvalResult` with `both_passed=False` and an error message, but the reward computation (`compute_cooperbench_reward`) treats this identically to a genuine test failure (reward=0.0). The CooperBench paper's `sandbox.py` also catches exceptions broadly. At 300+ runs with 4 parallel containers, Docker resource exhaustion is likely.

**How to avoid:**
1. Tag every `CooperBenchEvalResult` with a status field: `pass`, `fail`, `infra_error`, `timeout`. The existing `error` field is not sufficient because it is a free-text string.
2. Implement a retry policy: infra_errors get retried up to 2 times with exponential backoff. Genuine test failures are never retried.
3. Monitor Docker daemon health between runs: check `docker info`, available disk, memory pressure.
4. Set per-container resource limits explicitly (`--memory=4g`, `--cpus=2` -- already in evaluator) AND set a system-wide limit on concurrent containers.
5. Log container exit codes separately from test exit codes. Docker OOM kill = exit code 137; distinguish from pytest failure exit code 1.
6. Post-run audit: count total runs, infra errors, genuine failures. If infra_error rate > 5%, investigate before computing metrics.

**Warning signs:**
- Sudden clusters of failures (10+ consecutive) suggest Docker daemon issues, not agent quality.
- Error messages containing "Cannot connect to the Docker daemon", "no space left on device", "OCI runtime", "Killed" (OOM).
- Pass rate is suspiciously lower than expected (e.g., < 5% when paper reports 25%).
- Execution times for "failed" runs are very short (< 5 seconds) or exactly at the timeout boundary.

**Phase to address:**
Evaluation infrastructure phase -- must be built before running benchmark at scale.

---

### Pitfall 3: Wilson CI Computation With Degenerate Buckets

**What goes wrong:**
The paper uses 10 equal-width difficulty buckets spanning [0,1] and computes Wilson CIs per bucket. With 100 pairs (lite subset), each bucket gets approximately 10 pairs. Some buckets will have 0/10 or 10/10 success rates, producing Wilson CIs that are technically valid but practically misleading for visualization. Worse, some buckets may be empty (0 pairs), causing division-by-zero or requiring special handling. The AUC computed via trapezoidal integration over these 10 sparse buckets will have high variance.

**Why it happens:**
The difficulty distribution is not uniform -- tasks cluster at certain difficulty levels. With 100 pairs from the lite subset and 10 buckets, the expected count per bucket is 10, but actual counts will vary from 0 to 20+. Wilson CIs at n=5 with p=0 are [0.0, 0.434], which is wide enough to obscure any meaningful signal.

**How to avoid:**
1. Use adaptive bucket boundaries (quantile-based) instead of equal-width, ensuring each bucket has at least 8-10 samples.
2. Report the number of pairs per bucket alongside the CI -- readers need to judge reliability.
3. For AUC computation, handle empty buckets by linear interpolation from neighbors, not by treating them as 0% success.
4. Consider using the full 652-pair dataset instead of the 100-pair lite subset for Figure 6, if compute budget allows. This increases per-bucket sample size to ~65.
5. Use `statsmodels.stats.proportion.proportion_confint(method='wilson')` for correct Wilson CI computation -- do not hand-roll the formula.
6. Validate against known edge cases: n=0 (empty bucket), n=1 with k=0, n=1 with k=1.

**Warning signs:**
- Any bucket with n < 5 pairs.
- Any bucket with exactly 0% or 100% success rate AND n < 15.
- AUC values that change dramatically (> 0.1) when removing a single bucket.
- CIs wider than 0.4 for any bucket.

**Phase to address:**
Statistical analysis phase -- implement before generating figures.

---

### Pitfall 4: LLM-as-Judge Classification Noise Producing Unreliable Figure 5

**What goes wrong:**
The error classification pipeline (`cooperbench-eval/src/llm_judge/`) uses Cohere's Command R Plus to classify 10 failure modes from agent transcripts. LLM classifiers have inherent stochasticity: the same transcript may be classified differently across runs. At temperature=0.0, outputs are mostly deterministic but not guaranteed identical across API calls. With only 100 pairs and 10 classifiers, a 10% misclassification rate means ~100 incorrect labels, which can shift the failure mode frequency chart (Figure 5 equivalent) by several percentage points.

**Why it happens:**
LLM-as-judge reliability research shows that even strong models exhibit self-inconsistency on classification tasks (documented in "Rating Roulette: Self-Inconsistency in LLM-As-A-Judge," EMNLP 2025). The CooperBench paper itself notes that manual annotation validated their LLM-as-judge but kept it "descriptive rather than inferential." Our pipeline uses a different judge model (Command R Plus vs. whatever the paper used), adding a model-dependent bias.

**How to avoid:**
1. Run each LLM classification 3 times and use majority vote. This reduces random misclassification by ~70% at 3x the API cost.
2. Establish a ground truth on a small sample: manually annotate 20 transcripts for all 10 failure modes, then compute Cohen's kappa between LLM and human labels. Report this alongside Figure 5.
3. Use temperature=0.0 strictly (already set in `base.py`) and pin the model version to avoid API-side model updates changing results mid-run.
4. Report confidence scores from the LLM alongside detection rates. High-confidence detections are more reliable.
5. For low-prevalence categories (< 5% baseline: Dependency Access at 1.7%, Placeholder Misuse at 1.5%, Parameter Flow at 1.3%, Timing Dependency at 1.1%), a single misclassification can double the reported rate. Flag these as "low-confidence estimates" in the figure.
6. Consider a two-stage pipeline: heuristic pre-filter (existing `src/classifiers/` rules) followed by LLM judge only for ambiguous cases. This reduces cost and improves consistency.

**Warning signs:**
- Re-running classification produces different failure rates (> 2% delta on any category).
- Low-prevalence categories show rates wildly different from paper baselines.
- The LLM returns unparseable JSON (tracked in `parse_json_response` fallback).
- API rate limits cause skipped classifications (tracked as `skipped=True` in results).

**Phase to address:**
Classification and analysis phase -- must establish reliability before reporting results.

---

### Pitfall 5: Merge Conflict Detection Conflated With Implementation Failure

**What goes wrong:**
In coop mode, patches from two agents are merged before testing. A merge conflict means the patches touch overlapping code. The paper distinguishes merge conflicts (coordination failure) from test failures (implementation failure). Our evaluator (`evaluator.py` lines 77-97) returns `merge_status="failed"` with `both_passed=False` when merging fails, but this is a fundamentally different failure mode than "patches merged cleanly but tests failed." If these are not distinguished, Figure 4's merge conflict analysis (panel b) will be wrong, and overall coop success rates will be misleadingly low.

**Why it happens:**
The current merge pipeline tries naive merge, then union merge, and if both fail, returns a combined error result. The `test_merged` function in the paper's `sandbox.py` has the same structure but records the merge status separately from test results. Our pipeline conflates them at the reward computation level: `compute_cooperbench_reward` returns 0.0 for both merge failures and test failures.

**How to avoid:**
1. Track merge outcomes as a separate dimension in results: `{merge_clean, merge_union, merge_failed}` x `{tests_pass, tests_fail}`.
2. For Figure 4 panel (b), compute merge conflict rate = `merge_failed / total_coop_runs`. Do not include test failures in this count.
3. When computing coop success rate, clearly state whether merge failures are counted as failures (they should be -- the task was not solved) but report them separately.
4. Implement the paper's exact merge strategy: create branches, apply patches, attempt naive merge, fall back to union. The paper's `sandbox.py` does this inside the sandbox (git operations in-container). Our `evaluator.py` does a simplified version outside the container. These may produce different merge outcomes for the same patches.
5. Run the merge step IN the sandbox (inside Docker) using the actual repository state, not in a temporary directory. The paper's approach uses the full repo context.

**Warning signs:**
- Merge failure rate is significantly higher or lower than the paper's ~76.5% gold conflict rate (from `gold_conflict_report.json`).
- All coop failures are classified as merge failures (suggesting implementation bugs in the merge pipeline).
- Merge "succeeds" but produces garbage patches (concatenation without conflict resolution).

**Phase to address:**
Evaluation infrastructure phase -- merge logic must be correct before running at scale.

---

### Pitfall 6: Agent SDK Mismatch -- Our Pipeline vs Paper's OpenHands Agent

**What goes wrong:**
The CooperBench paper uses the OpenHands agent SDK (`repos/CooperBench/src/cooperbench/agents/openhands_agent_sdk/`) with SWE-agent tools, a git connector for collaboration, and a messaging system. Our pipeline (`src/evaluation/cooperbench/pipeline.py`) uses a simplified debate-as-cooperation loop with LLM API calls and prompt-based patch generation. This is a fundamentally different agent architecture: the paper's agents can browse files, run commands, use tools, and interact with a real git repository. Our agents generate patches from a single LLM call. Results are not directly comparable.

**Why it happens:**
Reproducing the full OpenHands agent SDK pipeline requires integrating with the OpenHands runtime, which has its own sandbox, tool system, and message passing infrastructure. Our pipeline was designed as a simpler debate framework. The gap is architectural, not just parametric.

**How to avoid:**
1. Acknowledge this explicitly in any written analysis: "We use a prompt-based patch generation approach rather than the paper's interactive agent SDK. Results reflect the combined effect of the agent architecture and the model."
2. Focus comparisons on relative patterns (coop/solo ratio, difficulty curves, failure mode distribution) rather than absolute success rates.
3. If absolute comparability is needed, integrate with the CooperBench CLI directly: `cooperbench run --subset lite --setting coop` using their infrastructure with Command A as the backing model.
4. Measure the agent-architecture confound: run the paper's agent SDK with Command A on a small subset (10 pairs) and compare to our pipeline's results on the same pairs.

**Warning signs:**
- Solo success rates differ from paper by > 20 percentage points.
- Failure mode distribution is dominated by categories not seen in the paper (suggesting agent architecture artifacts, not model limitations).
- Patches generated by our pipeline are structurally different from the paper's (e.g., shorter, fewer files touched, no tool use traces).

**Phase to address:**
Architecture decision phase -- must be decided at project start. Either commit to using the paper's agent SDK or commit to the prompt-based approach with clear comparability caveats.

---

### Pitfall 7: Cost Blowup From Uncontrolled API Calls

**What goes wrong:**
The full evaluation requires: (a) 100 pairs x 3 settings x 3 rounds x 2 agents = 1,800 LLM generation calls for patch generation, (b) 100 pairs x 10 classifiers x (1-3 calls per classifier for retries) = 1,000-3,000 LLM classification calls, (c) potential re-runs for infrastructure failures. At Command A pricing, this can easily exceed budget. The pipeline currently has no cost tracking or budget guard.

**Why it happens:**
Each cooperation round generates two LLM calls (solver + verifier), each consuming up to 8,192 output tokens. Classification calls are cheaper (1,024 max tokens each) but numerous. Retries for API errors or Docker failures multiply the total.

**How to avoid:**
1. Implement a cost tracker that estimates and accumulates cost per API call. Set a hard budget ceiling that halts execution when reached.
2. Start with the flash subset (50 pairs) before scaling to lite (100 pairs). This cuts cost by 50% for initial validation.
3. Cache LLM responses: if a prompt is identical (same transcript, same classifier), reuse the cached result. Useful for re-runs after infrastructure fixes.
4. For classification, batch transcripts if the model supports it, or rate-limit to avoid 429 errors that trigger expensive retries.
5. Track token usage per call (the pipeline already records `TokenUsage`) and report cumulative cost in the results JSON.

**Warning signs:**
- Total token usage exceeds 2x the initial estimate.
- Retry rate > 10% on API calls (suggests rate limiting or model instability).
- Classification step takes longer than patch generation step (should be the reverse).

**Phase to address:**
Cost planning phase -- establish budget and tracking before any at-scale runs.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Concatenate patches instead of git merge | Avoids Docker for merge step | Wrong merge outcomes for overlapping files, incorrect Figure 4 data | Never for final results; OK for smoke tests |
| Skip majority-vote on LLM classifiers | 3x faster, 3x cheaper | Noisy Figure 5, unreproducible failure rates | Acceptable during development; never for final figures |
| Use lite subset (100 pairs) instead of full (652) | 6x cheaper, 6x faster | Sparse difficulty buckets, wide CIs, underpowered analysis | Acceptable if CIs are reported honestly; full dataset preferred for Figure 6 |
| Hand-roll Wilson CI formula | No dependency on statsmodels | Edge case bugs at p=0, p=1, n=0; incorrect CIs invalidate figures | Never -- use `statsmodels` or `scipy.stats` |
| Reuse paper's baseline rates from hardcoded `PAPER_BASELINE_RATES` dict | Faster comparison chart | If paper numbers are transcribed incorrectly or apply to different subset, comparison is invalid | Acceptable with careful transcription; better to parse from paper data if available |

## Integration Gotchas

Common mistakes when connecting to external services.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Cohere API (LLM calls) | Using production endpoint when staging key is required | Check `COHERE_API_URL` env var; current code defaults to staging (`stg.api.cohere.com`). If switching to production, update the URL AND the API key. |
| Cohere API (rate limits) | Sending 1,800+ calls without rate limiting | Implement per-second rate limiter (current `base.py` has 0.5s base delay but no token-bucket limiter). At 4 parallel tasks x 2 agents = 8 concurrent calls, this may hit limits. |
| Docker daemon | Assuming Docker is running and has pull permissions | Check `docker info` at pipeline start. Pre-pull all required images before starting evaluation to avoid timeout during first run. |
| Docker images | Building per-task images during evaluation | Pre-build all task images in a separate step. The CooperBench dataset has Dockerfiles per task; build them once and cache. Image builds can take 5-20 minutes per task. |
| CooperBench dataset | Assuming `repos/CooperBench/dataset/` has all required files | Verify each task directory has: Dockerfile, runner.sh, feature*/tests.patch, feature*/feature.patch. Some tasks may be incomplete. |
| Git operations inside containers | Assuming git is available in all task containers | Some Dockerfiles may not install git. The merge and test pipeline requires git. Check image contents before running. |

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Sequential Docker evaluation | Works for 5 pairs in 30 minutes | Use `max_parallel_tasks` (currently 4); but monitor Docker daemon memory. Each container uses 4GB. | > 20 pairs: wall time exceeds 6 hours sequential |
| Synchronous LLM classifier calls | Fine for 10 transcripts | Batch or parallelize classification; current `classify()` is synchronous with sleep-based rate limiting | > 50 transcripts: classification takes > 2 hours |
| Storing full test output in eval JSON | Works for 10 results | Truncate test output to 5KB per feature (paper's sandbox.py truncates to 5000 chars). Full output for 300 runs can exceed 1GB. | > 100 runs: JSON files become unwieldy |
| Loading all results into memory for analysis | Fine for 100 results | Use streaming/incremental loading for statistical computation | > 500 runs: memory pressure on analysis machine |

## Security Mistakes

Domain-specific security issues beyond general web security.

| Mistake | Risk | Prevention |
|---------|------|------------|
| Running agent-generated patches in Docker without `--network=none` | Agent patches could exfiltrate data or download malicious code | Already using `--network=none` in evaluator -- verify this is enforced for ALL containers |
| Storing Cohere API key in code or config files | Key leak in git history | Use `CO_API_KEY` env var only (current approach is correct); never commit `.env` files |
| Docker containers running as root without resource limits | Containers could exhaust host resources or escape | Use `--memory=4g --cpus=2` (already set) and consider `--read-only` filesystem with specific tmpfs mounts |
| Agent patches modifying test files | Agent can game evaluation by modifying test assertions | The paper's `_filter_test_files()` strips test file changes from patches -- verify our pipeline does the same |

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Difficulty scoring:** Computing d(t) from Command A solo results looks correct, but without verifying against paper's multi-model d(t), the Figure 6 x-axis is not comparable to the paper's. Verify by checking whether the difficulty distribution shape matches.
- [ ] **Wilson CIs:** Computing Wilson CIs per bucket looks correct, but edge cases (empty buckets, n=1) may produce NaN or infinite values. Verify by testing with synthetic data including all edge cases.
- [ ] **AUC computation:** Trapezoidal integration over 10 buckets looks correct, but empty buckets cause discontinuities. Verify by computing AUC both with and without empty buckets and comparing.
- [ ] **Merge pipeline:** Naive + union merge fallback looks correct, but the merge is done outside the actual repository context (temp directory vs. full repo clone). Verify by comparing merge outcomes on 10 pairs against the paper's gold_conflict_report.json.
- [ ] **LLM classifier prompt fidelity:** Classifier prompts match CooperBench failure categories, but prompts may not match the paper's exact prompts (if they release them). Verify by comparing our classification on the paper's example transcripts.
- [ ] **Figure 4 communication analysis:** Computing success rates for "with comm" vs "no comm" looks correct, but requires running the SAME pairs in both settings with identical seeds. Verify by checking that pair lists are identical.
- [ ] **Cost tracking:** Token counts are recorded per call, but conversion to USD requires knowing the model's pricing tier. Verify that cost estimates match Cohere's published pricing for Command A.
- [ ] **Result reproducibility:** LLM calls at temperature=0.2 (config default) are not deterministic. Verify by running the same pair twice and comparing patches.

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Docker failures corrupt results | MEDIUM | Re-run only the failed pairs (idempotent if eval.json is checked). Filter results where `error` field is non-null. |
| Wilson CIs computed incorrectly | LOW | Recompute from raw results JSON. No re-running needed if raw pass/fail data is saved per pair. |
| LLM classifier produced noisy results | HIGH | Re-run classification with majority voting (3x cost). Or: manually annotate the ~30 pairs where classifier confidence < 0.6. |
| Difficulty scores don't match paper | MEDIUM | Recompute using alternative proxy (gold conflict rate, patch complexity). Regenerate Figure 6 with new x-axis. |
| Merge pipeline produces wrong outcomes | HIGH | Must fix merge logic and re-run ALL coop evaluations. This is expensive. Test merge logic on 5 pairs first. |
| Agent SDK mismatch produces incomparable results | HIGH | Either integrate with paper's OpenHands SDK (major effort) or reframe analysis as "Command A with prompt-based agents" rather than "CooperBench reproduction." |
| Budget exceeded before completing all runs | MEDIUM | Prioritize: complete flash subset (50 pairs) first. Report results on flash subset with a note that lite subset results are partial. |

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Difficulty score mismatch | Data preparation | Compare difficulty distribution histogram to paper's Figure 6 x-axis; compute rank correlation if paper data available |
| Docker container failures | Evaluation infrastructure | Run 10-pair smoke test; verify infra_error rate < 5%; check Docker health monitoring is active |
| Wilson CI degenerate buckets | Statistical analysis pipeline | Unit test Wilson CI computation with n=0, n=1, p=0, p=1; verify bucket sample sizes > 5 |
| LLM classifier noise | Classification pipeline | Compute intra-run consistency on 10 pairs (3 runs each); verify Cohen's kappa > 0.6 vs manual labels |
| Merge conflict conflation | Evaluation infrastructure | Compare merge outcomes on 10 known-conflict pairs from gold_conflict_report.json |
| Agent SDK mismatch | Architecture decision (phase 0) | Run 5 pairs with both pipeline approaches; quantify success rate delta |
| Cost blowup | Cost planning (phase 0) | Estimate total cost before starting; set hard budget ceiling; validate estimate on flash subset |

## Sources

- [CooperBench paper (arXiv:2601.13295)](https://arxiv.org/abs/2601.13295) -- primary reference for methodology, figures, and evaluation approach
- [CooperBench HTML paper](https://arxiv.org/html/2601.13295) -- detailed figure descriptions and statistical methodology
- [Wilson CI properties for small samples](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval) -- Wilson interval behavior at boundaries
- [Rating Roulette: Self-Inconsistency in LLM-As-A-Judge (EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.1361.pdf) -- LLM judge reliability research
- [Can You Trust LLM Judgments? (arXiv:2412.12509)](https://arxiv.org/html/2412.12509v2) -- inter-rater reliability limitations for LLM judges
- [Replication in ML benchmarks](https://mlbenchmarks.org/07-replication-machine-learning.html) -- reproducibility challenges and benchmark comparability
- [A Survey on LLM-as-a-Judge (arXiv:2411.15594)](https://arxiv.org/abs/2411.15594) -- comprehensive survey of LLM judge methodology
- Codebase analysis: `src/evaluation/cooperbench/evaluator.py`, `repos/CooperBench/src/cooperbench/eval/sandbox.py`, `cooperbench-eval/src/llm_judge/base.py`, `cooperbench-eval/src/report/generator.py`

---
*Pitfalls research for: CooperBench paper reproduction with Command A model*
*Researched: 2026-02-14*
