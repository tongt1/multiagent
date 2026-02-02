# Phase 2: Scale & Verify - Research

**Researched:** 2026-02-01
**Domain:** Concurrent async execution, dataset integration, verifiable rewards, distributed computing
**Confidence:** MEDIUM-HIGH

## Summary

Phase 2 extends the Phase 1 single-problem pipeline to handle hundreds/thousands of problems concurrently with verifiable rewards. Research covered five critical domains:

**Asyncio concurrency:** Python 3.11+ TaskGroup is the modern standard, offering automatic cancellation on failure and superior error handling vs. gather(). Semaphores limit concurrent operations (recommend 5-20 for LLM APIs). The existing async pipeline can be wrapped with semaphore-controlled batch execution.

**Dataset integration:** The Bee framework (released late 2025) is a multimodal LLM project with Honey-Data-15M dataset, but it's focused on supervised fine-tuning for vision-language tasks, not math/code evaluation. The MATH dataset (Hendrycks 2021) is the standard for math evaluation with 12,500 problems using LaTeX-formatted answers in `\boxed{}` tags. HumanEval (OpenAI) is the standard for code with 164 problems using pass@k metrics.

**Verifiable rewards:** Math answers use sympy.equals() for symbolic equivalence after extracting from `\boxed{}` (avoid regex, use LLM extraction at 99.5%+ accuracy). Code execution requires subprocess with resource limits (resource module + preexec_fn), timeout enforcement (signal-based or thread-based), and isolated execution per test case.

**Training trajectories:** RL training expects per-step rewards with agent roles, turn IDs, and timestamps. The existing TrajectoryEntry model needs extension to include reward scores and supports JSONL append-only for concurrent writes. OpenAI/OpenRLHF use messages-based JSONL with role annotations.

**Distributed execution:** "kjobs/apiary" appear to be project-specific or internal systems (no public documentation found). Kubernetes Job/CronJob is the standard for batch processing with frameworks like Kueue for queue management. Python job submission uses Kubernetes Python client or kubectl subprocess calls.

**Primary recommendation:** Use Python 3.11+ asyncio.TaskGroup with semaphore-controlled concurrency, extend TrajectoryEntry to include per-step rewards, implement math verification with sympy and code verification with subprocess sandboxing, and target standard Kubernetes Job APIs for distributed execution.

## Standard Stack

The established libraries/tools for concurrent LLM pipeline execution and verifiable evaluation:

### Core Concurrency
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| asyncio.TaskGroup | Python 3.11+ | Concurrent task execution with auto-cancellation | Superior error handling vs gather(), recommended in official docs |
| asyncio.Semaphore | Python 3.3+ | Rate limiting concurrent operations | Built-in primitive for controlling concurrency |
| instructor | 1.0+ | Structured LLM outputs with async support | Already in use, excellent asyncio.gather integration |

### Math Evaluation
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| sympy | 1.14+ | Symbolic math equivalence checking | Industry standard for mathematical comparison, handles algebraic equivalence |
| datasets | 2.0+ | HuggingFace dataset loading | Standard for loading MATH dataset and evaluation benchmarks |

### Code Execution
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| subprocess | Built-in | Isolated code execution | Standard library, production-ready |
| resource | Built-in (Unix) | Memory/CPU limits via setrlimit | POSIX standard for resource constraints |
| pytest-timeout | 2.0+ | Test timeout enforcement | De facto standard for Python test timeouts |

### Distributed Execution
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| kubernetes | 28+ | Kubernetes Python client | For programmatic Job creation and monitoring |
| kubectl | 1.28+ | Kubernetes CLI | For simple job submission scripts |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| jsonlines | 4.0+ | JSONL read/write | For loading trajectory files or dataset batch processing |
| loguru | 0.7+ | Structured logging | Already in use, essential for debugging concurrent execution |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| TaskGroup | gather() with return_exceptions=True | gather() doesn't auto-cancel on failure, requires manual exception handling |
| sympy | Regex + string matching | Regex fails on equivalent forms (e.g., 1/4 vs 0.25), unreliable for math |
| subprocess | Docker-based sandboxes | Docker adds overhead, not needed for simple test execution with timeouts |
| Kubernetes Jobs | Ray, Dask | K8s is simpler for batch inference, Ray/Dask add complexity for non-distributed training |

**Installation:**
```bash
# Core dependencies (most already present from Phase 1)
pip install instructor litellm sympy datasets jsonlines

# Optional for Kubernetes integration
pip install kubernetes
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── orchestration/
│   ├── pipeline.py              # Existing: SolverVerifierJudgePipeline.run()
│   ├── batch_executor.py        # NEW: BatchPipelineExecutor with semaphore control
│   └── iteration.py             # Existing: IterationController
├── evaluation/
│   ├── math_verifier.py         # NEW: Extract answer from \boxed{}, sympy equivalence
│   ├── code_executor.py         # NEW: Subprocess execution with resource limits
│   └── reward_calculator.py     # NEW: Compute per-step rewards from verification
├── data/
│   ├── dataset_loader.py        # NEW: Load MATH/HumanEval datasets
│   └── batch_reader.py          # NEW: Batch problem iteration
├── models/
│   ├── trajectory.py            # EXTEND: Add reward field to TrajectoryEntry
│   └── evaluation.py            # EXTEND: Add ground_truth comparison
└── infrastructure/
    ├── trajectory_logger.py     # Existing: Thread-safe JSONL append
    └── kubernetes_client.py     # NEW: Job submission and monitoring
```

### Pattern 1: Semaphore-Controlled Batch Execution
**What:** Wrap existing Pipeline.run() in semaphore-controlled concurrent execution
**When to use:** Processing multiple problems with rate-limited LLM APIs
**Example:**
```python
# Source: Python official docs + instructor blog pattern
import asyncio
from src.orchestration.pipeline import SolverVerifierJudgePipeline

class BatchPipelineExecutor:
    def __init__(self, pipeline: SolverVerifierJudgePipeline, max_concurrent: int = 10):
        self.pipeline = pipeline
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def run_one(self, problem: dict) -> PipelineResult:
        """Run single problem with semaphore."""
        async with self.semaphore:
            return await self.pipeline.run(
                problem_description=problem["problem"],
                problem_metadata={"id": problem["id"], "type": problem["type"]}
            )

    async def run_batch(self, problems: list[dict]) -> list[PipelineResult]:
        """Run batch with TaskGroup for auto-cancellation."""
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(self.run_one(p)) for p in problems]
        # TaskGroup automatically awaits all tasks and cancels on first exception
        return [task.result() for task in tasks]
```

### Pattern 2: Math Answer Extraction and Verification
**What:** Extract LaTeX answers from `\boxed{}` and verify equivalence with sympy
**When to use:** MATH dataset evaluation
**Example:**
```python
# Source: MATH dataset paper + sympy docs
import re
import sympy as sp

def extract_boxed_answer(solution: str) -> str | None:
    """Extract answer from LaTeX \\boxed{...}."""
    # Match nested braces
    match = re.search(r'\\boxed\{([^}]+)\}', solution)
    return match.group(1) if match else None

def verify_math_answer(predicted: str, ground_truth: str) -> bool:
    """Verify symbolic equivalence using sympy."""
    try:
        pred_expr = sp.sympify(predicted)
        true_expr = sp.sympify(ground_truth)
        # Use equals() for symbolic comparison after simplification
        return pred_expr.equals(true_expr)
    except (sp.SympifyError, AttributeError):
        # Fall back to string comparison if sympify fails
        return predicted.strip() == ground_truth.strip()
```

### Pattern 3: Sandboxed Code Execution with Resource Limits
**What:** Execute code in subprocess with memory/CPU/time limits
**When to use:** HumanEval or code problem verification
**Example:**
```python
# Source: Python resource module docs + pytest-subprocess patterns
import subprocess
import resource
import signal

def limit_resources():
    """Preexec function to set resource limits."""
    # Limit memory to 256MB
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
    # Limit CPU time to 5 seconds
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))

def execute_code(code: str, test_input: str, timeout: int = 10) -> tuple[bool, str]:
    """Execute code with resource limits and timeout."""
    try:
        result = subprocess.run(
            ["python", "-c", code],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=limit_resources,  # Unix only
        )
        return result.returncode == 0, result.stdout
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)
```

### Pattern 4: Training-Structured Trajectory with Per-Step Rewards
**What:** Extend TrajectoryEntry to include reward scores for RL training
**When to use:** Exporting trajectories for downstream RLHF/RL training
**Example:**
```python
# Source: OpenRLHF schema + existing TrajectoryEntry model
from pydantic import BaseModel

class TrajectoryEntry(BaseModel):
    """Extended trajectory entry with reward."""
    timestamp: str
    run_id: str
    step_id: int
    agent: str           # "solver", "verifier", "judge"
    action: str          # "generate", "validate", "score"
    input: dict[str, Any]
    output: dict[str, Any]
    metadata: dict[str, Any]

    # NEW: Reward fields for RL training
    reward: float | None = None              # Per-step reward score
    terminal: bool = False                    # Is this the final step?
    success: bool | None = None              # Did task succeed?
```

### Pattern 5: Dataset Loading and Batch Processing
**What:** Load MATH/HumanEval datasets and iterate in batches
**When to use:** Batch evaluation of problems from standard benchmarks
**Example:**
```python
# Source: HuggingFace datasets library
from datasets import load_dataset

def load_math_dataset(split: str = "test") -> list[dict]:
    """Load MATH dataset from HuggingFace."""
    dataset = load_dataset("hendrycks/competition_math", split=split)
    return [
        {
            "id": idx,
            "problem": item["problem"],
            "solution": item["solution"],
            "level": item["level"],
            "type": item["type"],
        }
        for idx, item in enumerate(dataset)
    ]

def batch_iterator(items: list, batch_size: int = 100):
    """Iterate over items in batches."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]
```

### Pattern 6: Kubernetes Job Submission
**What:** Submit batch inference jobs to Kubernetes
**When to use:** Scaling to thousands of problems on distributed cluster
**Example:**
```python
# Source: Kubernetes Python client docs
from kubernetes import client, config

def submit_batch_job(job_name: str, image: str, problems_batch: list[dict]) -> str:
    """Submit Kubernetes Job for batch inference."""
    config.load_kube_config()
    batch_v1 = client.BatchV1Api()

    job = client.V1Job(
        metadata=client.V1ObjectMeta(name=job_name),
        spec=client.V1JobSpec(
            template=client.V1PodTemplateSpec(
                spec=client.V1PodSpec(
                    containers=[
                        client.V1Container(
                            name="inference",
                            image=image,
                            env=[
                                client.V1EnvVar(
                                    name="PROBLEMS",
                                    value=json.dumps(problems_batch)
                                )
                            ],
                            resources=client.V1ResourceRequirements(
                                requests={"cpu": "2", "memory": "8Gi"},
                                limits={"cpu": "4", "memory": "16Gi"}
                            )
                        )
                    ],
                    restart_policy="Never"
                )
            ),
            backoff_limit=3
        )
    )

    response = batch_v1.create_namespaced_job(namespace="default", body=job)
    return response.metadata.name
```

### Anti-Patterns to Avoid
- **Using gather() without return_exceptions=True:** Unhandled exceptions kill entire batch, no partial results
- **No semaphore rate limiting:** Overwhelms LLM API, hits rate limits, wastes retries
- **Regex for math answer extraction:** Fails on equivalent forms, use LLM extraction or sympy parsing
- **Running untrusted code without resource limits:** Memory bombs, infinite loops crash host
- **Holding references to completed tasks:** Prevents exception logging, causes silent failures
- **Sequential execution of independent problems:** Wastes concurrency potential, 10-100x slower

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Math equivalence checking | String comparison or regex | sympy.equals() | Handles symbolic equivalence (1/2 == 0.5, x+1 == 1+x), simplification |
| Code execution sandboxing | Custom process isolation | subprocess + resource module + timeout | POSIX-standard resource limits, battle-tested, handles edge cases |
| LLM answer extraction | Regex for \boxed{} | GPT-4 extraction (99.5% accuracy) | Handles nested braces, malformed LaTeX, ambiguous formatting |
| Concurrent rate limiting | Manual sleep() timing | asyncio.Semaphore | Automatic backpressure, no wasted wait time, production-ready |
| Exception handling in gather() | Try-except around each task | asyncio.TaskGroup (Python 3.11+) | Auto-cancellation on first failure, structured exception groups |
| JSONL concurrent writes | File locking mechanisms | Append-only writes with flush | JSONL format is naturally append-safe, flush ensures durability |
| Dataset loading | Custom parsers for MATH/HumanEval | HuggingFace datasets library | Handles caching, streaming, format conversion automatically |
| Kubernetes job management | kubectl subprocess calls with parsing | kubernetes Python client | Type-safe API, handles auth, retries, watches |

**Key insight:** Concurrency primitives (Semaphore, TaskGroup), mathematical comparison (sympy), and standard dataset loaders (HuggingFace) are mature solutions. Custom implementations introduce bugs and miss edge cases that took years to discover in production.

## Common Pitfalls

### Pitfall 1: TaskGroup Exception Handling
**What goes wrong:** Expecting partial results when one task fails in TaskGroup
**Why it happens:** TaskGroup auto-cancels all tasks on first exception (unlike gather)
**How to avoid:** Use gather(return_exceptions=True) if you need partial results despite failures, or wrap individual tasks in try-except to handle errors internally
**Warning signs:** "All my tasks are cancelled even though only one failed"

### Pitfall 2: Semaphore Value Too High
**What goes wrong:** Setting semaphore to 100+ concurrent LLM calls hits rate limits, costs spike
**Why it happens:** Assuming more concurrency = faster, ignoring API quotas
**How to avoid:** Start with 5-10 concurrent requests, monitor rate limit errors, tune based on provider limits (OpenAI: ~3500 RPM for tier 1, Cohere: ~1000 RPM)
**Warning signs:** 429 rate limit errors, tenacity retry storms, unexpected API costs

### Pitfall 3: sympy.equals() Returns None
**What goes wrong:** Treating None as False, marking correct answers as wrong
**Why it happens:** sympy can't prove equivalence in reasonable time, returns None instead of True/False
**How to avoid:** Check for None explicitly, fall back to string comparison or mark as "uncertain" for human review
**Warning signs:** Lower-than-expected accuracy on algebraically complex but correct answers

### Pitfall 4: Resource Limits Don't Work on Non-Unix
**What goes wrong:** resource.setrlimit() fails on Windows, code execution has no limits
**Why it happens:** resource module is Unix-only, subprocess preexec_fn doesn't exist on Windows
**How to avoid:** Platform detection + fallback to timeout-only on Windows, or require Unix for code execution
**Warning signs:** ImportError on Windows, runaway processes consuming system memory

### Pitfall 5: Concurrent JSONL Writes Without Flush
**What goes wrong:** Lost trajectory entries when process crashes, incomplete logs
**Why it happens:** File buffering delays writes, concurrent writes interleave in buffer
**How to avoid:** Call flush() after every write (already implemented in TrajectoryLogger), use separate files per run_id to avoid interleaving
**Warning signs:** Trajectory files have fewer entries than expected, last entries missing after crash

### Pitfall 6: Forgetting task_done() in Queue Pattern
**What goes wrong:** queue.join() hangs forever waiting for consumers
**Why it happens:** Consumers call queue.get() but forget queue.task_done(), join() never completes
**How to avoid:** Always pair get() with task_done() in try-finally, or use async for item in queue pattern
**Warning signs:** Program hangs at queue.join(), consumers idle but join() never returns

### Pitfall 7: Kubernetes Job Without Resource Limits
**What goes wrong:** Single job consumes all cluster resources, starves other workloads
**Why it happens:** K8s defaults to unlimited resources, greedy jobs monopolize nodes
**How to avoid:** Always set requests and limits in Job spec, start conservative (2 CPU, 8Gi), tune based on metrics
**Warning signs:** Cluster node exhaustion, other pods evicted, OOMKilled errors

### Pitfall 8: Loading Entire Dataset Into Memory
**What goes wrong:** 12,500 MATH problems with full solutions OOMs on large batches
**Why it happens:** Calling list(dataset) loads everything at once, datasets supports streaming but not used
**How to avoid:** Use dataset.iter(batch_size=100) for streaming, or load indices only and fetch on-demand
**Warning signs:** Memory usage spikes at dataset loading, OOMKilled during initialization

## Code Examples

Verified patterns from official sources:

### Concurrent Batch Processing with Error Handling
```python
# Source: Python 3.11+ asyncio docs + instructor blog
import asyncio
from typing import Any

async def process_batch_with_errors(
    problems: list[dict],
    pipeline: SolverVerifierJudgePipeline,
    max_concurrent: int = 10
) -> tuple[list[PipelineResult], list[Exception]]:
    """Process batch with semaphore, collect both results and exceptions."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore(problem: dict):
        async with semaphore:
            return await pipeline.run(
                problem_description=problem["problem"],
                problem_metadata=problem
            )

    # Use gather with return_exceptions for partial results
    results = await asyncio.gather(
        *[run_with_semaphore(p) for p in problems],
        return_exceptions=True
    )

    # Separate successes from failures
    successes = [r for r in results if not isinstance(r, Exception)]
    failures = [r for r in results if isinstance(r, Exception)]

    # Log failures
    for exc in failures:
        logger.error(f"Pipeline failed: {exc}")

    return successes, failures
```

### Math Verification with Ground Truth Comparison
```python
# Source: MATH dataset evaluation harness + sympy docs
import sympy as sp
from src.models.evaluation import VerificationResult

def compute_math_reward(
    solution: str,
    ground_truth_solution: str
) -> tuple[float, VerificationResult]:
    """Compute reward by comparing extracted answers."""
    # Extract answers
    predicted = extract_boxed_answer(solution)
    expected = extract_boxed_answer(ground_truth_solution)

    if predicted is None or expected is None:
        return 0.0, VerificationResult(
            passed=False,
            critique="Could not extract answer from solution",
            confidence=0.0,
            scores={}
        )

    # Symbolic equivalence
    is_correct = verify_math_answer(predicted, expected)

    return (
        1.0 if is_correct else 0.0,
        VerificationResult(
            passed=is_correct,
            critique="" if is_correct else f"Expected {expected}, got {predicted}",
            confidence=1.0 if is_correct else 0.0,
            scores={"correctness": 1.0 if is_correct else 0.0}
        )
    )
```

### Code Execution with Test Case Verification
```python
# Source: HumanEval evaluation + subprocess best practices
def execute_with_tests(
    code: str,
    test_cases: list[dict[str, str]],
    timeout: int = 5
) -> tuple[float, list[bool]]:
    """Execute code against test cases, return pass rate and results."""
    results = []

    for test in test_cases:
        passed, output = execute_code(
            code=code,
            test_input=test["input"],
            timeout=timeout
        )

        if passed:
            # Check output matches expected
            passed = output.strip() == test["expected_output"].strip()

        results.append(passed)

    pass_rate = sum(results) / len(results) if results else 0.0
    return pass_rate, results
```

### Producer-Consumer with asyncio.Queue
```python
# Source: asyncio Queue documentation
import asyncio

async def batch_processor_queue_pattern(
    problems: list[dict],
    pipeline: SolverVerifierJudgePipeline,
    num_workers: int = 10
):
    """Process problems using queue-based producer-consumer."""
    queue = asyncio.Queue(maxsize=100)  # Bounded queue for backpressure

    async def producer():
        """Add problems to queue."""
        for problem in problems:
            await queue.put(problem)
        # Signal completion
        for _ in range(num_workers):
            await queue.put(None)

    async def consumer():
        """Process problems from queue."""
        results = []
        while True:
            problem = await queue.get()
            if problem is None:
                queue.task_done()
                break

            try:
                result = await pipeline.run(
                    problem_description=problem["problem"],
                    problem_metadata=problem
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {problem['id']}: {e}")
            finally:
                queue.task_done()

        return results

    # Start producer and consumers
    producer_task = asyncio.create_task(producer())
    consumer_tasks = [asyncio.create_task(consumer()) for _ in range(num_workers)]

    # Wait for all work to complete
    await queue.join()
    await producer_task

    # Gather results
    all_results = []
    for task in consumer_tasks:
        all_results.extend(await task)

    return all_results
```

### Trajectory Export for RL Training
```python
# Source: OpenRLHF trajectory format + existing TrajectoryEntry
import json
from pathlib import Path

def export_trajectory_for_training(
    trajectory_path: Path,
    output_path: Path
):
    """Convert pipeline trajectory to RL training format."""
    entries = []
    with open(trajectory_path) as f:
        for line in f:
            entries.append(json.loads(line))

    # Group by run_id
    runs = {}
    for entry in entries:
        run_id = entry["run_id"]
        if run_id not in runs:
            runs[run_id] = []
        runs[run_id].append(entry)

    # Convert to training format
    training_data = []
    for run_id, steps in runs.items():
        # Build turn-based trajectory
        trajectory = {
            "id": run_id,
            "turns": [],
            "total_reward": 0.0
        }

        for step in steps:
            turn = {
                "turn_id": step["step_id"],
                "agent": step["agent"],
                "action": step["action"],
                "input": step["input"],
                "output": step["output"],
                "reward": step.get("reward", 0.0),
                "timestamp": step["timestamp"]
            }
            trajectory["turns"].append(turn)
            trajectory["total_reward"] += turn["reward"]

        training_data.append(trajectory)

    # Write as JSONL
    with open(output_path, "w") as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| asyncio.gather() for concurrent tasks | asyncio.TaskGroup (Python 3.11+) | October 2022 | Auto-cancellation on failure, structured exception groups, safer defaults |
| Regex for math answer extraction | LLM-based extraction (GPT-4) | 2024 | 99.5%+ accuracy vs ~85% for regex, handles malformed LaTeX |
| Docker-based code sandboxing | subprocess + resource limits | Ongoing | Lower overhead for simple execution, Docker still preferred for untrusted workloads |
| Manual Kubernetes Job YAML | Kubernetes Python client | Stable since 2018 | Type-safe, programmatic job creation, easier integration |
| Custom dataset formats | HuggingFace datasets library | ~2020 | Standardized loading, caching, streaming for MATH/HumanEval |
| gather() exception handling | return_exceptions=True | Always available | Partial results on failures, explicit error handling |

**Deprecated/outdated:**
- **asyncio.gather() without error handling:** TaskGroup is now preferred for new code (Python 3.11+), gather() still valid but requires explicit exception handling
- **Regex-only math verification:** Modern evaluations use LLM extraction + sympy equivalence
- **Synchronous subprocess calls in async code:** Use asyncio.create_subprocess_exec() for true async, though sync subprocess is acceptable in semaphore-controlled context

## Open Questions

Things that couldn't be fully resolved:

1. **kjobs/apiary specifics**
   - What we know: Terms mentioned in project requirements, likely internal/project-specific distributed execution systems
   - What's unclear: API contracts, authentication, job submission format, monitoring capabilities
   - Recommendation: Substitute with standard Kubernetes Job APIs during planning, document integration points for later replacement. Create abstraction layer (DistributedExecutor interface) to swap implementations.

2. **Bee framework dataset integration**
   - What we know: Bee-8B is a multimodal LLM with Honey-Data-15M dataset (released late 2025), focused on vision-language tasks, not math/code evaluation
   - What's unclear: Whether "Bee framework" in requirements refers to this MLLM project or a different evaluation framework
   - Recommendation: Use standard MATH/HumanEval datasets for Phase 2, add Bee dataset support as optional extension if clarified later

3. **Optimal semaphore concurrency for different LLM providers**
   - What we know: 5-10 is safe starting point, provider-dependent rate limits exist
   - What's unclear: Exact limits for command-r-plus used in config, optimal batching for cost vs speed
   - Recommendation: Start with 10 concurrent, add metrics collection, tune based on rate limit errors and latency

4. **Windows compatibility for code execution**
   - What we know: resource module is Unix-only, no direct equivalent on Windows
   - What's unclear: Whether Windows support is required for Phase 2
   - Recommendation: Require Unix/Linux for code execution workloads, document platform requirement, consider psutil for cross-platform memory monitoring (without hard limits)

5. **RL training format compatibility**
   - What we know: OpenRLHF and similar frameworks use messages-based JSONL with role/turn structure
   - What's unclear: Exact schema required by downstream training consumers
   - Recommendation: Export flexible JSONL format with all fields (agent, turn_id, reward, input, output), document schema for training team integration

## Sources

### Primary (HIGH confidence)
- [Python asyncio.Semaphore documentation](https://docs.python.org/3/library/asyncio-sync.html) - Semaphore usage patterns, best practices
- [Python asyncio.TaskGroup documentation](https://docs.python.org/3/library/asyncio-task.html) - TaskGroup vs gather, error handling
- [SymPy documentation](https://docs.sympy.org/latest/modules/core.html) - Symbolic math comparison, equals() method
- [MATH dataset HuggingFace](https://huggingface.co/datasets/hendrycks/competition_math) - Dataset structure, fields, LaTeX formatting
- [Python resource module](https://docs.python.org/3/library/resource.html) - Resource limits for subprocess

### Secondary (MEDIUM confidence)
- [Instructor asyncio.gather guide](https://python.useinstructor.com/blog/2023/11/13/learn-async/) - Concurrent LLM processing patterns
- [HumanEval benchmark overview](https://github.com/openai/human-eval) - Code evaluation with pass@k metrics
- [Google Cloud Kubernetes batch best practices](https://docs.cloud.google.com/kubernetes-engine/docs/best-practices/batch-platform-on-gke) - Job patterns, resource management
- [Bee-8B project](https://open-bee.github.io/) - Multimodal LLM framework (not math/code evaluation)
- [OpenRLHF GitHub](https://github.com/OpenRLHF/OpenRLHF) - RLHF trajectory format examples

### Tertiary (LOW confidence - marked for validation)
- [Kueue for Kubernetes](https://www.coreweave.com/blog/kueue-a-kubernetes-native-system-for-ai-training-workloads) - Job queueing for AI workloads (WebSearch only, not verified with official docs)
- [MathVista answer extraction](https://github.com/lupantech/MathVista) - Claims 99.5% GPT-4 extraction accuracy (paper claims, not independently verified)
- Various asyncio exception handling blog posts - Best practices align with official docs but add context

## Metadata

**Confidence breakdown:**
- Asyncio concurrency patterns: HIGH - Official Python docs, widely adopted patterns
- Math verification (sympy): HIGH - Official docs, standard in evaluation harnesses
- Code execution sandboxing: MEDIUM-HIGH - Standard library approach, platform-dependent limitations
- Dataset loading (MATH/HumanEval): HIGH - Official HuggingFace datasets, well-documented
- Bee framework integration: LOW - Unclear if requirements refer to Bee-8B MLLM or different framework
- kjobs/apiary: LOW - No public documentation found, likely internal systems
- Kubernetes patterns: MEDIUM - Official K8s docs, but specific cluster setup unknown
- Training trajectory format: MEDIUM - Common patterns documented, exact downstream schema unclear

**Research date:** 2026-02-01
**Valid until:** ~30 days (stable technologies), 7 days for fast-moving (LLM APIs, new frameworks)
**Recommended validation:** Confirm kjobs/apiary APIs, verify Bee framework identity, test semaphore limits with actual LLM provider
