# Phase 1: Execution Infrastructure - Context

**Gathered:** 2026-02-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Set up and run the CooperBench benchmark with Command A across 3 experimental settings (solo, coop-no-comm, coop-with-comm) on the lite subset (100 pairs), using the upstream CooperBench CLI with Docker sandbox evaluation. Includes Docker image management, cost tracking, retry logic, and run resumability. Analysis and figure generation are separate phases.

</domain>

<decisions>
## Implementation Decisions

### CLI Integration
- Use the upstream `cooperbench` CLI directly via subprocess — NOT the custom wrapper in `src/evaluation/cooperbench/pipeline.py`
- Model string: `command-a-03-2025` with `COHERE_API_KEY` configured
- Agent type: `mini_swe_agent` (supports Docker backend; openhands_sdk is Modal-only — resolved during research)
- API endpoint: staging (`stg.api.cohere.com`) — current codebase default
- Command pattern: `cooperbench run -m command-a-03-2025 -a mini_swe_agent -s lite --setting {solo|coop} [--no-messaging] --backend docker`

### Docker Image Strategy
- Pull all missing images from Docker Hub (`docker pull akhatua/cooperbench-*:task{id}`)
- Block benchmark runs until all 26 lite subset images are available locally
- Verify each image after pulling (start container, check workspace exists)
- Use mounted disk for Docker storage — main disk is 128GB and insufficient for all images

### Run Orchestration
- Run solo setting first (needed for difficulty scores, fastest, validates pipeline)
- Then run coop-with-comm, then coop-no-comm
- Concurrency: 4 tasks in parallel (low parallelism — manageable resource usage)
- Must support resume: skip already-completed tasks on restart (critical for 300+ task runs)
- Run naming: `command-a-solo`, `command-a-coop-comm`, `command-a-coop-nocomm`

### Cost and Failure Handling
- No budget ceiling — track costs but don't halt
- Retry policy: 2 retries (3 total attempts) per task on infrastructure failure
- Infrastructure errors (retryable): Docker OOM/crash, API timeouts
- Genuine failures (not retried): clean test pass/fail results
- Task timeout: 10 minutes per task
- Tag results as `pass`, `fail`, or `infra_error` for downstream analysis

### Claude's Discretion
- Upstream CLI installation method (pip install, editable install, or PATH setup)
- Redis configuration for coop mode inter-agent messaging
- Exact orchestrator script structure and progress reporting format
- How to configure Docker to use mounted disk storage

</decisions>

<specifics>
## Specific Ideas

- The upstream CLI pattern is: `cooperbench run -m MODEL -a AGENT [-s SUBSET] [--setting SETTING]`
- The upstream repo is at `repos/CooperBench/` and can be installed as a package
- Docker images follow the naming convention `akhatua/cooperbench-{repo}:task{id}` where repo names have underscores replaced with dashes and `_task` suffix removed

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-execution-infrastructure*
*Context gathered: 2026-02-14*
