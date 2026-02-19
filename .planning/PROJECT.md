# CooperBench Reproduction

## What This Is

A reproduction of Figures 4, 5, and 6 from the CooperBench paper (arxiv:2601.13295) using Cohere's internal Command A model. The project runs the CooperBench benchmark in solo, coop-no-comm, and coop-with-comm settings, then generates publication-quality figures showing the coordination gap, communication effects, and communication error taxonomy. Runs on the existing cooperbench-repro codebase with Docker-based evaluation sandboxes.

## Core Value

Produce verifiable figures (4, 5, 6) that replicate the paper's key findings — the solo-coop coordination gap, communication's failure to improve cooperation despite reducing merge conflicts, and the breakdown of communication errors — using Command A instead of the paper's external models.

## Requirements

### Validated

- Upstream CooperBench repo with Docker eval backend — existing
- CooperBench pipeline wrapper (`src/evaluation/cooperbench/`) — existing
- Failure mode analysis package (`cooperbench-eval/`) — existing
- 18/26 lite subset Docker images — existing

### Active

- [ ] Pull/build remaining 8 Docker images for lite subset (outlines 1655/1706, dspy 8587/8635, go-chi 27, llama-index 17244, react-hook-form 85/153)
- [ ] End-to-end Docker eval pipeline verified working (apply patch + run tests in container)
- [ ] Run Command A in solo mode on lite subset (100 pairs)
- [ ] Run Command A in coop mode without communication on lite subset
- [ ] Run Command A in coop mode with communication on lite subset
- [ ] Compute task difficulty scores d(t) in [0,1] for each task
- [ ] Figure 4: Difficulty-stratified success curves with Wilson 95% CIs, AUC gap, and retention (Algorithm 1 from paper)
- [ ] Figure 5(a): Comm vs no-comm success rate comparison with statistical significance
- [ ] Figure 5(b): Merge conflict rate with/without communication
- [ ] Figure 5(c): Communication overhead breakdown by speech act type (plan, question, update)
- [ ] Figure 6: Communication error taxonomy breakdown using C1a/C1b/C2/C3b/C4a/C4b classifier prompt
- [ ] Qualitative transcript analysis: Plan:Question ratio, first-turn planning effect, specificity metrics (line number/file path mentions)
- [ ] All figures saved as publication-quality PDFs/PNGs

### Out of Scope

- Multiple models — using Command A only, not replicating the paper's 5-6 model comparison
- Full dataset (652 tasks) — start with lite (100 pairs), only scale to full if lite validates
- Training pipeline — this is evaluation/analysis only, no RL training
- Figure 1, 2, 3 from the paper — focusing on figures 4, 5, 6 only
- GCP/Modal backends — Docker backend only

## Context

- The CooperBench paper studies multi-agent code cooperation where two agents each implement a feature in a shared codebase
- Key finding: agents perform worse cooperating than solo (coordination gap), and communication doesn't close this gap despite reducing merge conflicts
- Docker images are hosted on Docker Hub at `akhatua/cooperbench-*`, per-task images contain the repo at a specific commit
- The upstream `cooperbench` CLI (`cooperbench run -m MODEL`) orchestrates benchmark runs using OpenHands agent SDK
- Communication error taxonomy from the paper: C1a (unanswered direct question, no reply), C1b (unanswered, ignored), C2 (non-answer/vague), C3b (incorrect claim, corrected), C4a (spammy repetition, same info), C4b (spammy repetition, near-duplicate status blocks)
- Difficulty score d(t) likely derives from gold patch complexity or test suite difficulty — needs investigation from upstream repo
- The analysis pipeline needs: Wilson confidence intervals, trapezoidal AUC integration, difficulty bucketing (10 equal buckets over [0,1])

## Constraints

- **Model**: Command A (command-a-03-2025) only — internal Cohere model
- **Eval backend**: Docker only — images already partially pulled locally
- **Dataset**: Lite subset first (26 tasks, 100 pairs, 12 repos), full later
- **Output**: Figures must be directly comparable to paper's figures 4/5/6 in structure and methodology

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Command A only (no multi-model) | Simplifies initial reproduction, validates pipeline first | -- Pending |
| Start with lite subset | 100 pairs sufficient for statistical analysis, faster iteration | -- Pending |
| Docker backend only | Images already available locally, simplest path | -- Pending |
| Use upstream cooperbench CLI directly | Avoid reimplementing agent orchestration, focus on analysis | -- Pending |

---
*Last updated: 2026-02-14 after initialization*
