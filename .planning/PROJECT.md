# CooperBench Coordination Evaluation Metrics

## What This Is

A Python evaluation framework that measures 10 coordination failure modes from the CooperBench paper across multi-agent system transcripts. It produces failure rate analysis, graphical output, and supports evaluation on Cohere models.

## Core Value

Accurately measure and visualize all 10 CooperBench failure modes — work overlap, divergent architecture, repetition, unresponsiveness, unverifiable claims, broken commitment, dependency access, placeholder misuse, parameter flow, and timing dependency — from multi-agent transcripts.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Implement detection for all 10 failure modes from CooperBench paper
- [ ] Produce failure rate bar graph (figures/failure_rates.png)
- [ ] Support evaluation on Cohere model transcripts
- [ ] Measure failure rates matching CooperBench baselines: Work overlap (33.2%), Divergent architecture (29.7%), Repetition (14.7%), Unresponsiveness (8.7%), Unverifiable claims (4.3%), Broken commitment (3.7%), Dependency access (1.7%), Placeholder misuse (1.5%), Parameter flow (1.3%), Timing dependency (1.3%)

### Out of Scope

- Training or fine-tuning models — evaluation only
- CooperBench figure reproduction (beta teammate's task)
- Small training runs (alpha teammate's task)

## Context

CooperBench is a benchmark for evaluating multi-agent coordination quality. The paper identifies 10 failure categories with specific occurrence rates from their evaluation. This module implements automated detection of these failure modes from agent conversation transcripts and produces visualization of failure rates.

The evaluation files are:
- `evaluation/failure_modes.py` — Failure mode detection classes and logic
- `evaluation/metrics.py` — Metrics computation and aggregation
- `figures/failure_rates.png` — Output visualization

Cross-references:
- Beta teammate builds CooperBench reproduction (figures 4/5/6)
- Alpha teammate runs small training experiments

## Constraints

- **File ownership**: Only modify evaluation/failure_modes.py, evaluation/metrics.py, figures/failure_rates.png
- **Python**: Implementation in Python
- **Standalone**: Must work independently without beta/alpha teammate outputs

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Python implementation | Standard for ML evaluation | — Pending |
| matplotlib for graphing | Lightweight, produces PNG directly | — Pending |

---
*Last updated: 2026-02-13 after initialization*
