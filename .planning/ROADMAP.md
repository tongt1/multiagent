# Roadmap: CooperBench Coordination Evaluation Metrics

## Overview

Single-phase delivery of a complete CooperBench failure mode evaluation framework. Phase 1 implements all 10 failure mode detectors, metrics aggregation, and bar graph visualization, producing a standalone evaluation pipeline that can run on Cohere model transcripts.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Failure Mode Evaluation Pipeline** - Detect all 10 CooperBench failure modes, compute metrics, and produce failure rate visualization

## Phase Details

### Phase 1: Failure Mode Evaluation Pipeline
**Goal**: Users can evaluate multi-agent transcripts for all 10 CooperBench coordination failure modes and see results as a bar graph
**Depends on**: Nothing (first phase)
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, EVAL-06, EVAL-07, EVAL-08, EVAL-09, EVAL-10, METR-01, METR-02, METR-03
**Success Criteria** (what must be TRUE):
  1. Running the evaluation on a transcript produces per-failure-mode detection results for all 10 categories (work overlap, divergent architecture, repetition, unresponsiveness, unverifiable claims, broken commitment, dependency access, placeholder misuse, parameter flow, timing dependency)
  2. Metrics module computes occurrence rates and the output matches CooperBench baseline distribution shape
  3. A failure rate bar graph is saved to figures/failure_rates.png showing all 10 failure modes
  4. The pipeline runs end-to-end on Cohere model transcripts without errors
**Plans:** 2 plans

Plans:
- [ ] 01-01-PLAN.md — All 10 failure mode detector classes with data models
- [ ] 01-02-PLAN.md — Metrics computation, bar graph visualization, and end-to-end runner

## Progress

**Execution Order:**
Phases execute in numeric order: 1

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Failure Mode Evaluation Pipeline | 0/2 | Not started | - |
