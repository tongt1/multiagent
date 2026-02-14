# Requirements: Reward Shaping Comparison Experiment

**Defined:** 2026-02-14
**Core Value:** All 5 reward strategies must be fully wired end-to-end so shaped rewards affect gradient updates and produce meaningful WandB comparison curves.

## v1 Requirements

### Reward Integration

- [ ] **RINT-01**: DebateMetricStreamer applies selected reward shaping strategy to raw rewards before they reach the learner
- [ ] **RINT-02**: Shaped rewards replace raw rewards in GRPO/RLOO gradient computation (not just logged as side metrics)
- [ ] **RINT-03**: Per-role reward dicts (from difference_rewards, coma_advantage, reward_mixing) are correctly handled — mapped to per-turn rewards based on role labels
- [ ] **RINT-04**: Identity strategy produces identical training behavior to no-strategy baseline (regression check)

### Experiment Configuration

- [ ] **ECFG-01**: SWEEP config for identity baseline (SmolLM-135M, flex queue, dev-low, MATH-500)
- [ ] **ECFG-02**: SWEEP config for difference_rewards strategy
- [ ] **ECFG-03**: SWEEP config for potential_based strategy (debate_length potential, gamma=0.99)
- [ ] **ECFG-04**: SWEEP config for coma_advantage strategy (n_rollouts_per_prompt=4)
- [ ] **ECFG-05**: SWEEP config for reward_mixing strategy (alpha=0.5)
- [ ] **ECFG-06**: All 5 configs share identical hyperparameters except reward shaping (fair comparison)
- [ ] **ECFG-07**: All 5 configs use post-training flex queue with dev-low priority
- [ ] **ECFG-08**: All 5 configs use MATH-500 online data from existing GCS path

### Observability

- [ ] **OBSV-01**: WandB logs shaped reward values per training step (distinct from raw correctness_score)
- [ ] **OBSV-02**: WandB logs reward strategy name as run metadata for filtering/grouping
- [ ] **OBSV-03**: Training loss, accuracy, and shaped reward curves visible and comparable across all 5 runs in same WandB project

## v2 Requirements

### Extended Analysis

- **ANLZ-01**: Statistical comparison of convergence rates across strategies
- **ANLZ-02**: Per-role reward decomposition visualization for multi-role strategies
- **ANLZ-03**: Hyperparameter sensitivity analysis per strategy

### Scale-up

- **SCAL-01**: Repeat comparison with 7B model
- **SCAL-02**: Extended training steps for convergence analysis

## Out of Scope

| Feature | Reason |
|---------|--------|
| Multi-model training (separate solver/verifier models) | Orthogonal to reward shaping comparison |
| Rollout strategies (best-of-N, self-consistency) | Separate concern, keep identity rollout |
| Per-strategy hyperparameter tuning | Fair comparison requires same defaults first |
| Production-scale runs | 135M is for pipeline validation, scale later |
| Automated analysis reports | Manual WandB analysis sufficient for v1 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| RINT-01 | — | Pending |
| RINT-02 | — | Pending |
| RINT-03 | — | Pending |
| RINT-04 | — | Pending |
| ECFG-01 | — | Pending |
| ECFG-02 | — | Pending |
| ECFG-03 | — | Pending |
| ECFG-04 | — | Pending |
| ECFG-05 | — | Pending |
| ECFG-06 | — | Pending |
| ECFG-07 | — | Pending |
| ECFG-08 | — | Pending |
| OBSV-01 | — | Pending |
| OBSV-02 | — | Pending |
| OBSV-03 | — | Pending |

**Coverage:**
- v1 requirements: 15 total
- Mapped to phases: 0
- Unmapped: 15 ⚠️

---
*Requirements defined: 2026-02-14*
*Last updated: 2026-02-14 after initial definition*
