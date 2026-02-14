# Roadmap: Reward Shaping Comparison Experiment

## Overview

Wire the existing reward shaping modules end-to-end through DebateMetricStreamer into training gradients, create 5 matched SWEEP configs (one per strategy), and validate that shaped rewards produce distinct, comparable WandB learning curves on SmolLM-135M with MATH-500.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: End-to-End Reward Integration** - Shaped rewards flow through DebateMetricStreamer into GRPO/RLOO gradient computation
- [ ] **Phase 2: Experiment Configuration** - Five matched SWEEP configs ready for submission on flex queue
- [ ] **Phase 3: Observability and Comparison** - WandB captures strategy-specific metrics enabling side-by-side comparison

## Phase Details

### Phase 1: End-to-End Reward Integration
**Goal**: Shaped rewards from any registered strategy replace raw rewards in the training loss, so gradient updates reflect the chosen reward shaping -- not just raw correctness scores
**Depends on**: Nothing (first phase)
**Requirements**: RINT-01, RINT-02, RINT-03, RINT-04
**Success Criteria** (what must be TRUE):
  1. When a reward shaping strategy is configured in DebateMetricStreamerConfig, the shaped reward values (not raw correctness scores) are the values used by the GRPO/RLOO learner for gradient computation
  2. Per-role reward dicts from difference_rewards, coma_advantage, and reward_mixing are mapped to the correct per-turn rewards based on role labels in the debate transcript
  3. Running the identity strategy produces numerically identical loss values and gradient norms as running with no strategy configured (regression confirmed)
  4. A unit/integration test demonstrates that shaped rewards differ from raw rewards for at least one non-identity strategy on a sample input
**Plans**: TBD

Plans:
- [ ] 01-01: Wire reward shaping into DebateMetricStreamer and verify gradient path
- [ ] 01-02: Implement per-role reward mapping and identity regression check

### Phase 2: Experiment Configuration
**Goal**: Five SWEEP configs exist that are identical except for reward shaping strategy, ready to submit to flex queue for an apples-to-apples comparison
**Depends on**: Phase 1
**Requirements**: ECFG-01, ECFG-02, ECFG-03, ECFG-04, ECFG-05, ECFG-06, ECFG-07, ECFG-08
**Success Criteria** (what must be TRUE):
  1. Each of the 5 SWEEP configs (identity, difference_rewards, potential_based, coma_advantage, reward_mixing) can be loaded and validated without errors
  2. Diffing any two configs shows differences only in reward_shaping_strategy and reward_shaping_params fields -- all other hyperparameters, data paths, queue settings, and model configs are identical
  3. All 5 configs target post-training flex queue with dev-low priority and reference MATH-500 online data from the existing GCS path
  4. Strategy-specific params are set correctly: potential_based uses debate_length potential with gamma=0.99, coma_advantage uses n_rollouts_per_prompt=4, reward_mixing uses alpha=0.5
**Plans**: TBD

Plans:
- [ ] 02-01: Create base config and generate 5 strategy-specific SWEEP configs

### Phase 3: Observability and Comparison
**Goal**: All 5 training runs log shaped reward metrics to WandB with strategy metadata, producing comparable learning curves in a single project
**Depends on**: Phase 1, Phase 2
**Requirements**: OBSV-01, OBSV-02, OBSV-03
**Success Criteria** (what must be TRUE):
  1. Each training run logs a shaped_reward metric per step that is distinct from the raw correctness_score metric
  2. Each WandB run includes the reward_shaping_strategy name in its run metadata/config, enabling filtering and grouping by strategy
  3. Training loss, accuracy, and shaped reward curves from all 5 runs are visible on the same WandB project dashboard for side-by-side comparison
**Plans**: TBD

Plans:
- [ ] 03-01: Add shaped reward and strategy metadata to WandB logging
- [ ] 03-02: Submit all 5 runs and validate comparison dashboard

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. End-to-End Reward Integration | 0/2 | Not started | - |
| 2. Experiment Configuration | 0/1 | Not started | - |
| 3. Observability and Comparison | 0/2 | Not started | - |
