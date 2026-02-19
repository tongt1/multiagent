# Reward Shaping Comparison Experiment

## What This Is

End-to-end integration and comparison of 4 reward shaping strategies (difference rewards, potential-based, COMA counterfactual advantage, reward mixing) plus an identity baseline for multi-agent debate RL training. Uses SmolLM-135M on MATH-500 with post-training flex queue to produce WandB learning curves comparing all 5 approaches.

## Core Value

All 5 reward strategies must be fully wired end-to-end — from the reward shaping module through DebateMetricStreamer into the training loss — so that shaped rewards actually affect gradient updates and produce meaningful WandB comparison curves.

## Requirements

### Validated

- ✓ Reward shaping strategies implemented — `src/training/reward_shaping/` has identity, difference_rewards, potential_based, coma_advantage, reward_mixing with registry pattern — existing
- ✓ SmolLM-135M run config — `run_configs/smollm_135m_rloo_math.run` with MATH-500 data paths — existing
- ✓ Math debate Comb environment — `src/training/comb_math_debate_env.py` with `@register_builder("math_debate")` — existing
- ✓ SWEEP config template — `configs/sweep_math_debate_grpo.py` with commented reward shaping params — existing
- ✓ DebateMetricStreamerConfig — accepts `reward_shaping_strategy` and `reward_shaping_params` — existing

### Active

- [ ] Wire reward shaping end-to-end through DebateMetricStreamer so shaped rewards replace raw rewards in training loss
- [ ] Verify shaped rewards flow through to GRPO/RLOO gradient computation (not just logged as metrics)
- [ ] Create 5 SWEEP configs for SmolLM-135M: identity (baseline), difference_rewards, potential_based, coma_advantage, reward_mixing
- [ ] Configure all configs for post-training flex queue, dev-low priority, MATH-500 online data
- [ ] Submit all 5 runs and confirm WandB logging captures per-strategy reward curves
- [ ] Produce comparable WandB curves: training loss, accuracy, shaped reward magnitude across all 5 strategies

### Out of Scope

- Production-scale training (this is 135M model for fast iteration) — too expensive for comparison study
- Multi-model training (solver/verifier as separate models) — orthogonal to reward shaping comparison
- Rollout strategies (best-of-N, self-consistency) — separate concern, keep identity rollout
- Hyperparameter tuning per reward strategy — use reasonable defaults first, tune later
- Automated analysis/report — will analyze WandB curves manually

## Context

The codebase has a multi-agent debate RL pipeline where a single model plays solver, verifier, and judge roles through prompt engineering. Reward shaping strategies have been implemented as pluggable modules with a registry pattern, but they're not yet wired end-to-end into the training loop.

The `DebateMetricStreamerConfig` in the SWEEP config accepts `reward_shaping_strategy` and `reward_shaping_params` fields (currently commented out). The critical integration point is ensuring the shaped rewards from `DebateMetricStreamer` actually replace the raw rewards used by the RLOO/GRPO learner for gradient computation — not just appear as WandB metrics.

SmolLM-135M is chosen for fast iteration (fits on 1 GPU, trains quickly) to validate the reward shaping pipeline before scaling up.

## Constraints

- **Model**: SmolLM-135M (fast iteration, validate pipeline before scaling)
- **Queue**: post-training flex queue, dev-low priority (shared cluster resources)
- **Data**: MATH-500 online (already configured in run config at `gs://cohere-dev-central-2/comb/data/math_500/`)
- **Budget**: 5 concurrent runs on flex queue

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| SmolLM-135M over 7B | Fast iteration to validate pipeline, minutes not hours | — Pending |
| All 5 strategies in single comparison | Apples-to-apples comparison with same hyperparams | — Pending |
| Dev-low priority on flex queue | Minimize cost while getting reasonable turnaround | — Pending |
| Default strategy hyperparams | Fair comparison baseline before per-strategy tuning | — Pending |

---
*Last updated: 2026-02-14 after initialization*
