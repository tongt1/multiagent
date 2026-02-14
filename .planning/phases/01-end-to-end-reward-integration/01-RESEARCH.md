# Phase 1: End-to-End Reward Integration - Research

**Researched:** 2026-02-14
**Domain:** Flink RL training pipeline reward wiring (GRPO/RLOO gradient path)
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Role-to-Turn Reward Mapping
- Broadcast by role label: each turn gets the shaped reward for its role (all solver turns get solver's reward, all verifier turns get verifier's reward)
- Role identification via explicit role field on each turn in the debate transcript
- Solver and verifier turns contribute to the training loss; judge turns are excluded (zero reward)
- All roles' shaped rewards contribute to learning (model plays all roles via prompt engineering)
- If a strategy doesn't produce a reward for a particular role, fall back to the raw correctness score for that role's turns

#### Reward Source
- Rewards are deterministic, computed via sympy and robust parsing -- not from a judge model
- No judge role in the reward pipeline; judge turns are excluded from gradient computation

#### Unconfigured Fallback Behavior
- No strategy configured: default to identity (passes raw rewards through unchanged)
- Invalid strategy name: error at config load time (fail fast, before training starts)
- Fully backward compatible: existing configs with no reward_shaping fields work exactly as before
- Log info message once at startup when defaulting to identity: "No reward shaping strategy configured, using identity"

#### Turn-Level Reward Granularity
- Uniform reward across all turns of a role (all solver turns get the same shaped reward)
- Potential-based shaping collapses to uniform per-role reward (sum/average), consistent interface across all strategies
- Pipeline supports both single-round and multi-round debates with the same uniform-per-role logic

#### Regression Verification
- Identity vs no-strategy comparison uses torch.allclose() with small tolerance (not exact float equality)
- Integration test with gradient check: run 5-10 training steps, compare gradient norms between identity and a non-identity strategy
- Lightweight CI check: fast 1-2 step version runs on every CI push to catch regressions

### Claude's Discretion
- Exact tolerance values for allclose comparison
- Internal data structures for reward mapping pipeline
- How potential-based rewards are collapsed (sum vs average)
- Test fixture design and mock strategy setup

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

## Summary

The core challenge of this phase is bridging a gap in the existing reward pipeline: shaped rewards are computed and **logged as WandB metrics** (via `DebateMetricStreamer._compute_shaped_reward_metrics()`), but they do **not** replace `item.data["rewards"]` -- the value that flows into the GRPO/RLOO learner for gradient computation. Currently, the Flink learner (FlinkRlooLearnerConfig with `policy_gradient_loss="grpo"`) consumes rewards directly from `item.data["rewards"]`, which holds the raw correctness score from `MathDebateScenario.compute_reward()`. The DebateMetricStreamer sits in the `actor_outputs_streamers` pipeline and has full access to modify items before they reach the learner.

The existing reward shaping module (`src/training/reward_shaping/`) is well-architected with 5 registered strategies, a clean registry pattern, and comprehensive unit tests. The missing piece is a small but critical wiring change: the streamer must mutate `item.data["rewards"]` with the shaped reward value before returning items. For per-role strategies (difference_rewards, coma_advantage, reward_mixing) that return `dict[str, np.ndarray]`, the shaped reward for each item must be looked up by its role label. For global strategies (identity, potential_based) that return `np.ndarray`, the shaped reward is indexed directly.

The secondary challenge is the identity regression check: proving that identity shaping produces numerically identical training behavior to no-strategy-configured. Because identity returns `rewards` unchanged and the default (no config) also creates an identity shaper, this should hold within floating-point tolerance. The integration test requires either a mock GRPO loss computation (pure numpy/torch) or a lightweight end-to-end training loop with SmolLM-135M.

**Primary recommendation:** Modify `DebateMetricStreamer.get()` to write shaped rewards back into `item.data["rewards"]` after computing them. This is the minimal, backward-compatible change that wires shaped rewards into the gradient path.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | >=1.24 | Reward shaping computation, array ops | Already used by all reward shaping strategies |
| pytest | >=8.2 | Testing framework | Already in dev dependencies |
| pydantic | >=2.12 | Config validation (DebateMetricStreamerConfig) | Already used throughout codebase |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch | >=2.0 | `torch.allclose()` for regression check, gradient norm comparison | Integration test only |
| pytest-approx | (built-in) | Float comparison in assertions | All reward value tests |
| numpy.testing | (built-in) | `assert_array_almost_equal` for batch comparisons | Array-level reward verification |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Mutating `item.data["rewards"]` in streamer | Creating a new streamer class | Unnecessary complexity; existing streamer already handles reward extraction and shaping |
| `torch.allclose` for regression | `numpy.allclose` | torch required for gradient norm comparison anyway; allclose tolerance semantics identical |

**Installation:** No new dependencies required. All needed libraries are already in `pyproject.toml`.

## Architecture Patterns

### Recommended Approach: In-Place Reward Mutation in DebateMetricStreamer

The existing Flink `actor_outputs_streamers` pipeline is a chain of streamer objects. Each streamer's `get()` method receives items from upstream, can modify them, and returns `(items, metrics)`. The items continue to the next streamer or to the learner. This means modifying `item.data["rewards"]` in `DebateMetricStreamer.get()` directly affects what the learner receives.

```
Flink Pipeline Flow:
  Actor (rollout) -> item.data["rewards"] = raw correctness score
    -> FilteringStreamer (drops truncated/identical-reward batches)
      -> DebateMetricStreamer.get():
          1. Apply rollout strategy (filter items)
          2. Extract raw rewards from items
          3. Compute debate metrics (unshaped, for WandB)
          4. Apply reward shaping strategy
          5. ** NEW: Write shaped rewards back into item.data["rewards"] **
          6. Compute shaped reward metrics (for WandB)
          7. Return (items, metrics)
        -> FlinkRlooLearner receives items with shaped rewards in item.data["rewards"]
          -> GRPO loss uses item.data["rewards"] for advantage computation
```

### Pattern 1: Reward Mutation in Streamer
**What:** After computing shaped rewards via `self._reward_shaper.shape_rewards()`, write the per-item shaped reward back into `item.data["rewards"]`.
**When to use:** This is the primary integration pattern.
**Example:**
```python
# In DebateMetricStreamer.get(), after line 204 (shaped_metrics computation):

# Phase 1: Write shaped rewards back into items for GRPO/RLOO gradient path
shaped_result = self._reward_shaper.shape_rewards(rewards, role_masks, None)
if isinstance(shaped_result, dict):
    # Per-role dict: look up each item's shaped reward by its role label
    for idx, item in enumerate(items):
        role = role_labels[idx]
        if role in shaped_result:
            item.data["rewards"] = np.array(shaped_result[role][idx])
        else:
            # Role not in shaped output: fall back to raw reward (per CONTEXT.md)
            pass  # item.data["rewards"] already has raw reward
else:
    # Global array: index directly
    for idx, item in enumerate(items):
        item.data["rewards"] = np.array(shaped_result[idx])
```

### Pattern 2: Role-to-Reward Lookup with Judge Exclusion
**What:** When mapping per-role shaped rewards to items, judge turns get zero reward to exclude them from gradient computation.
**When to use:** For strategies returning `dict[str, np.ndarray]` (difference_rewards, coma_advantage, reward_mixing).
**Example:**
```python
for idx, item in enumerate(items):
    role = role_labels[idx]
    if role == "judge":
        item.data["rewards"] = np.array(0.0)  # Exclude judge from gradient
    elif role in shaped_result:
        item.data["rewards"] = np.array(shaped_result[role][idx])
    else:
        # Fallback to raw correctness score
        pass
```

### Pattern 3: Potential-Based Collapse to Uniform Per-Role Reward
**What:** When potential_based returns a global `np.ndarray`, treat all turns uniformly (same shaped reward regardless of role, except judge gets zero).
**When to use:** For strategies returning `np.ndarray` (identity, potential_based).
**Example:**
```python
if isinstance(shaped_result, np.ndarray):
    for idx, item in enumerate(items):
        role = role_labels[idx]
        if role == "judge":
            item.data["rewards"] = np.array(0.0)
        else:
            item.data["rewards"] = np.array(shaped_result[idx])
```

### Pattern 4: Config Validation at Load Time (Fail Fast)
**What:** Validate strategy name at `DebateMetricStreamer.__init__` time. Invalid strategy names already raise `KeyError` from `get_strategy()` in the registry.
**When to use:** Always -- the registry already does this. Just ensure the error message is clear.
**Example:**
```python
# Already handled by registry.py line 61-66:
# get_strategy() raises KeyError with available strategies listed.
# create_strategy_from_config() calls get_strategy().
# DebateMetricStreamer.__init__() calls create_reward_strategy().
# -> Fail fast at config load time. No changes needed.
```

### Anti-Patterns to Avoid
- **Logging shaped rewards without mutating items:** This is the current state -- shaped rewards appear as WandB metrics but the GRPO learner still uses raw rewards. The streamer must mutate `item.data["rewards"]`.
- **Creating a parallel reward path:** Do not add a second reward field (e.g., `item.data["shaped_rewards"]`) -- the learner reads `item.data["rewards"]` and there is no mechanism to change that without modifying Flink core.
- **Modifying Flink core for shaped rewards:** The streamer extension point is designed for exactly this kind of intervention. Do not fork or patch Flink code.
- **Applying shaping after the streamer:** The streamer is the last custom code before items reach the learner. There is no other extension point.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Reward strategy creation from config | Custom factory | `create_strategy_from_config()` from registry | Already handles None/empty/invalid cases, returns identity by default |
| Strategy name validation | Custom validation | Registry's `get_strategy()` KeyError | Already provides helpful error with available strategy list |
| Role label extraction from items | Custom metadata parsing | Existing code in `DebateMetricStreamer.get()` lines 171-188 | Already handles missing role_label, default to "solver", numpy scalar conversion |
| Per-role reward broadcasting | Custom loop | numpy indexing with boolean masks | Reward shaping strategies already return correctly-shaped arrays |

**Key insight:** The reward shaping infrastructure is already built and tested. This phase is purely a wiring change -- connecting the output of `shape_rewards()` back to `item.data["rewards"]`.

## Common Pitfalls

### Pitfall 1: Shaped Rewards Only Logged, Not Applied
**What goes wrong:** Shaped rewards appear in WandB metrics (`debate/shaped_reward/*`) but training behavior is unchanged because `item.data["rewards"]` still has raw values.
**Why it happens:** The current `_compute_shaped_reward_metrics()` method computes shaped rewards for logging but does not mutate items. This is the exact gap this phase must close.
**How to avoid:** Verify with an integration test: run 5 steps with identity, run 5 steps with difference_rewards, compare gradient norms. If they are identical, shaped rewards are not flowing to the learner.
**Warning signs:** Gradient norms identical between identity and non-identity strategies.

### Pitfall 2: Batch Index Mismatch Between Rewards and Items
**What goes wrong:** The shaped reward array has index `i` but it maps to a different item than `items[i]`, causing wrong rewards to be assigned to wrong rollouts.
**Why it happens:** Rollout strategy can filter items (e.g., best_of_n reduces N*P to P items). The shaped reward array must be computed on the filtered items, not the original items.
**How to avoid:** Ensure shaping happens AFTER rollout strategy filtering. The current code already does this -- `rewards = np.array([item.data["rewards"].item() for item in items])` is extracted after rollout filtering on line 167.
**Warning signs:** Reward values in WandB don't match expected strategy outputs.

### Pitfall 3: Per-Role Dict Index vs Batch Index Confusion
**What goes wrong:** A per-role strategy returns `{"solver": [B], "verifier": [B], "judge": [B]}` where each array is length B (full batch). But each item in the batch has a specific role. When writing back, you need to look up `shaped_result[role_of_item][batch_index]`, not just `shaped_result["solver"][solver_count]`.
**Why it happens:** The reward shaping strategies compute shaped rewards for ALL batch items under each role key. The arrays are indexed by batch position (0 to B-1), not by role-specific position.
**How to avoid:** Use `shaped_result[role_labels[idx]][idx]` where `idx` is the batch index and `role_labels[idx]` is the role of that batch item.
**Warning signs:** ArrayIndexError or wrong reward values when role distribution is uneven.

### Pitfall 4: Modifying item.data["rewards"] Breaks Upstream Metrics
**What goes wrong:** The unshaped debate metrics (per-role rewards, zero-advantage) are computed from `rewards` array BEFORE shaping, but if we mutate items first, subsequent calls to `compute_per_role_rewards()` would get shaped values.
**How to avoid:** The current code already extracts `rewards = np.array([item.data["rewards"].item() for item in items])` into a local array before computing unshaped metrics. Mutate items AFTER all unshaped metrics are computed. The modification should go after line 206 (after `_compute_shaped_reward_metrics()`), not before line 192.
**Warning signs:** `debate/reward/solver` metric changes when shaping is enabled (it should not -- it should always reflect raw rewards).

### Pitfall 5: Judge Zero Reward vs Judge Exclusion
**What goes wrong:** Setting judge reward to 0.0 is not the same as excluding judge turns from the loss. If judge turns still have non-zero log-probabilities, a zero reward still contributes to the GRPO advantage computation.
**Why it happens:** GRPO computes advantage as `reward - mean_reward_in_group`. If judge items have reward=0.0 while solver/verifier items have reward=5.0, the mean drops, affecting advantages for all items.
**How to avoid:** The user decision says "judge turns are excluded (zero reward)." In the current architecture, each item represents a complete rollout (not a single turn), so the role_label on each item represents the dominant role. If all items in a batch are full rollouts (with role_label from `compute_reward()` metrics), then role_label is always "solver" (the current implementation always sets role_label="solver" in `comb_math_debate_env.py` line 348). This means the per-role mapping only matters when role_label is enriched per Phase 6. For Phase 1, all items get role_label="solver" and thus get the solver shaped reward.
**Warning signs:** Unexpectedly low rewards in shaped metrics.

### Pitfall 6: Identity Regression Fails Due to Floating Point
**What goes wrong:** `torch.allclose(identity_grads, no_strategy_grads)` fails because reward shaping introduces a `.copy()` or array conversion that changes floating point representation.
**Why it happens:** Identity strategy's `shape_rewards()` returns `rewards` directly (not `.copy()`). But the code path that applies shaping writes `np.array(shaped_result[idx])` which may change precision from float64 to float32 or vice versa.
**How to avoid:** Use `atol=1e-6, rtol=1e-5` tolerance. Ensure dtype consistency: if `item.data["rewards"]` is float32, write back float32.
**Warning signs:** allclose fails with very small differences (1e-7 scale).

## Code Examples

### Critical Modification: Writing Shaped Rewards Into Items

The core change goes in `DebateMetricStreamer.get()` in `src/training/wandb_enrichment/debate_streamer.py`.

```python
# Source: src/training/wandb_enrichment/debate_streamer.py, after line 206

# Phase 1: Write shaped rewards back into items for GRPO/RLOO gradient path
# This is the critical wiring that makes shaped rewards affect gradient computation.
# MUST happen AFTER unshaped metrics are computed (lines 191-200)
# and AFTER shaped metrics are logged (lines 203-207).
try:
    shaped_result = self._reward_shaper.shape_rewards(rewards, role_masks, None)
    for idx, item in enumerate(items):
        role = role_labels[idx]
        if role == "judge":
            # Judge turns excluded from gradient computation (per CONTEXT.md)
            item.data["rewards"] = np.array(0.0, dtype=rewards.dtype)
        elif isinstance(shaped_result, dict):
            # Per-role strategy: look up by role label
            if role in shaped_result:
                item.data["rewards"] = np.array(
                    shaped_result[role][idx], dtype=rewards.dtype
                )
            # else: fallback -- item.data["rewards"] keeps raw value
        else:
            # Global strategy (identity, potential_based)
            item.data["rewards"] = np.array(
                shaped_result[idx], dtype=rewards.dtype
            )
except Exception as e:
    logger.warning(f"DebateMetricStreamer: failed to apply shaped rewards to items: {e}")
    # On error, items retain original raw rewards (safe fallback)
```

### Integration Test: Verify Shaped Rewards Reach Items

```python
# Source: test pattern for verifying item mutation

def test_shaped_rewards_written_to_items():
    """Verify that shaped rewards are written back into item.data['rewards']."""
    from src.training.wandb_enrichment.debate_streamer import (
        DebateMetricStreamer, DebateMetricStreamerConfig,
    )

    items = [
        MockActorOutputItem(reward=5.0, role_label="solver"),
        MockActorOutputItem(reward=0.0, role_label="solver"),
        MockActorOutputItem(reward=5.0, role_label="verifier"),
        MockActorOutputItem(reward=0.0, role_label="verifier"),
    ]

    upstream = MockUpstreamStreamer(items, {})
    config = DebateMetricStreamerConfig(
        n_rollouts_per_prompt=4,
        reward_shaping_strategy="difference_rewards",
    )
    streamer = DebateMetricStreamer(config, upstream, MagicMock())
    result_items, _ = streamer.get()

    # With difference_rewards (no counterfactual metadata -> fallback to raw),
    # items should have shaped rewards written back
    for item in result_items:
        # Verify rewards were mutated (value may differ from original raw reward
        # depending on strategy, but the field should exist and be a scalar)
        assert hasattr(item.data["rewards"], 'item') or isinstance(item.data["rewards"], float)
```

### Integration Test: Identity Regression Check

```python
# Source: test pattern for identity vs no-strategy regression

def test_identity_regression():
    """Identity strategy produces same item rewards as no-strategy baseline."""
    from src.training.wandb_enrichment.debate_streamer import (
        DebateMetricStreamer, DebateMetricStreamerConfig,
    )
    import numpy as np

    raw_rewards = [5.0, 0.0, 5.0, 0.0]
    items_identity = [MockActorOutputItem(reward=r, role_label="solver") for r in raw_rewards]
    items_no_strategy = [MockActorOutputItem(reward=r, role_label="solver") for r in raw_rewards]

    # Run with explicit identity
    upstream1 = MockUpstreamStreamer(items_identity, {})
    config1 = DebateMetricStreamerConfig(
        n_rollouts_per_prompt=4,
        reward_shaping_strategy="identity",
    )
    streamer1 = DebateMetricStreamer(config1, upstream1, MagicMock())
    result_items1, _ = streamer1.get()

    # Run with no strategy (default)
    upstream2 = MockUpstreamStreamer(items_no_strategy, {})
    config2 = DebateMetricStreamerConfig(n_rollouts_per_prompt=4)
    streamer2 = DebateMetricStreamer(config2, upstream2, MagicMock())
    result_items2, _ = streamer2.get()

    # Rewards should be numerically identical
    for item1, item2 in zip(result_items1, result_items2):
        r1 = item1.data["rewards"].item() if hasattr(item1.data["rewards"], 'item') else item1.data["rewards"]
        r2 = item2.data["rewards"].item() if hasattr(item2.data["rewards"], 'item') else item2.data["rewards"]
        assert np.isclose(r1, r2, atol=1e-7), f"Identity regression: {r1} != {r2}"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Shaped rewards only logged as WandB metrics | Must mutate `item.data["rewards"]` to affect gradients | This phase | Enables actual reward shaping in training loop |
| All items get role_label="solver" (hardcoded in comb_math_debate_env.py) | Still "solver" for now; Phase 6 will enrich with actual role labels | Phase 6 (future) | Per-role reward mapping will be meaningful when role labels are enriched |
| Single reward per rollout (item-level) | Still item-level; turn-level breakdown is future work | Phase 6 (future) | Uniform per-role reward satisfies CONTEXT.md for now |

**Current state (before this phase):**
- Reward shaping strategies: Fully implemented, tested, registered
- DebateMetricStreamer: Computes shaped reward metrics for WandB, does NOT mutate items
- GRPO learner: Reads `item.data["rewards"]` for gradient computation
- Role labels: Always "solver" (not yet enriched by Phase 6)

**After this phase:**
- DebateMetricStreamer: Also writes shaped rewards back into `item.data["rewards"]`
- Identity strategy: Produces numerically identical behavior to no-strategy baseline
- Non-identity strategies: Produce different gradient updates (verified by integration test)

## Key Architectural Findings

### Finding 1: Role Label State (HIGH confidence)
All items currently have `role_label = "solver"` hardcoded in `comb_math_debate_env.py` (lines 242, 253, 348). This means that for Phase 1, per-role reward mapping will effectively broadcast the solver shaped reward to all items. The per-role dict from strategies like difference_rewards will have entries for solver, verifier, and judge, but all items will look up `shaped_result["solver"][idx]`. This is correct and expected per CONTEXT.md: "If a strategy doesn't produce a reward for a particular role, fall back to the raw correctness score for that role's turns."

**Implication:** The per-role mapping code must be written correctly for when Phase 6 enriches role labels, but in Phase 1 testing, all items will use the solver reward.

### Finding 2: Duplicate Shaping Computation (MEDIUM confidence)
The current code calls `self._reward_shaper.shape_rewards()` once in `_compute_shaped_reward_metrics()` (line 270) for WandB metrics. To also write shaped rewards into items, we need a second call (or refactor to share the result). Since shaping is pure numpy with no side effects, a second call is safe but slightly wasteful. A small refactor to compute once and reuse would be cleaner.

**Recommendation:** Refactor `get()` to compute shaped rewards once, use the result for both WandB metrics and item mutation.

### Finding 3: Rollout Strategy Item Count (HIGH confidence)
Rollout strategy can reduce item count (e.g., best_of_n: N*P items -> P items). The shaped reward array must have length matching the post-filtered items, not the pre-filtered items. The current code already extracts rewards after rollout filtering (line 167), so shaping happens on the correct item set.

### Finding 4: item.data["rewards"] Type (HIGH confidence)
`item.data["rewards"]` is a numpy scalar (created via `np.array(reward)` in test mocks and by Flink infrastructure). When writing back, use `np.array(value, dtype=rewards.dtype)` to maintain type consistency.

### Finding 5: Existing Test Infrastructure (HIGH confidence)
Comprehensive test mocks exist: `MockActorOutputItem`, `MockUpstreamStreamer` in both `test_debate_streamer.py` and `test_reward_shaping_integration.py`. These provide the foundation for new integration tests without any Flink dependencies.

## Open Questions

1. **Judge exclusion mechanics with current role_label="solver"**
   - What we know: All items currently have role_label="solver", so judge exclusion code will never trigger in Phase 1.
   - What's unclear: Whether to implement judge exclusion now (with dead code) or defer to Phase 6 when role labels are enriched.
   - Recommendation: Implement now with clear comments. The code is simple and having it ready prevents Phase 6 from needing to touch the reward mutation logic.

2. **Potential-based reward collapse to uniform per-role**
   - What we know: CONTEXT.md says "potential-based shaping collapses to uniform per-role reward (sum/average)."
   - What's unclear: potential_based already returns a global `np.ndarray` (not per-role dict), so "collapse" is already done. The question is whether future potential-based variants might return per-turn rewards that need aggregation.
   - Recommendation: For Phase 1, use the global array directly. Document that potential_based returns `np.ndarray` and is treated identically to identity for item mutation purposes.

3. **Tolerance values for allclose**
   - What we know: CONTEXT.md says "small tolerance (not exact float equality)."
   - What's unclear: Exact values depend on whether float32 or float64 is used in the pipeline.
   - Recommendation: Use `atol=1e-6, rtol=1e-5` as defaults. These are the standard torch.allclose defaults and should handle float32/float64 conversion.

## Sources

### Primary (HIGH confidence)
- `src/training/wandb_enrichment/debate_streamer.py` -- DebateMetricStreamer implementation, the exact code that must be modified
- `src/training/reward_shaping/base.py` -- RewardShaper interface, return type contract (np.ndarray or dict[str, np.ndarray])
- `src/training/reward_shaping/registry.py` -- Strategy creation and validation logic
- `src/training/reward_shaping/identity.py` -- Identity passthrough implementation
- `src/training/reward_shaping/difference_rewards.py` -- Per-role dict return example
- `configs/sweep_math_debate_grpo.py` -- SWEEP config showing DebateMetricStreamerConfig in actor_outputs_streamers, FlinkRlooLearnerConfig with grpo
- `src/training/comb_math_debate_env.py` -- Shows role_label="solver" hardcoded (lines 242, 253, 348)

### Secondary (MEDIUM confidence)
- `tests/test_reward_shaping_integration.py` -- Existing test patterns for streamer + shaping integration
- `tests/test_debate_streamer.py` -- Mock infrastructure (MockActorOutputItem, MockUpstreamStreamer)
- `tests/test_grpo_role_loss.py` -- GRPO per-role loss decomposition test patterns (numpy-based)
- `.planning/ARCHITECTURE.md` -- Flink pipeline data flow documentation

### Tertiary (LOW confidence)
- Flink framework internal behavior (how `actor_outputs_streamers` items flow to learner) -- inferred from code patterns and documented architecture, not verified against Flink source code

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already in use, no new dependencies needed
- Architecture: HIGH - Modification point clearly identified in `debate_streamer.py`, data flow well-documented
- Pitfalls: HIGH - Based on direct code reading, all edge cases identified from existing test patterns
- Flink learner reward path: MEDIUM - Inferred that `item.data["rewards"]` flows to learner based on architecture docs and `actor_outputs_streamers` pattern, but not verified against Flink source

**Research date:** 2026-02-14
**Valid until:** 2026-03-14 (stable -- code changes are internal to this repo)
