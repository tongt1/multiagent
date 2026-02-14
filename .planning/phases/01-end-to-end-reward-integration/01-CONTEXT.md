# Phase 1: End-to-End Reward Integration - Context

**Gathered:** 2026-02-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire shaped rewards from reward shaping modules through DebateMetricStreamer into GRPO/RLOO gradient computation. Shaped rewards must replace raw correctness scores in the training loss so gradient updates reflect the chosen strategy. Identity strategy must regress to baseline behavior.

</domain>

<decisions>
## Implementation Decisions

### Role-to-Turn Reward Mapping
- Broadcast by role label: each turn gets the shaped reward for its role (all solver turns get solver's reward, all verifier turns get verifier's reward)
- Role identification via explicit role field on each turn in the debate transcript
- Solver and verifier turns contribute to the training loss; judge turns are excluded (zero reward)
- All roles' shaped rewards contribute to learning (model plays all roles via prompt engineering)
- If a strategy doesn't produce a reward for a particular role, fall back to the raw correctness score for that role's turns

### Reward Source
- Rewards are deterministic, computed via sympy and robust parsing -- not from a judge model
- No judge role in the reward pipeline; judge turns are excluded from gradient computation

### Unconfigured Fallback Behavior
- No strategy configured: default to identity (passes raw rewards through unchanged)
- Invalid strategy name: error at config load time (fail fast, before training starts)
- Fully backward compatible: existing configs with no reward_shaping fields work exactly as before
- Log info message once at startup when defaulting to identity: "No reward shaping strategy configured, using identity"

### Turn-Level Reward Granularity
- Uniform reward across all turns of a role (all solver turns get the same shaped reward)
- Potential-based shaping collapses to uniform per-role reward (sum/average), consistent interface across all strategies
- Pipeline supports both single-round and multi-round debates with the same uniform-per-role logic

### Regression Verification
- Identity vs no-strategy comparison uses torch.allclose() with small tolerance (not exact float equality)
- Integration test with gradient check: run 5-10 training steps, compare gradient norms between identity and a non-identity strategy
- Lightweight CI check: fast 1-2 step version runs on every CI push to catch regressions

### Claude's Discretion
- Exact tolerance values for allclose comparison
- Internal data structures for reward mapping pipeline
- How potential-based rewards are collapsed (sum vs average)
- Test fixture design and mock strategy setup

</decisions>

<specifics>
## Specific Ideas

- The critical verification is that shaped rewards flow to GRPO/RLOO loss, not just appear as WandB metrics
- Identity strategy must be a true no-op path (within floating point tolerance)
- The 5-10 step integration test should show gradient norms differ between identity and difference_rewards (or another non-identity strategy)

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 01-end-to-end-reward-integration*
*Context gathered: 2026-02-14*
