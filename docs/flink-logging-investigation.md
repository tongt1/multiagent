# Flink Logging Capabilities Investigation

**Date:** 2026-02-04
**Phase:** 05-wandb-logging-enrichment
**Purpose:** Document existing Flink/Fax logging infrastructure for debate training integration

---

## Executive Summary

Flink is JAX-based and uses the `fax` framework for training orchestration. It provides:
- **Gradient norm logging** via `advanced_logging` config
- **Debug data persistence** to Parquet files via `save_debug_data_from_batch()`
- **W&B integration** via `accumulating_plotter` and plotter callbacks
- **Generation logging** via `log_train_generations_every_steps`

Our SWEEP configs already enable `save_debug_data_every_steps=1` and `log_train_generations_every_steps=25`.

---

## 1. Gradient Norm Logging

### Configuration

**Location:** `fax/config/config_run.py`

**Config class:** `AdvancedLoggingConfig`

```python
class AdvancedLoggingConfig(FaxConfig):
    enabled: bool = False
    norm_granularity: Tuple[AdvancedLoggingGranularity, ...] = (AdvancedLoggingGranularity.GLOBAL,)
    norm_target_trees: Tuple[AdvancedLoggingTarget, ...] = (
        AdvancedLoggingTarget.GRAD,
        AdvancedLoggingTarget.UPDATE,
    )
    log_histograms: bool = False
    log_moe_histograms: bool = True
    hist_interval: int = 1
    hist_granularity: Tuple[AdvancedLoggingGranularity, ...] = (AdvancedLoggingGranularity.GLOBAL,)
    hist_target_trees: Tuple[AdvancedLoggingTarget, ...] = (AdvancedLoggingTarget.GRAD,)
    hist_bins: int = 20
    hist_proportion: float = 0.01
    log_opt_aux_metrics: bool = True
```

### How to Enable

**Patch the run config:**

```python
patch_run_config = {
    "advanced_logging": {
        "enabled": True,
        "norm_granularity": ("global",),
        "norm_target_trees": ("grad", "update"),
        "log_histograms": False,  # Disable to reduce overhead
    }
}
```

### Expected W&B Metric Keys

When enabled, Flink logs gradient norms to W&B with keys following the pattern:
- Global gradient norm: `grad/global_norm` or `grad_global_norm` (exact key depends on fax version)
- Update norm: `update/global_norm` or `update_global_norm`

**Note:** `log_grad_norms` is **deprecated** in favor of `advanced_logging`.

---

## 2. Debug Data Saving

### Configuration

**Location:** `post_training/flink/flink_zord.py`

**Config fields:**
- `save_debug_data_every_steps: int | None = None` — Frequency of saving debug data to disk
- `max_number_wandb_logged_samples: int | None = None` — Max number of samples to log to W&B at each step

**Already enabled in our SWEEP configs:**
- `save_debug_data_every_steps=1` (save every step)
- `log_train_generations_every_steps=25` (log to W&B every 25 steps)

### Debug Data Format

**Function:** `post_training/relax/metrics_processing.py::save_debug_data_from_batch()`

**Output path pattern:**
```
{output_dir}/flink_logs/batch_debug_data_{train_or_eval}_{step}_{fname_suffix}.parquet
```

**Example:**
```
/path/to/output/flink_logs/batch_debug_data_train_100.parquet
```

**Parquet schema (from BatchDebugData class):**

| Column              | Type                  | Description                                          |
| ------------------- | --------------------- | ---------------------------------------------------- |
| env_name            | list[str \| None]     | COMB environment names                               |
| trajectory          | list[str \| None]     | Rendered trajectories (multi-turn string)            |
| agent_trajectories  | list[str \| None]     | AgentTrajectory JSON dumps                           |
| exception_info      | list[str \| None]     | Exception names/tracebacks for failed trajectories   |
| reward              | list[float \| None]   | Rewards for trajectories                             |
| reward_metrics      | list[dict \| None]    | Reward metrics (dict per trajectory)                 |
| reward_text_info    | list[str \| None]     | Textual reward info (extracted answer, etc.)         |
| unique_sample_id    | list[str]             | Sample IDs (dataset + index)                         |

**Note:** For Phase 7 (debate analysis), we will **extend this schema** to include per-role rewards, KL divergences, and debate metadata. A `schema_version` column will be added to support schema evolution.

### How Data is Saved

1. `FlinkZord._maybe_save_debug_data()` checks if `step % save_debug_data_every_steps == 0`
2. Calls `metrics_processing.save_debug_data_from_batch()`
3. Extracts `BatchDebugData` from batch metadata
4. Saves to Parquet (or CSV if Parquet fails)

---

## 3. W&B Callback Integration

### Where Flink Logs to W&B

**Primary integration:** `fax/plotting/accumulating_plotter.py`

**Flow:**
1. Flink uses an `AccumulatingPlotter` that wraps one or more plotters (W&B, TensorBoard, etc.)
2. Metrics are accumulated via `plotter.accumulate_scalar(name, value, mode)`
3. Periodically flushed via `plotter.log_accumulated_scalars(step)`

**Accumulation modes:**
- `NONE` — Log instantaneous values
- `AVERAGE` — Log average since last flush
- `MINIMUM` — Log minimum
- `MAXIMUM` — Log maximum

### Where to Inject Custom Metrics

**Option 1: Extend learner metrics**

The learner returns a metrics dict. Add custom debate metrics here:
```python
# In debate_grpo_learner.py
def __call__(self, batch, step):
    # ... existing GRPO logic ...
    metrics = {
        "loss": loss,
        "debate/reward/solver": solver_reward_mean,
        "debate/reward/verifier": verifier_reward_mean,
        "debate/kl/solver": solver_kl_mean,
        # ... etc
    }
    return LearnerOutput(loss=loss, metrics=metrics)
```

**Option 2: Inject via batch metadata**

Flink extracts metrics from `batch.metadata` via `metrics_processing.extract_metrics_from_metadata()`:
```python
# In debate actor
metadata = {
    "trajectory": ...,
    "debate/reward/solver": solver_rewards,  # Will be auto-averaged
    "debate/kl/solver": solver_kls,
}
```

**Option 3: Direct plotter access**

Access the plotter directly in actor/learner:
```python
self.plotter.accumulate_scalar("debate/reward/solver", value, mode=AccumulationMode.AVERAGE)
```

**Recommended:** Use **Option 1** (learner metrics) for global/aggregated metrics, and **Option 2** (metadata) for per-sample metrics that need aggregation.

---

## 4. Generation Logging

### Configuration

**Location:** `post_training/flink/flink_zord.py`

**Config field:**
- `log_train_generations_every_steps: int | None = None`

**Already enabled in our SWEEP configs:**
- `log_train_generations_every_steps=25`

### How It Works

1. `FlinkZord._maybe_log_train_generations()` checks if `step % log_train_generations_every_steps == 0`
2. Extracts generation info via `metrics_processing.extract_log_generations_input_from_batch()`
3. Prunes to `max_number_wandb_logged_samples` (if set)
4. Logs via `self.plotter.log_generations(step_number, name, **info)`

**W&B integration:**
- Logged as W&B Tables (for structured data)
- Supports columns: prompts, completions, rewards, etc.

---

## 5. Existing SWEEP Configuration

Our SWEEP configs already set:

```python
# In SWEEP config
flink_config = {
    "save_debug_data_every_steps": 1,
    "log_train_generations_every_steps": 25,
    "max_number_wandb_logged_samples": None,  # No limit (will set to 16-32 in Phase 5)
}
```

**Implications for Phase 5:**
- Debug data is already being saved every step
- Generations are already logged to W&B every 25 steps
- We need to **add debate-specific columns** to the debug data schema
- We need to **inject debate metrics** into the learner metrics dict

---

## 6. Recommendations for Phase 5

### For Plan 05-02 (Debate Metric Collection)

1. **Extend learner metrics:** Add per-role rewards, KL divergences, zero-advantage metrics to `DebateGRPOLearner` output
2. **Use debate/ namespace:** All custom metrics must use `debate/` prefix to avoid collision with Flink built-ins
3. **Cap W&B samples:** Set `max_number_wandb_logged_samples=16` to avoid 25 MB payload limit

### For Plan 05-03 (Debug Data Extension)

1. **Extend BatchDebugData:** Add `schema_version`, `role_assignments`, `solver_reward`, `verifier_reward`, `judge_reward`, `solver_kl`, `verifier_kl`, `judge_kl`
2. **Monkey-patch or subclass:** Either monkey-patch `extract_batch_debug_data_from_batch()` or create a debate-specific version
3. **Schema versioning:** Always save `schema_version=1` for forward compatibility

---

## 7. Open Questions

- **Q:** What is the exact W&B metric key for global gradient norm?
  - **A:** Likely `grad/global_norm` or `grad_global_norm` — verify during testing

- **Q:** Can we enable `advanced_logging` without modifying Flink source?
  - **A:** Yes, patch via `patch_run_config` in SWEEP config

- **Q:** How to test W&B Table INCREMENTAL mode?
  - **A:** W&B Server v0.70.0+ required (verify during Phase 5 execution)

---

## References

- Flink source: `repos/post_training/post_training/flink/flink_zord.py`
- Fax config: `repos/fax/fax/config/config_run.py`
- Metrics processing: `repos/post_training/post_training/relax/metrics_processing.py`
- Plotting: `repos/fax/fax/plotting/accumulating_plotter.py`
