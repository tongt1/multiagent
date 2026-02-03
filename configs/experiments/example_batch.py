"""Example experiment batch configuration.

Copy this file and modify for your experiments.

Usage:
    # Preview experiments (dry run - default)
    python scripts/run_experiment_batch.py --config configs/experiments/example_batch.py

    # Submit experiments
    python scripts/run_experiment_batch.py --config configs/experiments/example_batch.py --submit

    # Resume failed experiments
    python scripts/run_experiment_batch.py --config configs/experiments/example_batch.py --submit --resume
"""

from configs.experiments.experiment_config import ExperimentConfig, ExperimentBatchConfig

# Define individual experiments
# Only specify parameters that DIFFER from Phase 2 base SWEEP config defaults.
# Defaults: model=command-a-03-2025, lr=1.5e-6, batch_size=64, steps=500, kl_beta=0.01

debate_cmdA = ExperimentConfig(
    name="debate_cmdA",
    mode="debate",
    model="command-a-03-2025",
    linked_experiment="baseline_cmdA",  # Link for comparison
)

baseline_cmdA = ExperimentConfig(
    name="baseline_cmdA",
    mode="baseline",
    model="command-a-03-2025",
    linked_experiment="debate_cmdA",
)

# Example: Open-source model comparison
debate_llama8b = ExperimentConfig(
    name="debate_llama8b",
    mode="debate",
    model="meta-llama/Llama-3-8b",
    total_train_steps=300,  # Fewer steps for smaller model
    linked_experiment="baseline_llama8b",
)

baseline_llama8b = ExperimentConfig(
    name="baseline_llama8b",
    mode="baseline",
    model="meta-llama/Llama-3-8b",
    total_train_steps=300,
    linked_experiment="debate_llama8b",
)

# Batch configuration
BATCH_CONFIG = ExperimentBatchConfig(
    experiments=[debate_cmdA, baseline_cmdA, debate_llama8b, baseline_llama8b],
    batch_name="debate_vs_baseline_initial",
    base_ckpt_path="s3://us-east-01a/foundations-experiments/your/checkpoint/path/ckpt-0/",
    gcs_bucket="your-bucket",
    max_retries=3,
    parallel=True,
)
