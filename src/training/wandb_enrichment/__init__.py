"""W&B logging enrichment for multi-agent debate training.

This package provides:
- Canonical debate metric names with debate/ namespace prefix (metric_schema.py)
- Debate metric computation functions (debate_metrics.py)
- W&B rollout table creation and sampling (rollout_table.py)
- Parquet debug data writer for Phase 7 analysis (debug_data_writer.py)
- W&B workspace template for consistent dashboard layout (workspace_template.py)
- DebateMetricStreamer for training pipeline integration (debate_streamer.py)
- Integration layer bridging Flink to rollout tables (rollout_integration.py)
- Workspace initialization helper (workspace_init.py)

Integration architecture:
    DebateMetricStreamer.get()
    -> compute debate scalar metrics (per-role rewards, zero-advantage)
    -> log_debate_rollout_table() (W&B Tables with sampled rollouts)
    -> write_debate_debug_data() (Parquet files for Streamlit)
    -> all metrics flow through Flink batching -> W&B plotter
"""
