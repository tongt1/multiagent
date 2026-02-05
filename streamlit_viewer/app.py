"""Streamlit app entry point for GRPO training rollout viewer.

This is the main entry point for the Streamlit viewer application. It provides
sidebar navigation for data source selection and tabs for different viewing modes.

Run with: streamlit run streamlit_viewer/app.py
"""

import streamlit as st

from streamlit_viewer.config import (
    DATA_SOURCE_PARQUET,
    DATA_SOURCE_WANDB,
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from streamlit_viewer.lib.data_loader import auto_detect_source, discover_parquet_files
from streamlit_viewer.pages.step_browser import render_step_browser


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="GRPO Training Rollout Viewer",
    page_icon="🔍",
    layout="wide",
)


# ============================================================================
# Session State Initialization
# ============================================================================

if "selected_step" not in st.session_state:
    st.session_state.selected_step = None

if "selected_prompt" not in st.session_state:
    st.session_state.selected_prompt = None

if "comparison_rollout_ids" not in st.session_state:
    st.session_state.comparison_rollout_ids = []

if "run_configs" not in st.session_state:
    # Default run configuration
    st.session_state.run_configs = [
        {
            "label": "Run 1",
            "source_type": DATA_SOURCE_PARQUET,
            "path_or_run": ".",
        }
    ]


# ============================================================================
# Sidebar: Data Source Selection
# ============================================================================

st.sidebar.title("GRPO Rollout Viewer")
st.sidebar.markdown("---")

st.sidebar.subheader("Data Source")

# Data source mode selection
source_mode = st.sidebar.radio(
    "Select mode:",
    ["Auto-detect", "Local Parquet", "W&B Tables"],
    help="Choose how to load training rollout data",
)

# Source-specific inputs
parquet_dir = None
wandb_entity = None
wandb_project = None
wandb_run_id = None

if source_mode in ["Local Parquet", "Auto-detect"]:
    parquet_dir = st.sidebar.text_input(
        "Parquet directory:",
        value=".",
        help="Directory containing batch_debug_data_train_*.parquet files",
    )

if source_mode in ["W&B Tables", "Auto-detect"]:
    wandb_entity = st.sidebar.text_input(
        "W&B Entity:",
        value=WANDB_ENTITY,
        help="W&B entity name (usually 'cohere')",
    )
    wandb_project = st.sidebar.text_input(
        "W&B Project:",
        value=WANDB_PROJECT,
        help="W&B project name",
    )
    if source_mode == "W&B Tables":
        wandb_run_id = st.sidebar.text_input(
            "Run ID (optional):",
            value="",
            help="Specific W&B run ID to load",
        )

# Auto-detect and display detected source
if source_mode == "Auto-detect":
    detected_source, metadata = auto_detect_source(
        parquet_dir=parquet_dir,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
    )
    if detected_source == DATA_SOURCE_PARQUET:
        st.sidebar.info(
            f"✓ Detected Parquet files: {metadata.get('num_files', 0)} files, "
            f"steps {min(metadata.get('steps', [0]))} to {max(metadata.get('steps', [0]))}"
        )
    elif detected_source == DATA_SOURCE_WANDB:
        st.sidebar.info(
            f"✓ Detected W&B: {metadata.get('num_runs', 0)} runs in {metadata['project']}"
        )
    else:
        st.sidebar.warning("⚠ No data source found. Check paths and credentials.")


# ============================================================================
# Sidebar: Multi-Run Support (for comparison)
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.subheader("Runs")

# Allow adding up to 4 runs for comparison
num_runs = len(st.session_state.run_configs)

for i, run_config in enumerate(st.session_state.run_configs):
    with st.sidebar.expander(f"Run {i+1}: {run_config['label']}", expanded=(i == 0)):
        run_config["label"] = st.text_input(
            f"Label {i+1}:",
            value=run_config["label"],
            key=f"run_label_{i}",
        )
        run_config["source_type"] = st.radio(
            f"Source type {i+1}:",
            [DATA_SOURCE_PARQUET, DATA_SOURCE_WANDB],
            index=0 if run_config["source_type"] == DATA_SOURCE_PARQUET else 1,
            key=f"run_source_type_{i}",
        )
        run_config["path_or_run"] = st.text_input(
            f"Path/Run ID {i+1}:",
            value=run_config["path_or_run"],
            key=f"run_path_{i}",
        )

if num_runs < 4:
    if st.sidebar.button("➕ Add run", key="add_run_button"):
        st.session_state.run_configs.append({
            "label": f"Run {num_runs + 1}",
            "source_type": DATA_SOURCE_PARQUET,
            "path_or_run": ".",
        })
        st.rerun()


# ============================================================================
# Sidebar: Footer
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Multiagent Debate RL**
    Version: 0.1.0
    [GitHub](https://github.com/cohere-ai/multiagent-debate-rl)
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Main Content: Tab Navigation
# ============================================================================

st.title("GRPO Training Rollout Viewer")
st.markdown(
    "Interactive visualization of multi-agent debate GRPO training rollouts. "
    "Explore training progress, compare rollouts, and analyze per-role contributions."
)
st.markdown("---")

# Create tabs for different views
tabs = st.tabs(["📊 Overview", "🔬 Step Browser", "⚖️ Comparison"])

# Tab 1: Overview (placeholder for Plan 07-03)
with tabs[0]:
    st.info("📊 **Overview tab coming in Plan 07-03**")
    st.markdown("This tab will show:")
    st.markdown("- Training progress timeline")
    st.markdown("- Per-role reward trends")
    st.markdown("- Zero-advantage detection metrics")

    # Display current data source config for verification
    st.markdown("---")
    st.subheader("Current Data Source Configuration")
    if source_mode == "Auto-detect":
        st.json({"mode": "Auto-detect", "parquet_dir": parquet_dir, "wandb_entity": wandb_entity})
    elif source_mode == "Local Parquet":
        st.json({"mode": "Local Parquet", "directory": parquet_dir})
    else:
        st.json({"mode": "W&B Tables", "entity": wandb_entity, "project": wandb_project, "run_id": wandb_run_id or "all"})

# Tab 2: Step Browser
with tabs[1]:
    render_step_browser(st.session_state.run_configs)

# Tab 3: Comparison (placeholder for Plan 07-03)
with tabs[2]:
    st.info("⚖️ **Comparison tab coming in Plan 07-03**")
    st.markdown("This tab will show:")
    st.markdown("- Side-by-side debate timeline visualization")
    st.markdown("- Per-turn reward breakdown")
    st.markdown("- Text diff for solver/verifier/judge turns")
