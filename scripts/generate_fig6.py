#!/usr/bin/env python3
"""Generate Figure 6: Communication error taxonomy bar chart with paper comparison.

Reads: data/fig6_metrics.json
Writes: figures/fig6_error_taxonomy.{pdf,png}

Uses system Python (matplotlib 3.10.8, numpy 2.4.2).
Do NOT run with the project .venv (no matplotlib there).
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("agg")  # Headless backend -- must be set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Enable import of paper_baselines from scripts/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_baselines import PAPER_BASELINES

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT = PROJECT_ROOT / "data" / "fig6_metrics.json"
FIG_DIR = PROJECT_ROOT / "figures"

# --- Publication-quality settings ---
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "pdf.fonttype": 42,   # TrueType fonts in PDF (not Type 3)
    "ps.fonttype": 42,
})

# --- Color scheme (matching Phase 1 convention) ---
COLORS = {
    "solo": "#4878CF",
    "coop_comm": "#6ACC65",
    "coop_nocomm": "#D65F5F",
    "paper": "#888888",
}

# Colors for grouping brackets (one per paper category)
GROUP_COLORS = {
    "Repetition": "#E8A838",       # amber
    "Unresponsiveness": "#8B5CF6", # purple
    "Hallucination": "#EF4444",    # red
}


def main():
    # Load metrics
    with open(INPUT) as f:
        metrics = json.load(f)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    freq = metrics["frequency"]
    summary = metrics["summary"]

    # Category order: C4a, C4b, C1a, C1b, C2, C3b (grouped by paper categories)
    # Group by paper mapping: Repetition (C4a, C4b), Unresponsiveness (C1a, C1b), Hallucination (C2, C3b)
    cat_order = ["C4a", "C4b", "C1a", "C1b", "C2", "C3b"]
    cat_labels = [
        "C4a:\nSame Info",
        "C4b:\nNear-dup",
        "C1a:\nNo Reply",
        "C1b:\nIgnored",
        "C2:\nVague",
        "C3b:\nCorrected",
    ]

    counts = [freq[c]["count"] for c in cat_order]
    pcts = [freq[c]["pct_of_errors"] for c in cat_order]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(cat_order))
    bars = ax.bar(x, pcts, width=0.6, color=COLORS["coop_comm"], edgecolor="white")

    # Count + percentage annotations on top of each bar
    for i in range(len(cat_order)):
        ax.text(
            x[i], pcts[i] + 1.2,
            f"n={counts[i]}\n({pcts[i]:.1f}%)",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=9)
    ax.set_ylabel("% of Total Errors")
    ax.set_ylim(0, 58)  # Room for annotations above tallest bar (41.6%)

    # --- Title ---
    ax.set_title("Communication Error Taxonomy", fontsize=14, fontweight="bold")

    # --- Subtitle annotation ---
    total_errors = summary["total_errors"]
    transcripts_with = summary["transcripts_with_errors"]
    total_transcripts = summary["total_transcripts"]
    error_rate = transcripts_with / total_transcripts * 100
    ax.text(
        0.5, 0.95,
        f"{total_errors} errors across {transcripts_with}/{total_transcripts} transcripts ({error_rate:.0f}% error rate)",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=10, color="gray", fontstyle="italic",
    )

    # =================================================================
    # Paper category grouping brackets below x-axis
    # =================================================================
    # Groups: Repetition (indices 0-1), Unresponsiveness (indices 2-3), Hallucination (indices 4-5)
    category_mapping = PAPER_BASELINES["fig6"]["category_mapping"]
    groups = [
        ("Repetition", 0, 1),
        ("Unresponsiveness", 2, 3),
        ("Hallucination", 4, 5),
    ]

    # Compute aggregated percentages for each paper category from our data
    group_pcts = {}
    for group_name, idx_start, idx_end in groups:
        group_total = sum(pcts[idx_start:idx_end + 1])
        group_pcts[group_name] = group_total

    # Draw grouping brackets below x-axis labels
    bracket_y = -14.0  # Below the x-tick labels
    label_y = -17.5

    for group_name, idx_start, idx_end in groups:
        color = GROUP_COLORS[group_name]
        x_left = x[idx_start] - 0.35
        x_right = x[idx_end] + 0.35
        x_mid = (x[idx_start] + x[idx_end]) / 2

        # Draw bracket: horizontal line with small vertical ticks at ends
        bracket_line_y = bracket_y
        tick_height = 1.2

        # Horizontal line
        ax.plot(
            [x_left, x_right], [bracket_line_y, bracket_line_y],
            color=color, linewidth=2.0, clip_on=False,
            transform=ax.get_xaxis_transform() if False else ax.transData,
        )
        # Left tick
        ax.plot(
            [x_left, x_left], [bracket_line_y, bracket_line_y + tick_height],
            color=color, linewidth=2.0, clip_on=False,
        )
        # Right tick
        ax.plot(
            [x_right, x_right], [bracket_line_y, bracket_line_y + tick_height],
            color=color, linewidth=2.0, clip_on=False,
        )
        # Group label
        ax.text(
            x_mid, label_y,
            f"{group_name}\n({group_pcts[group_name]:.1f}%)",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color=color, clip_on=False,
        )

    # Extend y-axis bottom to make room for brackets
    ax.set_ylim(-24, 58)

    # Hide the negative y-axis region grid and ticks
    ax.set_yticks([0, 10, 20, 30, 40, 50])

    # Add a line at y=0 to separate brackets from chart
    ax.axhline(y=0, color="black", linewidth=0.8)

    # =================================================================
    # Annotation box: paper category mapping (bottom right)
    # =================================================================
    annotation_lines = [
        "Paper maps to 3 categories:",
        f"  Repetition (C4a+C4b): {group_pcts['Repetition']:.1f}%",
        f"  Unresponsiveness (C1a+C1b): {group_pcts['Unresponsiveness']:.1f}%",
        f"  Hallucination (C2+C3b): {group_pcts['Hallucination']:.1f}%",
        "(Paper exact % not reported)",
    ]
    annotation_text = "\n".join(annotation_lines)

    ax.text(
        0.98, 0.70,
        annotation_text,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8.5,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="#cccccc",
            alpha=0.9,
        ),
    )

    plt.tight_layout()

    # --- Save as both PDF and PNG ---
    for ext in ["pdf", "png"]:
        path = FIG_DIR / f"fig6_error_taxonomy.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {path}")

    plt.close(fig)
    print("Figure 6 generation complete.")


if __name__ == "__main__":
    main()
