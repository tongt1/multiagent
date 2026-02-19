#!/usr/bin/env python3
"""Generate Figure 4: Difficulty-stratified success curves with paper comparison.

Reads: data/fig4_metrics.json
Writes: figures/fig4_difficulty_curves.{pdf,png}

Uses system Python (matplotlib 3.10.8, numpy 2.4.2).
Do NOT run with the project .venv (no matplotlib there).
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("agg")  # Headless backend -- must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np

# Enable import of paper_baselines from scripts/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_baselines import PAPER_BASELINES

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT = PROJECT_ROOT / "data" / "fig4_metrics.json"
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


def main():
    # Load metrics
    with open(INPUT) as f:
        metrics = json.load(f)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Extract bucket data
    buckets = metrics["buckets"]
    x = np.array([b["center"] for b in buckets])

    # All 10 bucket centers for x-axis context
    all_centers = [round(0.05 + 0.1 * i, 2) for i in range(10)]
    populated_centers = set(b["center"] for b in buckets)

    fig, ax = plt.subplots(figsize=(8, 5))

    # --- Shade unpopulated bucket regions in light gray ---
    for c in all_centers:
        if c not in populated_centers:
            ax.axvspan(c - 0.05, c + 0.05, color="#f0f0f0", alpha=0.3, zorder=0)

    # --- Plot each setting: CI bands + scatter+line ---
    settings_config = [
        ("solo", COLORS["solo"], "Solo (Command A)"),
        ("coop_comm", COLORS["coop_comm"], "Coop-Comm (Command A)"),
        ("coop_nocomm", COLORS["coop_nocomm"], "Coop-NoComm (Command A)"),
    ]

    for setting_key, color, label in settings_config:
        rates = np.array([b[setting_key]["rate"] for b in buckets])
        ci_lo = np.array([b[setting_key]["ci_lower"] for b in buckets])
        ci_hi = np.array([b[setting_key]["ci_upper"] for b in buckets])

        # CI shaded bands at each populated bucket (narrow x-width bands)
        for i in range(len(x)):
            ax.fill_between(
                [x[i] - 0.03, x[i] + 0.03],
                [ci_lo[i], ci_lo[i]],
                [ci_hi[i], ci_hi[i]],
                color=color,
                alpha=0.15,
            )

        # Rate values as markers connected by lines
        ax.plot(
            x, rates, "-o",
            color=color,
            label=label,
            markersize=7,
            linewidth=1.5,
            zorder=3,
        )

    # --- Axes configuration ---
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(all_centers)
    ax.set_xticklabels([f"{c:.2f}" for c in all_centers], fontsize=8, rotation=45)
    ax.set_xlabel("Task Difficulty d(t)")
    ax.set_ylabel("Success Rate")

    # --- Title with subtitle ---
    ax.set_title(
        "Difficulty-Stratified Success Curves\n"
        "(Command A, 3 of 10 difficulty buckets populated)",
        fontsize=13,
    )

    # --- Legend in upper right ---
    ax.legend(loc="upper right", framealpha=0.9)

    # --- AUC + retention annotation with paper comparison ---
    auc = metrics["auc"]
    retention = metrics["retention"]
    pooled = PAPER_BASELINES["fig4"]["pooled"]

    annotation_text = (
        f"Command A: Solo AUC={auc['solo']['value']:.3f}, "
        f"Coop AUC={auc['coop_comm']['value']:.3f}\n"
        f"Retention: Comm={retention['coop_comm']:.2f}, "
        f"NoComm={retention['coop_nocomm']:.2f}\n"
        f"\n"
        f"Paper (5 models, 10 buckets):\n"
        f"Solo AUC={pooled['solo_auc']:.3f}, "
        f"Coop AUC={pooled['coop_auc']:.3f}, "
        f"Retention={pooled['retention']:.2f}"
    )

    ax.text(
        0.02, 0.98, annotation_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="#cccccc",
            alpha=0.9,
        ),
    )

    # --- Save as both PDF and PNG ---
    for ext in ["pdf", "png"]:
        path = FIG_DIR / f"fig4_difficulty_curves.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {path}")

    plt.close(fig)
    print("Figure 4 generation complete.")


if __name__ == "__main__":
    main()
