#!/usr/bin/env python3
"""Generate Figure 5: Communication effects (3-panel) with paper comparison.

Reads: data/fig5_metrics.json
Writes: figures/fig5_communication.{pdf,png}

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
INPUT = PROJECT_ROOT / "data" / "fig5_metrics.json"
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={"wspace": 0.35})

    # =================================================================
    # Panel (a): Success Rates
    # =================================================================
    ax_a = axes[0]
    sr = metrics["success_rates"]
    comm_rate = sr["coop_comm"]["rate"] * 100
    nocomm_rate = sr["coop_nocomm"]["rate"] * 100
    comm_ci_upper = sr["coop_comm"]["ci_upper"] * 100
    nocomm_ci_upper = sr["coop_nocomm"]["ci_upper"] * 100
    comm_n = sr["coop_comm"]["total"]
    nocomm_n = sr["coop_nocomm"]["total"]

    x_pos = np.array([0, 1])
    bar_vals = [comm_rate, nocomm_rate]
    bar_colors = [COLORS["coop_comm"], COLORS["coop_nocomm"]]
    # Error bars: lower bound is 0 (rate is 0), upper bound is Wilson CI upper
    yerr_lower = [0, 0]
    yerr_upper = [comm_ci_upper, nocomm_ci_upper]

    bars_a = ax_a.bar(
        x_pos, bar_vals, width=0.5, color=bar_colors, edgecolor="white",
        yerr=[yerr_lower, yerr_upper], capsize=6, error_kw={"linewidth": 1.5}
    )

    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels([f"Comm\n(n={comm_n})", f"No-Comm\n(n={nocomm_n})"])
    ax_a.set_ylabel("Success Rate (%)")
    ax_a.set_ylim(0, 10)
    ax_a.set_title("(a) Success Rates")

    # Annotate center: "0% in both settings"
    ax_a.text(
        0.5, 0.5, "0% in both settings",
        transform=ax_a.transAxes, ha="center", va="center",
        fontsize=10, color="gray", fontstyle="italic",
    )

    # Annotate CI upper bounds
    ax_a.annotate(
        f"CI upper: {comm_ci_upper:.1f}%",
        xy=(0, comm_ci_upper), xytext=(0.3, comm_ci_upper + 1.5),
        fontsize=8, color="gray", ha="center",
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
    )
    ax_a.annotate(
        f"CI upper: {nocomm_ci_upper:.1f}%",
        xy=(1, nocomm_ci_upper), xytext=(1.0, nocomm_ci_upper + 1.5),
        fontsize=8, color="gray", ha="center",
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
    )

    # =================================================================
    # Panel (b): Merge Conflict Rates
    # =================================================================
    ax_b = axes[1]
    mcr = metrics["merge_conflict_rates"]
    comm_conflict = mcr["coop_comm"]["rate"] * 100
    nocomm_conflict = mcr["coop_nocomm"]["rate"] * 100
    comm_conflict_ci_lo = mcr["coop_comm"]["ci_lower"] * 100
    comm_conflict_ci_hi = mcr["coop_comm"]["ci_upper"] * 100
    nocomm_conflict_ci_lo = mcr["coop_nocomm"]["ci_lower"] * 100
    nocomm_conflict_ci_hi = mcr["coop_nocomm"]["ci_upper"] * 100

    bar_vals_b = [comm_conflict, nocomm_conflict]
    yerr_b_lower = [comm_conflict - comm_conflict_ci_lo, nocomm_conflict - nocomm_conflict_ci_lo]
    yerr_b_upper = [comm_conflict_ci_hi - comm_conflict, nocomm_conflict_ci_hi - nocomm_conflict]

    bars_b = ax_b.bar(
        x_pos, bar_vals_b, width=0.5, color=bar_colors, edgecolor="white",
        yerr=[yerr_b_lower, yerr_b_upper], capsize=6, error_kw={"linewidth": 1.5}
    )

    # Bold percentage labels on top of bars
    for i, val in enumerate(bar_vals_b):
        ax_b.text(
            x_pos[i], val + yerr_b_upper[i] + 1.5,
            f"{val:.0f}%", ha="center", va="bottom",
            fontsize=11, fontweight="bold",
        )

    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels([f"Comm\n(n=100)", f"No-Comm\n(n=100)"])
    ax_b.set_ylabel("Merge Conflict Rate (%)")
    ax_b.set_title("(b) Merge Conflict Rates")
    ax_b.set_ylim(0, 80)

    # Paper baseline reference lines
    paper_fig5 = PAPER_BASELINES["fig5"]["first_turn_planning"]
    ax_b.axhline(
        y=paper_fig5["conflict_with"], linestyle="--", color=COLORS["paper"],
        linewidth=1.2, alpha=0.8, label=f"Paper: with planning ({paper_fig5['conflict_with']}%)",
    )
    ax_b.axhline(
        y=paper_fig5["conflict_without"], linestyle=":", color=COLORS["paper"],
        linewidth=1.2, alpha=0.8, label=f"Paper: without planning ({paper_fig5['conflict_without']}%)",
    )
    ax_b.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # =================================================================
    # Panel (c): Speech Acts & Communication Overhead
    # =================================================================
    ax_c = axes[2]
    sa = metrics["speech_acts"]
    categories = ["plan", "question", "update", "other"]
    cat_labels = ["Plan", "Question", "Update", "Other"]
    pcts = [sa[c]["pct"] for c in categories]

    x_cats = np.arange(len(categories))
    bars_c = ax_c.bar(
        x_cats, pcts, width=0.6, color=COLORS["coop_comm"], edgecolor="white",
    )

    # Percentage labels on top of bars
    for i, pct in enumerate(pcts):
        ax_c.text(
            x_cats[i], pct + 1.0,
            f"{pct:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax_c.set_xticks(x_cats)
    ax_c.set_xticklabels(cat_labels)
    ax_c.set_ylabel("% of Messages")
    ax_c.set_ylim(0, 60)

    # Paper baseline reference line at ~1/3 each
    paper_speech_pct = PAPER_BASELINES["fig5"]["speech_acts"]["plan"]  # 33.3%
    ax_c.axhline(
        y=paper_speech_pct, linestyle="--", color=COLORS["paper"],
        linewidth=1.2, alpha=0.8, label=f"Paper: ~1/3 each ({paper_speech_pct:.1f}%)",
    )
    ax_c.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # Overhead annotation
    overhead = metrics["overhead"]
    mean_overhead = overhead["mean_pct"]
    ax_c.set_title(f"(c) Speech Acts (overhead: {mean_overhead}%)")

    # Secondary annotation: paper overhead
    paper_overhead = PAPER_BASELINES["fig5"]["comm_overhead_pct"]
    ax_c.text(
        0.98, 0.85,
        f"Paper overhead: ~{paper_overhead:.0f}%",
        transform=ax_c.transAxes, ha="right", va="top",
        fontsize=8, color=COLORS["paper"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    # --- Suptitle ---
    fig.suptitle(
        "Communication Effects: Comm vs No-Comm",
        fontsize=14, fontweight="bold", y=1.02,
    )

    plt.tight_layout()

    # --- Save as both PDF and PNG ---
    for ext in ["pdf", "png"]:
        path = FIG_DIR / f"fig5_communication.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {path}")

    plt.close(fig)
    print("Figure 5 generation complete.")


if __name__ == "__main__":
    main()
