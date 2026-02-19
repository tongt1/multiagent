#!/usr/bin/env python3
"""Generate Phase 1 execution figures for Experiment 12: CooperBench Reproduction."""

import json
import os
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

LOGS_DIR = Path("/mnt/data/terry/home/cooperbench-repro/repos/CooperBench/logs")
FIG_DIR = Path(os.path.expanduser("~/multiagent/experiments/figures/12_cooperbench_repro"))
FIG_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS = {
    "solo": {"dir": "command-a-solo/solo", "color": "#4878CF", "label": "Solo"},
    "coop-comm": {"dir": "command-a-coop-comm/coop", "color": "#6ACC65", "label": "Coop-Comm"},
    "coop-nocomm": {"dir": "command-a-coop-nocomm/coop", "color": "#D65F5F", "label": "Coop-NoComm"},
}


def load_results():
    """Load all result.json files, return dict[setting] -> list[dict]."""
    results = {}
    for setting, info in SETTINGS.items():
        setting_dir = LOGS_DIR / info["dir"]
        records = []
        for result_file in setting_dir.rglob("result.json"):
            with open(result_file) as f:
                records.append(json.load(f))
        results[setting] = records
        print(f"  {setting}: {len(records)} results loaded")
    return results


def load_conversations():
    """Load conversation.json files for coop settings."""
    convos = {}
    for setting in ["coop-comm", "coop-nocomm"]:
        info = SETTINGS[setting]
        setting_dir = LOGS_DIR / info["dir"]
        records = []
        for conv_file in setting_dir.rglob("conversation.json"):
            with open(conv_file) as f:
                records.append(json.load(f))
        convos[setting] = records
    return convos


def fig1_submission_and_cost(results):
    """Bar chart: submission rates and costs by setting."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    settings = list(SETTINGS.keys())
    colors = [SETTINGS[s]["color"] for s in settings]
    labels = [SETTINGS[s]["label"] for s in settings]

    # Submission rates
    rates = []
    for s in settings:
        recs = results[s]
        if s == "solo":
            submitted = sum(1 for r in recs if r["agent"]["status"] == "Submitted")
            total = len(recs)
        else:
            submitted = sum(
                (1 if a["status"] == "Submitted" else 0)
                for r in recs
                for a in r["agents"].values()
            )
            total = sum(len(r["agents"]) for r in recs)
        rates.append(submitted / total * 100)

    bars1 = ax1.bar(labels, rates, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("Submission Rate (%)")
    ax1.set_title("Agent Submission Rate by Setting")
    ax1.set_ylim(95, 101)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    for bar, rate in zip(bars1, rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{rate:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Total cost
    costs = [sum(r["total_cost"] for r in results[s]) for s in settings]
    bars2 = ax2.bar(labels, costs, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Total API Cost ($)")
    ax2.set_title("Total API Cost by Setting")
    for bar, cost in zip(bars2, costs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"${cost:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    fig.suptitle("Phase 1 Execution Summary — Command A on CooperBench Lite", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "submission_and_cost.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  -> submission_and_cost.png")


def fig2_cost_distribution(results):
    """Histogram: per-pair cost distribution by setting."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, (setting, info) in zip(axes, SETTINGS.items()):
        costs = [r["total_cost"] for r in results[setting]]
        ax.hist(costs, bins=30, color=info["color"], alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Cost per Pair ($)")
        ax.set_title(f"{info['label']} (n={len(costs)})")
        ax.axvline(np.median(costs), color="black", linestyle="--", linewidth=1, label=f"median=${np.median(costs):.2f}")
        ax.axvline(np.mean(costs), color="black", linestyle=":", linewidth=1, label=f"mean=${np.mean(costs):.2f}")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Count")
    fig.suptitle("Per-Pair Cost Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "cost_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  -> cost_distribution.png")


def fig3_steps_distribution(results):
    """Box plot: steps per agent by setting."""
    fig, ax = plt.subplots(figsize=(8, 5))

    data = []
    labels = []
    colors = []
    for setting, info in SETTINGS.items():
        if setting == "solo":
            steps = [r["agent"]["steps"] for r in results[setting]]
        else:
            steps = [a["steps"] for r in results[setting] for a in r["agents"].values()]
        data.append(steps)
        labels.append(f"{info['label']}\n(n={len(steps)})")
        colors.append(info["color"])

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Steps per Agent")
    ax.set_title("Agent Step Count Distribution by Setting", fontsize=13, fontweight="bold")

    # Add mean markers
    for i, d in enumerate(data):
        ax.plot(i + 1, np.mean(d), "D", color="black", markersize=6, zorder=5)
        ax.annotate(f"mean={np.mean(d):.1f}", (i + 1, np.mean(d)),
                    textcoords="offset points", xytext=(35, 0), fontsize=9)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "steps_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  -> steps_distribution.png")


def fig4_patch_production(results):
    """Grouped bar: patch production rate and avg patch lines."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    settings = list(SETTINGS.keys())
    colors = [SETTINGS[s]["color"] for s in settings]
    labels = [SETTINGS[s]["label"] for s in settings]

    # Patch production rate (% of pairs that produced at least one non-empty patch)
    patch_rates = []
    for s in settings:
        recs = results[s]
        if s == "solo":
            has_patch = sum(1 for r in recs if r["agent"].get("patch_lines", 0) > 0)
        else:
            has_patch = sum(1 for r in recs
                           if any(a.get("patch_lines", 0) > 0 for a in r["agents"].values()))
        patch_rates.append(has_patch / len(recs) * 100)

    bars1 = ax1.bar(labels, patch_rates, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("Pairs with Patches (%)")
    ax1.set_title("Patch Production Rate")
    ax1.set_ylim(70, 105)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    for bar, rate in zip(bars1, patch_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{rate:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Avg patch lines (for pairs that produced patches)
    avg_lines = []
    for s in settings:
        recs = results[s]
        if s == "solo":
            lines = [r["agent"]["patch_lines"] for r in recs if r["agent"].get("patch_lines", 0) > 0]
        else:
            lines = [sum(a.get("patch_lines", 0) for a in r["agents"].values())
                     for r in recs
                     if any(a.get("patch_lines", 0) > 0 for a in r["agents"].values())]
        avg_lines.append(np.mean(lines) if lines else 0)

    bars2 = ax2.bar(labels, avg_lines, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Avg Patch Lines (per pair)")
    ax2.set_title("Average Patch Size (non-empty only)")
    for bar, val in zip(bars2, avg_lines):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    fig.suptitle("Patch Production — Command A on CooperBench Lite", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "patch_production.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  -> patch_production.png")


def fig5_per_repo_breakdown(results):
    """Heatmap-style: cost and steps by repository."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    # Collect per-repo stats
    repo_stats = defaultdict(lambda: defaultdict(list))
    for setting, recs in results.items():
        for r in recs:
            repo = r["repo"]
            repo_stats[repo][f"{setting}_cost"].append(r["total_cost"])
            if setting == "solo":
                repo_stats[repo][f"{setting}_steps"].append(r["agent"]["steps"])
            else:
                repo_stats[repo][f"{setting}_steps"].extend(
                    a["steps"] for a in r["agents"].values()
                )

    repos = sorted(repo_stats.keys())
    y = np.arange(len(repos))
    height = 0.25

    # Avg cost per pair by repo
    for i, (setting, info) in enumerate(SETTINGS.items()):
        vals = [np.mean(repo_stats[repo][f"{setting}_cost"]) if repo_stats[repo][f"{setting}_cost"] else 0
                for repo in repos]
        ax1.barh(y + i * height, vals, height, color=info["color"], label=info["label"])

    ax1.set_yticks(y + height)
    ax1.set_yticklabels([r.replace("_task", "") for r in repos], fontsize=8)
    ax1.set_xlabel("Avg Cost per Pair ($)")
    ax1.set_title("Cost by Repository")
    ax1.legend(fontsize=8)

    # Avg steps per agent by repo
    for i, (setting, info) in enumerate(SETTINGS.items()):
        vals = [np.mean(repo_stats[repo][f"{setting}_steps"]) if repo_stats[repo][f"{setting}_steps"] else 0
                for repo in repos]
        ax2.barh(y + i * height, vals, height, color=info["color"], label=info["label"])

    ax2.set_yticks(y + height)
    ax2.set_yticklabels([r.replace("_task", "") for r in repos], fontsize=8)
    ax2.set_xlabel("Avg Steps per Agent")
    ax2.set_title("Steps by Repository")
    ax2.legend(fontsize=8)

    fig.suptitle("Per-Repository Breakdown", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "per_repo_breakdown.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  -> per_repo_breakdown.png")


def fig6_communication_stats(results, convos):
    """Communication: message count distribution and overhead."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Message count distribution (coop-comm only)
    msg_counts = [r.get("messages_sent", 0) for r in results["coop-comm"]]
    ax1.hist(msg_counts, bins=range(0, max(msg_counts) + 2), color=SETTINGS["coop-comm"]["color"],
             alpha=0.85, edgecolor="white", linewidth=0.5, align="left")
    ax1.set_xlabel("Messages per Pair")
    ax1.set_ylabel("Count")
    ax1.set_title("Message Count Distribution (Coop-Comm)")
    ax1.axvline(np.mean(msg_counts), color="black", linestyle="--", linewidth=1,
                label=f"mean={np.mean(msg_counts):.1f}")
    ax1.legend()

    # Communication overhead: messages as % of total steps
    overheads = []
    for r in results["coop-comm"]:
        msgs = r.get("messages_sent", 0)
        total_steps = r["total_steps"]
        if total_steps > 0:
            overheads.append(msgs / total_steps * 100)

    ax2.hist(overheads, bins=20, color=SETTINGS["coop-comm"]["color"],
             alpha=0.85, edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Messages as % of Total Steps")
    ax2.set_ylabel("Count")
    ax2.set_title("Communication Overhead")
    ax2.axvline(np.mean(overheads), color="black", linestyle="--", linewidth=1,
                label=f"mean={np.mean(overheads):.1f}%")
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.legend()

    fig.suptitle("Communication Statistics — Coop-Comm Setting", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "communication_stats.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  -> communication_stats.png")


def fig7_setting_comparison_summary(results):
    """Summary radar-style comparison across settings."""
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ["Submission\nRate (%)", "Patch\nRate (%)", "Avg Cost\n($/pair)", "Avg Steps\n(per agent)", "Avg Patch\nLines"]
    x = np.arange(len(metrics))
    width = 0.22

    for i, (setting, info) in enumerate(SETTINGS.items()):
        recs = results[setting]
        n = len(recs)

        # submission rate
        if setting == "solo":
            sub = sum(1 for r in recs if r["agent"]["status"] == "Submitted") / n * 100
        else:
            agents_total = sum(len(r["agents"]) for r in recs)
            sub = sum(1 for r in recs for a in r["agents"].values() if a["status"] == "Submitted") / agents_total * 100

        # patch rate
        if setting == "solo":
            patch = sum(1 for r in recs if r["agent"].get("patch_lines", 0) > 0) / n * 100
        else:
            patch = sum(1 for r in recs if any(a.get("patch_lines", 0) > 0 for a in r["agents"].values())) / n * 100

        # avg cost
        avg_cost = np.mean([r["total_cost"] for r in recs])

        # avg steps
        if setting == "solo":
            avg_steps = np.mean([r["agent"]["steps"] for r in recs])
        else:
            avg_steps = np.mean([a["steps"] for r in recs for a in r["agents"].values()])

        # avg patch lines (including zeros)
        if setting == "solo":
            avg_patch = np.mean([r["agent"].get("patch_lines", 0) for r in recs])
        else:
            avg_patch = np.mean([sum(a.get("patch_lines", 0) for a in r["agents"].values()) for r in recs])

        vals = [sub, patch, avg_cost, avg_steps, avg_patch]

        bars = ax.bar(x + i * width, vals, width, color=info["color"], label=info["label"],
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            fmt = f"{val:.1f}%" if "%" in metrics[x[list(bars).index(bar)]] else (
                f"${val:.2f}" if "Cost" in metrics[x[list(bars).index(bar)]] else f"{val:.0f}")
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt, ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_title("Setting Comparison — All Metrics", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "setting_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  -> setting_comparison.png")


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 10, "figure.facecolor": "white"})

    print("Loading results...")
    results = load_results()
    convos = load_conversations()

    print("\nGenerating figures...")
    fig1_submission_and_cost(results)
    fig2_cost_distribution(results)
    fig3_steps_distribution(results)
    fig4_patch_production(results)
    fig5_per_repo_breakdown(results)
    fig6_communication_stats(results, convos)
    fig7_setting_comparison_summary(results)

    print(f"\nAll figures saved to {FIG_DIR}")
