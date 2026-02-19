# Phase 4: Figure Generation and Paper Comparison - Research

**Researched:** 2026-02-18
**Domain:** Publication-quality matplotlib figure generation with paper baseline overlay
**Confidence:** HIGH

## Summary

Phase 4 generates three publication-quality figures (4, 5, 6) from the structured JSON metrics produced by Phase 3, and overlays the CooperBench paper's published baseline numbers for direct comparison. The core challenge is not the plotting mechanics -- matplotlib 3.10.8 with agg backend is available on the system Python and fully supports PDF/PNG export at 300+ DPI -- but rather (1) designing effective visualizations for severely sparse data (3 of 10 buckets populated, 0% coop pass rates), (2) extracting and correctly positioning the paper's multi-model baseline numbers onto single-model plots, and (3) ensuring the figures are genuinely publication-quality with proper typography, consistent color schemes, and clear annotation.

The paper uses 5 models (GPT-5, Claude Sonnet 4.5, MiniMax-M2, Qwen3-Coder, Qwen3) and reports pooled AUC retention of 0.59. Our Command A data has AUC retention of 0.0 (zero coop successes), so the comparison will focus on: (a) showing our data clearly, (b) overlaying the paper's pooled numbers as reference lines/markers, and (c) providing contextual annotation explaining the single-model vs multi-model difference.

The Phase 1 figure generation script (`scripts/generate_phase1_figures.py`) establishes the project's matplotlib conventions: `seaborn-v0_8-whitegrid` style, `dpi=200`, color scheme `{solo: #4878CF, coop-comm: #6ACC65, coop-nocomm: #D65F5F}`. Phase 4 should adopt the same palette for consistency but increase DPI to 300+ per requirements.

**Primary recommendation:** Use system Python (`/usr/bin/python3`) with matplotlib 3.10.8 and numpy 2.4.2. Write one figure-generation script per figure under `scripts/`, reading from `data/fig{4,5,6}_metrics.json` and writing to `figures/`. Encode paper baseline data as constants in a shared `PAPER_BASELINES` dictionary. Use `seaborn-v0_8-whitegrid` style with custom rcParams for publication quality.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FIG4-07 | Generate Figure 4 with CI shaded bands, publication-quality PDF/PNG at 300+ DPI | matplotlib `fill_between()` for CI bands, `savefig(dpi=300)` for both PDF and PNG. Only 3 populated buckets (3, 6, 9) -- use scatter+line with CI bands rather than smooth curves. Annotate AUC and retention values. |
| FIG5-05 | Generate Figure 5 as 3-panel plot: (a) success rates, (b) conflict rates, (c) overhead breakdown | `plt.subplots(1, 3)` layout. Panel (a): grouped bars for comm/nocomm success rates (both 0% -- annotate). Panel (b): grouped bars for conflict rates (41% vs 55%). Panel (c): stacked/grouped bars for speech acts + overhead mean annotation. |
| FIG6-03 | Generate Figure 6 as error frequency bar chart | Horizontal or vertical bar chart with 6 categories (C1a, C1b, C2, C3b, C4a, C4b). Use `pct_of_errors` as primary y-axis. Add count annotations on bars. |
| COMP-01 | Overlay paper's published baseline numbers on Figures 4, 5, 6 for direct visual comparison | Paper's pooled AUC=0.338 (solo), 0.200 (coop), retention=0.59. Communication overhead ~20%, speech acts ~1/3 each. Paper error taxonomy uses different categories (Repetition/Unresponsiveness/Hallucination vs our C1a-C4b) -- requires mapping. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| matplotlib | 3.10.8 | All figure generation, PDF/PNG export | System Python `/usr/bin/python3`. Proven in `scripts/generate_phase1_figures.py`. Supports agg backend for headless rendering, PDF/PNG at any DPI. |
| numpy | 2.4.2 | Array operations for plotting data | System Python. Required by matplotlib for numerical arrays. |
| json (stdlib) | 3.12 | Read metrics JSON files | Standard data loading, matching Phase 3 pattern |
| pathlib (stdlib) | 3.12 | File path handling | Standard across all project scripts |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| matplotlib.ticker | 3.10.8 | Custom axis formatting (percentages, etc.) | Figure 5 panels, axis labels |
| matplotlib.patches | 3.10.8 | Legend proxy patches | CI band legend entries |
| PIL/Pillow | available | PNG export backend for matplotlib | Automatic, used by savefig for PNG format |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| matplotlib | plotly | Interactive but produces HTML, not PDF/PNG at controlled DPI. matplotlib is already proven in this project. |
| matplotlib | seaborn | Seaborn adds statistical plot abstractions but is NOT installed. The `seaborn-v0_8-whitegrid` style is available in matplotlib without seaborn. |
| System Python | Project venv | Project `.venv` does NOT have matplotlib. System Python has matplotlib 3.10.8 + numpy 2.4.2. Use `#!/usr/bin/env python3` or explicit `/usr/bin/python3`. |

**Installation:**
```bash
# No installation needed. System Python already has matplotlib 3.10.8 + numpy 2.4.2.
# Verify:
python3 -c "import matplotlib; print(matplotlib.__version__)"  # 3.10.8
python3 -c "import numpy; print(numpy.__version__)"            # 2.4.2
```

**Important:** The project `.venv` (used by Phase 3 analysis scripts) does NOT have matplotlib. Figure generation scripts MUST use system Python (`/usr/bin/python3` or `python3`), not the project venv.

## Architecture Patterns

### Recommended Project Structure
```
scripts/
    generate_fig4.py          # FIG4-07 + COMP-01 (Figure 4)
    generate_fig5.py          # FIG5-05 + COMP-01 (Figure 5)
    generate_fig6.py          # FIG6-03 + COMP-01 (Figure 6)
    paper_baselines.py        # Shared paper baseline constants
data/
    fig4_metrics.json         # Input (from Phase 3)
    fig5_metrics.json         # Input (from Phase 3)
    fig6_metrics.json         # Input (from Phase 3)
figures/
    fig4_difficulty_curves.pdf   # Output
    fig4_difficulty_curves.png   # Output
    fig5_communication.pdf       # Output
    fig5_communication.png       # Output
    fig6_error_taxonomy.pdf      # Output
    fig6_error_taxonomy.png      # Output
```

### Pattern 1: Figure Script Structure
**What:** Each figure script reads metrics JSON, generates the figure, saves as both PDF and PNG.
**When to use:** All three figure scripts follow this pattern.
**Example:**
```python
#!/usr/bin/env python3
"""Generate Figure 4: Difficulty-stratified success curves with paper comparison.

Reads: data/fig4_metrics.json
Writes: figures/fig4_difficulty_curves.{pdf,png}
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_baselines import PAPER_BASELINES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT = PROJECT_ROOT / "data" / "fig4_metrics.json"
FIG_DIR = PROJECT_ROOT / "figures"

# Publication-quality settings
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
    "pdf.fonttype": 42,      # TrueType fonts in PDF (not Type 3)
    "ps.fonttype": 42,
})

# Color scheme matching Phase 1 convention
COLORS = {
    "solo": "#4878CF",
    "coop_comm": "#6ACC65",
    "coop_nocomm": "#D65F5F",
    "paper": "#888888",       # Gray for paper baselines
}


def main():
    # Load metrics
    with open(INPUT) as f:
        metrics = json.load(f)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ... generate figure ...

    for ext in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"fig4_difficulty_curves.{ext}", dpi=300)
    plt.close()
    print(f"Saved to {FIG_DIR}/fig4_difficulty_curves.{{pdf,png}}")


if __name__ == "__main__":
    main()
```

### Pattern 2: Paper Baselines Module
**What:** Centralized constants for paper's published numbers, shared across all figure scripts.
**When to use:** COMP-01 requirement for all three figures.
**Example:**
```python
"""Paper baseline numbers from CooperBench (arXiv:2601.13295).

All values extracted from the paper's tables and text.
Used for COMP-01: overlay comparison on our figures.
"""

PAPER_BASELINES = {
    # Table 5: AUC and Retention (per-model)
    "fig4": {
        "models": {
            "GPT-5":       {"solo_auc": 0.506, "coop_auc": 0.325, "retention": 0.64},
            "Claude 4.5":  {"solo_auc": 0.469, "coop_auc": 0.283, "retention": 0.60},
            "MiniMax-M2":  {"solo_auc": 0.374, "coop_auc": 0.171, "retention": 0.46},
            "Qwen-Coder":  {"solo_auc": 0.236, "coop_auc": 0.148, "retention": 0.63},
            "Qwen":        {"solo_auc": 0.106, "coop_auc": 0.072, "retention": 0.68},
        },
        "pooled": {"solo_auc": 0.338, "coop_auc": 0.200, "delta_auc": 0.138, "retention": 0.59},
    },

    # Figure 4 (paper's Figure 4): Communication effects
    "fig5": {
        "comm_overhead_pct": 20.0,  # "as much as 20% of the steps"
        "speech_acts": {
            "plan": 33.3,    # "each almost takes up 1/3"
            "question": 33.3,
            "update": 33.3,
        },
        # First-turn planning effect
        "first_turn_planning": {
            "conflict_with": 29.4,    # conflict rate with first-turn planning
            "conflict_without": 51.5,  # conflict rate without first-turn planning
        },
        # Note: per-model comm/nocomm success rates and conflict rates
        # are shown visually in paper's Figure 4 but not reported numerically.
    },

    # Figure 5 (paper's Figure 5): Communication error detection
    "fig6": {
        # Paper uses 3 high-level categories, NOT our 6-category C1a-C4b taxonomy
        "paper_categories": ["Repetition", "Unresponsiveness", "Hallucination"],
        # Exact percentages not reported in paper text; only visual in figure.
        # Mapping from paper categories to our taxonomy:
        "category_mapping": {
            "Repetition": ["C4a", "C4b"],        # Spammy = Repetition
            "Unresponsiveness": ["C1a", "C1b"],   # Unanswered = Unresponsiveness
            "Hallucination": ["C2", "C3b"],       # Vague/Incorrect = Hallucination
        },
    },

    # Table 1: Failure symptoms (10 categories)
    "failure_symptoms": {
        "Work overlap": 33.2,
        "Divergent architecture": 29.7,
        "Repetition": 14.7,
        "Unresponsiveness": 8.7,
        "Unverifiable claims": 4.3,
        "Broken commitment": 3.7,
        "Dependency access": 1.7,
        "Placeholder misuse": 1.5,
        "Parameter flow": 1.3,
        "Timing dependency": 1.1,
    },
}
```

### Pattern 3: Figure 4 Visualization (Sparse Data)
**What:** Difficulty-stratified success curves with only 3 populated buckets.
**When to use:** FIG4-07.
**Design decisions for sparse data:**
- Use markers (scatter points) at populated bucket centers, NOT smooth interpolated curves
- Connect markers with thin lines to show trend direction
- CI bands (`fill_between`) at each populated bucket only
- Leave empty x-axis region between buckets to honestly represent missing data
- X-axis spans [0, 1] with tick marks at all 10 bucket centers, but data only at 3
- Annotate "3 of 10 buckets populated" in figure subtitle or caption

```python
def plot_fig4(metrics, paper_baselines):
    fig, ax = plt.subplots(figsize=(8, 5))

    buckets = metrics["buckets"]
    x = np.array([b["center"] for b in buckets])

    for setting_key, color, label in [
        ("solo", COLORS["solo"], "Solo (Command A)"),
        ("coop_comm", COLORS["coop_comm"], "Coop-Comm (Command A)"),
        ("coop_nocomm", COLORS["coop_nocomm"], "Coop-NoComm (Command A)"),
    ]:
        rates = np.array([b[setting_key]["rate"] for b in buckets])
        ci_lo = np.array([b[setting_key]["ci_lower"] for b in buckets])
        ci_hi = np.array([b[setting_key]["ci_upper"] for b in buckets])

        # CI shaded bands
        ax.fill_between(x, ci_lo, ci_hi, color=color, alpha=0.15)
        # Rate line with markers
        ax.plot(x, rates, "-o", color=color, label=label,
                markersize=6, linewidth=1.5, zorder=3)

    # Paper baseline: pooled AUC as horizontal reference
    pooled = paper_baselines["fig4"]["pooled"]
    ax.axhline(y=pooled["solo_auc"] / (1 - 0),  # Note: AUC != avg rate; use annotated text
               linestyle=":", color=COLORS["paper"], alpha=0)  # Not a direct y-value comparison

    # Annotate AUC values
    auc = metrics["auc"]
    text = (
        f"AUC: Solo={auc['solo']['value']:.3f}, "
        f"Coop-Comm={auc['coop_comm']['value']:.3f}, "
        f"Coop-NoComm={auc['coop_nocomm']['value']:.3f}\n"
        f"Paper pooled: Solo={pooled['solo_auc']:.3f}, "
        f"Coop={pooled['coop_auc']:.3f}, "
        f"Retention={pooled['retention']:.2f}"
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel("Task Difficulty d(t)")
    ax.set_ylabel("Success Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(np.arange(0.05, 1.0, 0.1))
    ax.set_title("Figure 4: Difficulty-Stratified Success Curves\n"
                 "(3 of 10 difficulty buckets populated)")
    ax.legend(loc="upper right")

    return fig
```

### Pattern 4: Figure 5 Three-Panel Layout
**What:** Communication effects as 3-panel figure.
**When to use:** FIG5-05.
**Design decisions:**
- Panel (a): Grouped bars for success rates (both 0% -- annotate "0% success in both settings")
- Panel (b): Grouped bars for merge conflict rates (this has real signal: 41% vs 55%)
- Panel (c): Stacked or grouped bars for speech act breakdown + overhead annotation

```python
def plot_fig5(metrics, paper_baselines):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Panel (a): Success Rates
    settings = ["coop_comm", "coop_nocomm"]
    labels = ["Comm", "No-Comm"]
    colors = [COLORS["coop_comm"], COLORS["coop_nocomm"]]
    success_rates = [metrics["success_rates"][s]["rate"] * 100 for s in settings]
    bars = ax1.bar(labels, success_rates, color=colors, edgecolor="white")
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("(a) Success Rates")
    ax1.set_ylim(0, 10)  # Low range since both are 0%
    ax1.annotate("0% in both settings", xy=(0.5, 0.5),
                 xycoords="axes fraction", ha="center", fontsize=10,
                 color="gray", style="italic")

    # Panel (b): Merge Conflict Rates
    conflict_rates = [metrics["merge_conflict_rates"][s]["rate"] * 100 for s in settings]
    bars = ax2.bar(labels, conflict_rates, color=colors, edgecolor="white")
    ax2.set_ylabel("Merge Conflict Rate (%)")
    ax2.set_title("(b) Merge Conflict Rates")
    # Paper baseline: first-turn planning effect
    planning = paper_baselines["fig5"]["first_turn_planning"]
    ax2.axhline(y=planning["conflict_with"], linestyle="--",
                color=COLORS["paper"], alpha=0.5,
                label=f"Paper: with planning ({planning['conflict_with']}%)")
    for bar, rate in zip(bars, conflict_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{rate:.0f}%", ha="center", fontweight="bold")

    # Panel (c): Communication Overhead Breakdown
    speech = metrics["speech_acts"]
    categories = ["plan", "question", "update", "other"]
    counts = [speech[c]["pct"] for c in categories]
    ax3.bar(categories, counts, color=COLORS["coop_comm"], edgecolor="white")
    ax3.set_ylabel("% of Messages")
    ax3.set_title(f"(c) Speech Acts (overhead: {metrics['overhead']['mean_pct']:.1f}%)")
    # Paper baseline
    paper_overhead = paper_baselines["fig5"]["comm_overhead_pct"]
    ax3.axhline(y=33.3, linestyle=":", color=COLORS["paper"], alpha=0.4,
                label=f"Paper: ~1/3 each ({paper_overhead}% overhead)")

    fig.suptitle("Figure 5: Communication Effects", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig
```

### Pattern 5: Figure 6 Bar Chart
**What:** Communication error frequency bar chart.
**When to use:** FIG6-03.
**Design decisions:**
- Vertical bar chart with 6 categories on x-axis
- Primary y-axis: percentage of total errors
- Bar annotations showing both count and percentage
- Paper comparison: map 3 paper categories to our 6 categories using grouping

```python
def plot_fig6(metrics, paper_baselines):
    fig, ax = plt.subplots(figsize=(8, 5))

    freq = metrics["frequency"]
    categories = ["C1a", "C1b", "C2", "C3b", "C4a", "C4b"]
    category_labels = [
        "C1a\nUnanswered\n(No Reply)",
        "C1b\nUnanswered\n(Ignored)",
        "C2\nNon-answer\n(Vague)",
        "C3b\nIncorrect\n(Corrected)",
        "C4a\nSpammy\n(Same Info)",
        "C4b\nSpammy\n(Near-dup)",
    ]
    pcts = [freq[c]["pct_of_errors"] for c in categories]
    counts = [freq[c]["count"] for c in categories]

    bars = ax.bar(range(len(categories)), pcts, color=COLORS["coop_comm"],
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(category_labels, fontsize=8)
    ax.set_ylabel("% of Total Errors")
    ax.set_title("Figure 6: Communication Error Taxonomy")

    # Add count annotations
    for bar, count, pct in zip(bars, counts, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"n={count}\n({pct:.1f}%)", ha="center", fontsize=8)

    # Paper category grouping brackets
    mapping = paper_baselines["fig6"]["category_mapping"]
    # Add grouped bracket annotations for paper categories
    # "Unresponsiveness" spans C1a+C1b, "Hallucination" spans C2+C3b, "Repetition" spans C4a+C4b

    return fig
```

### Anti-Patterns to Avoid
- **Smooth interpolation with 3 data points:** Do NOT fit curves through 3 points. Use scatter+line. The 7 empty buckets are genuinely missing data, not zero-rate data.
- **Using project venv for figure scripts:** The project `.venv` does NOT have matplotlib. Use system Python (`/usr/bin/python3`).
- **Confusing AUC with average success rate:** AUC is area under the difficulty-stratified curve, not a simple average. Paper AUC values (pooled solo=0.338) cannot be plotted directly on the y-axis of a success rate chart. They belong in annotation text or a separate comparison table.
- **Treating paper's 3 error categories as identical to our 6:** Paper uses Repetition/Unresponsiveness/Hallucination. Our data uses C1a/C1b/C2/C3b/C4a/C4b. Use the mapping (Repetition=C4a+C4b, Unresponsiveness=C1a+C1b, Hallucination=C2+C3b) for comparison, but display our full 6-category breakdown.
- **Type 3 fonts in PDF:** Use `pdf.fonttype: 42` (TrueType) to ensure fonts embed properly in the PDF. Type 3 fonts cause issues with some viewers and arXiv submission.
- **Hardcoding DPI below 300:** Requirements specify 300+ DPI. Use `dpi=300` consistently.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CI shaded bands | Custom polygon patches | `ax.fill_between()` | Built-in, handles edge cases, well-tested |
| Multi-panel layout | Manual figure positioning | `plt.subplots(1, 3)` | Handles spacing, alignment, shared axes automatically |
| PDF font embedding | Manual font subsetting | `pdf.fonttype: 42` in rcParams | Standard matplotlib approach, no external tools needed |
| Color management | Manual RGB tuples | Named hex colors in a COLORS dict | Consistent across all figures, easy to update |
| Annotation boxes | Manual rectangle + text | `ax.text(..., bbox=dict(...))` | Built-in text box support |
| Axis percentage formatting | Manual string formatting | `matplotlib.ticker.PercentFormatter()` | Standard, handles edge cases |

**Key insight:** matplotlib has built-in support for everything needed. The complexity is in the design decisions (how to represent sparse data, where to place annotations, how to map paper categories), not the plotting mechanics.

## Common Pitfalls

### Pitfall 1: Matplotlib Not Found in Venv
**What goes wrong:** `ModuleNotFoundError: No module named 'matplotlib'` when running figure scripts with the project venv.
**Why it happens:** The project `.venv` at `/mnt/data/terry/home/cooperbench-repro/.venv/` has numpy but NOT matplotlib. Phase 3 analysis scripts don't need matplotlib (they produce JSON).
**How to avoid:** Figure scripts MUST use system Python (`/usr/bin/python3` or bare `python3`), which has matplotlib 3.10.8 installed. Use `#!/usr/bin/env python3` shebang, NOT the venv Python. Document this in script headers.
**Warning signs:** ImportError on `matplotlib.pyplot`.

### Pitfall 2: Overplotting Empty Buckets as Zero
**What goes wrong:** Plotting 0% success rate for all 10 difficulty buckets, making the figure look like "nothing happened" everywhere.
**Why it happens:** 7 of 10 buckets have no data. Plotting them as 0% is misleading because 0% suggests "we tested and found no success" when the truth is "we have no data for these difficulty ranges."
**How to avoid:** Plot ONLY the 3 populated buckets (centers 0.35, 0.65, 0.95). Leave the x-axis spanning [0,1] for context, but draw data only at populated points. Add annotation explaining data sparsity.
**Warning signs:** A flat line at y=0 across the entire difficulty range.

### Pitfall 3: Direct AUC Comparison on Y-Axis
**What goes wrong:** Trying to plot paper's pooled AUC (0.338 solo, 0.200 coop) as horizontal lines on Figure 4's y-axis.
**Why it happens:** AUC is computed over the full [0,1] difficulty range with 10 populated buckets in the paper; it integrates the area under the curve. It's NOT a success rate value that belongs on the y-axis.
**How to avoid:** Present AUC comparisons as text annotations, not reference lines. Use a separate inset or text box: "Paper pooled AUC: solo=0.338, coop=0.200 (5 models, 10 buckets). Ours: solo=0.15, coop=0.0 (1 model, 3 buckets)."
**Warning signs:** A horizontal line at y=0.338 that doesn't correspond to any actual success rate.

### Pitfall 4: Paper Figure Numbers Don't Match Ours
**What goes wrong:** The CooperBench paper's Figure 4 shows communication effects (our Figure 5), and the paper's Figure 6 shows difficulty curves (our Figure 4). This creates confusion when extracting "paper baseline numbers for Figure 4."
**Why it happens:** The paper and our project use different figure numbering. The paper has Figures 1-6 covering different aspects; our REQUIREMENTS.md defines "Figure 4" = difficulty curves, "Figure 5" = communication, "Figure 6" = error taxonomy.
**How to avoid:** Map requirements to paper content by TOPIC, not number. Our Figure 4 (difficulty) maps to paper's Table 5 and Figure 6 content. Our Figure 5 (communication) maps to paper's Figure 4. Our Figure 6 (errors) maps to paper's Figure 5. Document this mapping in the paper_baselines module.
**Warning signs:** Looking at paper's "Figure 4" for AUC data (it shows communication effects, not difficulty curves).

### Pitfall 5: agg Backend and Interactive Display
**What goes wrong:** Calling `plt.show()` on a headless server produces an error or hangs.
**Why it happens:** The matplotlib backend is `agg` (non-interactive), which only supports saving to files.
**How to avoid:** Never use `plt.show()`. Always use `fig.savefig()`. This is already the pattern in `generate_phase1_figures.py`.
**Warning signs:** Hanging script or "cannot connect to display" errors.

### Pitfall 6: Misaligned Paper Category Comparison for Figure 6
**What goes wrong:** Trying to overlay paper's Repetition/Unresponsiveness/Hallucination percentages directly onto our C1a-C4b bar chart, creating an unreadable figure.
**Why it happens:** Paper uses 3 categories, we use 6. The categories don't align 1:1.
**How to avoid:** Two approaches: (a) Show our 6 categories as primary bars, add colored grouping brackets at the top mapping to paper's 3 categories. (b) Create a secondary grouped comparison showing our data aggregated into 3 paper-style categories alongside the paper's values. Approach (a) is preferred because it shows our full detail while still enabling comparison.
**Warning signs:** 6+3 = 9 bars on one chart that nobody can read.

## Code Examples

### Publication-Quality rcParams Setup
```python
# Source: matplotlib 3.10.8 documentation + academic publishing conventions
import matplotlib
matplotlib.use("agg")  # Ensure headless backend
import matplotlib.pyplot as plt

# Apply style and custom overrides
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    # Font sizes
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 14,

    # Figure appearance
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#cccccc",
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.5,

    # Export quality
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "savefig.facecolor": "white",

    # PDF font embedding (TrueType, not Type 3)
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # Line and marker defaults
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
})
```

### Dual-Format Export (PDF + PNG)
```python
def save_figure(fig, name: str, fig_dir: Path):
    """Save figure as both PDF and PNG at 300 DPI.

    PDF uses vector graphics (ideal for papers).
    PNG is raster at 300 DPI (for previews and web).
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        path = fig_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  -> {path}")
    plt.close(fig)
```

### CI Band Visualization for Sparse Data
```python
import numpy as np

def plot_difficulty_curves_sparse(ax, buckets, colors):
    """Plot difficulty curves with CI bands for sparse bucket data.

    Handles the case where only 3 of 10 buckets are populated.
    Uses scatter+line rather than smooth curves.
    """
    x = np.array([b["center"] for b in buckets])

    for setting_key, color, label in [
        ("solo", colors["solo"], "Solo"),
        ("coop_comm", colors["coop_comm"], "Coop-Comm"),
        ("coop_nocomm", colors["coop_nocomm"], "Coop-NoComm"),
    ]:
        rates = np.array([b[setting_key]["rate"] for b in buckets])
        ci_lo = np.array([b[setting_key]["ci_lower"] for b in buckets])
        ci_hi = np.array([b[setting_key]["ci_upper"] for b in buckets])

        # CI shaded bands (only at populated points)
        for i in range(len(x)):
            ax.fill_between(
                [x[i] - 0.03, x[i] + 0.03],
                [ci_lo[i], ci_lo[i]],
                [ci_hi[i], ci_hi[i]],
                color=color, alpha=0.15,
            )

        # Rate points with line
        ax.plot(x, rates, "-o", color=color, label=label,
                markersize=7, linewidth=1.5, zorder=3)

    # X-axis shows all 10 bucket centers for context
    all_centers = [round(0.05 + 0.1 * i, 2) for i in range(10)]
    ax.set_xticks(all_centers)
    ax.set_xticklabels([f"{c:.1f}" for c in all_centers], fontsize=8, rotation=45)

    # Shade unpopulated regions
    populated = set(b["center"] for b in buckets)
    for c in all_centers:
        if c not in populated:
            ax.axvspan(c - 0.05, c + 0.05, color="#f0f0f0", alpha=0.3, zorder=0)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Task Difficulty d(t)")
    ax.set_ylabel("Success Rate")
```

### Three-Panel Figure Layout
```python
def create_fig5_layout():
    """Create Figure 5 three-panel layout with consistent sizing.

    Returns (fig, (ax1, ax2, ax3)).
    """
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3,
        figsize=(15, 5),
        gridspec_kw={"wspace": 0.35},  # Wider spacing between panels
    )

    # Panel labels
    ax1.set_title("(a) Success Rates", fontsize=12, fontweight="bold")
    ax2.set_title("(b) Merge Conflict Rates", fontsize=12, fontweight="bold")
    ax3.set_title("(c) Speech Acts & Overhead", fontsize=12, fontweight="bold")

    return fig, (ax1, ax2, ax3)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `plt.style.use("seaborn-whitegrid")` | `plt.style.use("seaborn-v0_8-whitegrid")` | matplotlib 3.6+ (2023) | Old seaborn style names deprecated, renamed with `v0_8` prefix |
| `fig.savefig(..., dpi=200)` | `fig.savefig(..., dpi=300)` | Convention | 300 DPI is minimum for publication; 200 is adequate for screens only |
| `pdf.fonttype: 3` (default) | `pdf.fonttype: 42` (TrueType) | Best practice | Type 3 fonts cause issues in PDF viewers and arXiv; TrueType is universal |
| Manual tight_layout | `constrained_layout=True` or `bbox_inches="tight"` | matplotlib 3.x | Both work; `bbox_inches="tight"` is simpler for our use case |
| `numpy.trapz` | `numpy.trapezoid` | NumPy 2.0 (2024) | Old name deprecated. We have NumPy 2.4.2. Not needed for plotting (AUC already computed). |

**Deprecated/outdated:**
- `plt.style.use("seaborn-whitegrid")` -- must use `seaborn-v0_8-whitegrid` in matplotlib 3.6+
- `numpy.trapz` -- renamed to `numpy.trapezoid` in NumPy 2.0

## Open Questions

1. **Per-model paper data for Figure 4 overlay: lines vs text?**
   - What we know: Paper Table 5 gives per-model AUC values. Paper Figure 6 shows per-model difficulty curves. Our data is single-model.
   - What's unclear: Should we overlay all 5 paper model curves (from visual estimation) or just show pooled numbers? Visual estimation from a figure is imprecise.
   - Recommendation: Show pooled AUC comparison as text annotation only. Do NOT attempt to reverse-engineer per-model curves from the paper's figures. Use a comparison table inset: "Paper pooled retention: 0.59 | Ours: 0.0". Per-model AUC values from Table 5 can go in a supplementary table.
   - Confidence: HIGH -- attempting to extract curve points from paper figures is error-prone and adds no value.

2. **Figure 6 paper comparison: exact paper error percentages?**
   - What we know: Paper Figure 5 shows Repetition/Unresponsiveness/Hallucination frequencies but exact percentages are not reported in text. Paper Table 1 shows failure symptoms (work overlap 33.2%, etc.) which is a DIFFERENT taxonomy.
   - What's unclear: Whether we can extract approximate percentages from the paper's figure, or if we should note "paper percentages not available in text."
   - Recommendation: Show our 6-category data as the primary visualization. Add grouping brackets mapping to paper's 3 categories. Note in figure text: "Paper reports Repetition/Unresponsiveness/Hallucination but does not provide exact percentages." Do NOT fabricate numbers.
   - Confidence: HIGH -- honest about data limitations.

3. **Figure 5 Panel (a): What to do with two 0% bars?**
   - What we know: Both coop-comm and coop-nocomm have 0% success rate. Two empty bars are visually uninformative.
   - What's unclear: Should we still show them, or replace with a text annotation?
   - Recommendation: Show the bars (even at 0%) with Wilson CI upper bounds as error bars, plus text annotation "0% success in both settings (n=96 comm, n=88 nocomm, all bucket 9)." This is honest and complete. The paper reports ~25% coop success for top models -- add as reference line.
   - Confidence: MEDIUM -- design choice, either approach is valid.

4. **Output directory for figures?**
   - What we know: No `figures/` directory exists. Phase 1 figures went to `~/multiagent/experiments/figures/12_cooperbench_repro/`.
   - What's unclear: Where Phase 4 figures should go.
   - Recommendation: Create `figures/` at project root (`/mnt/data/terry/home/cooperbench-repro/figures/`). This follows the `scripts/` and `data/` convention. The Phase 1 external location was for a different project context.
   - Confidence: HIGH -- consistent with project structure.

## Paper-to-Project Figure Number Mapping

**Critical for COMP-01:** The paper's figure numbers do NOT match our project's figure numbers.

| Our Figure # | Our Content | Paper Figure # | Paper Content | Paper Data Source |
|---|---|---|---|---|
| Figure 4 | Difficulty-stratified success curves | Figure 6 | Difficulty curves + Wilson CIs | Table 5 (AUC, retention) |
| Figure 5 | Communication effects (3-panel) | Figure 4 | Communication effects (3-panel) | Text: "20% overhead", "1/3 each" |
| Figure 6 | Communication error taxonomy | Figure 5 | Communication error detection | Text only (no exact %) |

This mapping MUST be documented in the paper_baselines module to prevent confusion.

## Sources

### Primary (HIGH confidence)
- `scripts/generate_phase1_figures.py` -- Existing matplotlib convention in this project: colors, style, layout patterns
- `data/fig4_metrics.json` -- Actual Figure 4 input data (3 buckets, AUC, retention)
- `data/fig5_metrics.json` -- Actual Figure 5 input data (success rates, conflict rates, speech acts, overhead)
- `data/fig6_metrics.json` -- Actual Figure 6 input data (C1a-C4b error classifications and frequencies)
- System Python matplotlib 3.10.8 -- Verified: agg backend, PDF/PNG export, fill_between, subplots, seaborn-v0_8-whitegrid style
- CooperBench paper Table 5 (arXiv:2601.13295) -- Per-model AUC and retention values (extracted via HTML)

### Secondary (MEDIUM confidence)
- CooperBench paper text (arXiv:2601.13295) -- Communication overhead "~20%", speech acts "~1/3 each", first-turn planning conflict rates (29.4% vs 51.5%)
- CooperBench paper Figure 4 caption -- "Communication effects" (maps to our Figure 5)
- CooperBench paper Figure 5 caption -- "Communication error frequencies" with 3 categories
- matplotlib 3.10.8 documentation -- rcParams, savefig options, fill_between API

### Tertiary (LOW confidence)
- Paper's Figure 5 exact error percentages -- NOT reported in paper text; only visual bar chart. Cannot extract exact numbers.
- Paper's per-model merge conflict rates -- Shown visually in paper Figure 4(b) but not reported numerically.
- Color scheme of paper's figures -- No hex codes or color names provided in paper; not critical for our figures.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- matplotlib 3.10.8 verified on system Python with all needed features. Project color scheme established in Phase 1.
- Architecture: HIGH -- Clear input (JSON) to output (PDF/PNG) pipeline. Script pattern proven.
- Paper baselines: MEDIUM -- AUC/retention values from Table 5 are exact. Communication effects numbers are approximate ("~20%", "~1/3 each"). Figure 6 comparison limited by paper's different taxonomy and missing exact percentages.
- Pitfalls: HIGH -- All identified from actual testing and codebase analysis.
- Visualization design: MEDIUM -- Sparse data (3 buckets, 0% coop rates) requires careful design decisions; patterns recommended but need refinement during implementation.

**Research date:** 2026-02-18
**Valid until:** 2026-03-18 (stable -- data is fixed, matplotlib version pinned by system)
