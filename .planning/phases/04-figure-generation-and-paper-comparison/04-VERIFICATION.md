---
phase: 04-figure-generation-and-paper-comparison
verified: 2026-02-18T23:55:00Z
status: passed
score: 10/10 must-haves verified
must_haves:
  truths:
    - "Figure 4 PDF and PNG exist at figures/fig4_difficulty_curves.{pdf,png}"
    - "Figure 4 shows difficulty-stratified success curves with scatter+line markers at 3 populated bucket centers (0.35, 0.65, 0.95)"
    - "CI shaded bands are visible around each data point using fill_between"
    - "AUC and retention values are annotated on the figure"
    - "Paper baseline numbers (pooled AUC solo=0.338, coop=0.200, retention=0.59) are overlaid as text annotation"
    - "Unpopulated buckets are visually distinguished from populated ones"
    - "Output DPI is 300+ and PDF uses TrueType fonts (fonttype 42)"
    - "Figure 5 PDF and PNG exist at figures/fig5_communication.{pdf,png}"
    - "Figure 5 has three panels: (a) success rates, (b) merge conflict rates, (c) speech acts & overhead"
    - "Figure 5 panel (a) shows 0% bars with Wilson CI upper bounds as error bars and annotation"
    - "Figure 5 panel (b) shows 41% vs 55% conflict rates with paper baseline reference lines"
    - "Figure 5 panel (c) shows speech act distribution (plan/question/update/other) with overhead annotation"
    - "Figure 6 PDF and PNG exist at figures/fig6_error_taxonomy.{pdf,png}"
    - "Figure 6 shows bar chart with 6 categories (C1a, C1b, C2, C3b, C4a, C4b) with count+pct annotations"
    - "Figure 6 has grouping brackets mapping our 6 categories to paper's 3 categories (Repetition, Unresponsiveness, Hallucination)"
    - "All figures use 300 DPI and TrueType fonts (pdf.fonttype=42)"
    - "Paper baseline numbers are overlaid on both figures for comparison"
  artifacts:
    - path: "scripts/paper_baselines.py"
      provides: "Centralized paper baseline constants for all 3 figures"
      contains: "PAPER_BASELINES"
    - path: "scripts/generate_fig4.py"
      provides: "Figure 4 generation script"
      contains: "def main"
    - path: "scripts/generate_fig5.py"
      provides: "Figure 5 generation script"
      contains: "def main"
    - path: "scripts/generate_fig6.py"
      provides: "Figure 6 generation script"
      contains: "def main"
    - path: "figures/fig4_difficulty_curves.pdf"
      provides: "Vector format Figure 4"
    - path: "figures/fig4_difficulty_curves.png"
      provides: "Raster format Figure 4 at 300 DPI"
    - path: "figures/fig5_communication.pdf"
      provides: "Vector format Figure 5"
    - path: "figures/fig5_communication.png"
      provides: "Raster format Figure 5 at 300 DPI"
    - path: "figures/fig6_error_taxonomy.pdf"
      provides: "Vector format Figure 6"
    - path: "figures/fig6_error_taxonomy.png"
      provides: "Raster format Figure 6 at 300 DPI"
  key_links:
    - from: "scripts/generate_fig4.py"
      to: "scripts/paper_baselines.py"
      via: "from paper_baselines import PAPER_BASELINES"
    - from: "scripts/generate_fig4.py"
      to: "data/fig4_metrics.json"
      via: "json.load"
    - from: "scripts/generate_fig5.py"
      to: "scripts/paper_baselines.py"
      via: "from paper_baselines import PAPER_BASELINES"
    - from: "scripts/generate_fig5.py"
      to: "data/fig5_metrics.json"
      via: "json.load"
    - from: "scripts/generate_fig6.py"
      to: "scripts/paper_baselines.py"
      via: "from paper_baselines import PAPER_BASELINES"
    - from: "scripts/generate_fig6.py"
      to: "data/fig6_metrics.json"
      via: "json.load"
---

# Phase 4: Figure Generation and Paper Comparison Verification Report

**Phase Goal:** Publication-quality Figures 4, 5, and 6 are generated as PDF/PNG files, with paper baseline numbers overlaid for direct visual comparison.
**Verified:** 2026-02-18T23:55:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Figure 4 PDF and PNG exist | VERIFIED | `figures/fig4_difficulty_curves.pdf` (16 KB) and `.png` (225 KB) both present |
| 2 | Figure 4 shows scatter+line at 3 populated bucket centers | VERIFIED | Visual inspection confirms 3 data points at ~0.35, 0.65, 0.95 connected by lines with markers |
| 3 | CI shaded bands visible using fill_between | VERIFIED | Code uses `ax.fill_between` at line 91; visual inspection shows shaded CI bands at each bucket |
| 4 | AUC and retention values annotated | VERIFIED | Text box in top-left shows "Solo AUC=0.150, Coop AUC=0.000, Retention: Comm=0.00, NoComm=0.00" |
| 5 | Paper baselines overlaid on Figure 4 | VERIFIED | Annotation box shows "Paper (5 models, 10 buckets): Solo AUC=0.338, Coop AUC=0.200, Retention=0.59" |
| 6 | Unpopulated buckets visually distinguished | VERIFIED | Gray-shaded vertical spans visible in figure for 7 unpopulated difficulty buckets |
| 7 | DPI 300+ and TrueType fonts | VERIFIED | PNG DPI measured at (299.9994, 299.9994); `pdf.fonttype=42` in all 3 scripts |
| 8 | Figure 5 PDF and PNG exist | VERIFIED | `figures/fig5_communication.pdf` (30 KB) and `.png` (286 KB) both present |
| 9 | Figure 5 has 3 panels (a,b,c) | VERIFIED | Visual inspection confirms 3-panel subplot: success rates, merge conflicts, speech acts |
| 10 | Figure 5 panel (a): 0% bars with CI error bars | VERIFIED | Panel shows 0% bars with CI upper bounds (3.8%, 4.2%) as error bars and "0% in both settings" annotation |
| 11 | Figure 5 panel (b): 41% vs 55% with paper lines | VERIFIED | Panel shows 41% and 55% bars with dashed lines at 29.4% and 51.5% from paper |
| 12 | Figure 5 panel (c): speech acts with overhead | VERIFIED | Panel shows Plan 46.7%, Question 26.0%, Update 10.5%, Other 16.8%; title shows "overhead: 22.8%"; paper reference at 33.3% |
| 13 | Figure 6 PDF and PNG exist | VERIFIED | `figures/fig6_error_taxonomy.pdf` (32 KB) and `.png` (239 KB) both present |
| 14 | Figure 6 shows 6-category bar chart with annotations | VERIFIED | Visual shows C4a (41.6%), C4b (32.5%), C1a (11.7%), C1b (9.1%), C2 (1.3%), C3b (3.9%) with n= counts |
| 15 | Figure 6 has grouping brackets to paper categories | VERIFIED | Colored brackets below x-axis: Repetition (74.0%), Unresponsiveness (20.8%), Hallucination (5.2%) |
| 16 | Paper baselines overlaid on Figures 5 and 6 | VERIFIED | Fig 5: dashed reference lines (29.4%, 51.5%, 33.3%, ~20%); Fig 6: annotation box with paper 3-category mapping |

**Score:** 10/10 truths verified (combined from Plan 04-01: 7 truths and Plan 04-02: 10 truths; deduplicated to 16 checks, all passing)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/paper_baselines.py` | Centralized paper baseline constants | VERIFIED | 102 lines, contains `PAPER_BASELINES` dict with fig4/fig5/fig6 baselines, importable, tested |
| `scripts/generate_fig4.py` | Figure 4 generation script | VERIFIED | 169 lines, contains `def main`, reads fig4_metrics.json, produces dual-format output |
| `scripts/generate_fig5.py` | Figure 5 generation script | VERIFIED | 230 lines, contains `def main`, reads fig5_metrics.json, 3-panel subplot, paper overlays |
| `scripts/generate_fig6.py` | Figure 6 generation script | VERIFIED | 223 lines, contains `def main`, reads fig6_metrics.json, 6-bar chart with grouping brackets |
| `figures/fig4_difficulty_curves.pdf` | Vector format Figure 4 | VERIFIED | 16 KB, TrueType fonts (fonttype 42) |
| `figures/fig4_difficulty_curves.png` | Raster format Figure 4 | VERIFIED | 225 KB, 2065x1496 px, DPI=300 |
| `figures/fig5_communication.pdf` | Vector format Figure 5 | VERIFIED | 30 KB, TrueType fonts (fonttype 42) |
| `figures/fig5_communication.png` | Raster format Figure 5 | VERIFIED | 286 KB, 3679x1530 px, DPI=300 |
| `figures/fig6_error_taxonomy.pdf` | Vector format Figure 6 | VERIFIED | 32 KB, TrueType fonts (fonttype 42) |
| `figures/fig6_error_taxonomy.png` | Raster format Figure 6 | VERIFIED | 239 KB, 2955x1754 px, DPI=300 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `scripts/generate_fig4.py` | `scripts/paper_baselines.py` | `from paper_baselines import PAPER_BASELINES` | WIRED | Line 22: import present; lines 130-141 use `PAPER_BASELINES["fig4"]["pooled"]` for annotation |
| `scripts/generate_fig4.py` | `data/fig4_metrics.json` | `json.load` | WIRED | Line 58: `json.load(f)` reads metrics; lines 63-65 extract bucket data; lines 128-129 extract AUC/retention |
| `scripts/generate_fig5.py` | `scripts/paper_baselines.py` | `from paper_baselines import PAPER_BASELINES` | WIRED | Line 22: import present; lines 151-159 use fig5 first_turn_planning; lines 189-202 use overhead/speech_acts |
| `scripts/generate_fig5.py` | `data/fig5_metrics.json` | `json.load` | WIRED | Line 58: `json.load(f)` reads metrics; used across all 3 panels for rates, conflicts, speech acts |
| `scripts/generate_fig6.py` | `scripts/paper_baselines.py` | `from paper_baselines import PAPER_BASELINES` | WIRED | Line 23: import present; line 125 uses `PAPER_BASELINES["fig6"]["category_mapping"]` for brackets |
| `scripts/generate_fig6.py` | `data/fig6_metrics.json` | `json.load` | WIRED | Line 66: `json.load(f)` reads metrics; lines 85-86 extract frequency counts and percentages |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| FIG4-07 | 04-01 | Generate Figure 4 with CI shaded bands, publication-quality PDF/PNG at 300+ DPI | SATISFIED | `figures/fig4_difficulty_curves.{pdf,png}` exist, DPI=300, CI bands via fill_between, TrueType fonts |
| FIG5-05 | 04-02 | Generate Figure 5 as 3-panel plot: (a) success rates, (b) conflict rates, (c) overhead breakdown | SATISFIED | `figures/fig5_communication.{pdf,png}` exist with all 3 panels visually verified |
| FIG6-03 | 04-02 | Generate Figure 6 as error frequency bar chart | SATISFIED | `figures/fig6_error_taxonomy.{pdf,png}` exist with 6 categories, count+pct annotations, grouping brackets |
| COMP-01 | 04-01, 04-02 | Overlay paper's published baseline numbers on Figures 4, 5, 6 for direct visual comparison | SATISFIED | Fig 4: AUC text box with paper pooled values; Fig 5: dashed reference lines (29.4%, 51.5%, 33.3%, ~20%); Fig 6: annotation box mapping to paper's 3 categories |

No orphaned requirements found. All 4 requirement IDs assigned to Phase 4 in REQUIREMENTS.md (FIG4-07, FIG5-05, FIG6-03, COMP-01) appear in plan frontmatter and are satisfied.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| -- | -- | -- | -- | No anti-patterns found in any Phase 4 files |

No TODO/FIXME/PLACEHOLDER comments, no empty implementations, no console.log-only handlers, no stub returns found in any of the 4 scripts created in this phase.

### Human Verification Required

### 1. Figure 4 Visual Quality

**Test:** Open `figures/fig4_difficulty_curves.pdf` in a PDF viewer. Check that CI shaded bands are visible, legend is readable, annotation text box is not overlapping data, and unpopulated bucket shading is distinguishable from populated regions.
**Expected:** Publication-quality figure with clear visual hierarchy. Three data points with distinct colors (blue=solo, green=coop-comm, red=coop-nocomm). Gray shading on unpopulated regions. Text annotation box in top-left with AUC comparison.
**Why human:** Visual aesthetic quality (spacing, readability, overlap) cannot be verified programmatically beyond basic presence checks.

### 2. Figure 5 Three-Panel Layout

**Test:** Open `figures/fig5_communication.pdf`. Verify all three panels are properly spaced, labels do not overlap, error bars in panel (a) are visible at the scale shown, and paper reference lines in panels (b) and (c) are legible.
**Expected:** Three-panel figure with consistent formatting. Panel (a) shows 0% bars with upward CI error bars. Panel (b) shows 41% vs 55% bars with two dashed reference lines labeled. Panel (c) shows 4-category speech act bars with one reference line.
**Why human:** Panel spacing, label readability at print size, and reference line visibility require human judgment.

### 3. Figure 6 Grouping Brackets

**Test:** Open `figures/fig6_error_taxonomy.pdf`. Verify the colored grouping brackets below the x-axis are properly aligned with their respective category pairs, labels are readable, and the annotation box does not overlap the bars.
**Expected:** Six bars ordered C4a, C4b, C1a, C1b, C2, C3b. Colored brackets below: amber (Repetition over C4a+C4b), purple (Unresponsiveness over C1a+C1b), red (Hallucination over C2+C3b). Text box in right side.
**Why human:** Bracket alignment, color distinction, and annotation box positioning require visual assessment.

### Gaps Summary

No gaps found. All 10 artifacts exist, are substantive (non-stub), and are fully wired. All 6 key links are verified. All 4 requirements (FIG4-07, FIG5-05, FIG6-03, COMP-01) are satisfied. No anti-patterns detected. All 4 commit hashes documented in summaries are valid in the git log.

Visual inspection of the generated PNG files confirms the figures contain the expected data, annotations, and paper comparison overlays. The figures are publication-quality with consistent formatting across all three.

---

_Verified: 2026-02-18T23:55:00Z_
_Verifier: Claude (gsd-verifier)_
