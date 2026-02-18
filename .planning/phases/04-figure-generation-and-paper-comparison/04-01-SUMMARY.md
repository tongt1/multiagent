---
phase: 04-figure-generation-and-paper-comparison
plan: 01
subsystem: figures
tags: [matplotlib, publication-quality, difficulty-curves, paper-baselines, pdf, png]

# Dependency graph
requires:
  - phase: 03-analysis-modules
    provides: "fig4_metrics.json with bucket data, AUC, retention"
provides:
  - "scripts/paper_baselines.py: centralized paper baseline constants for all 3 figures"
  - "scripts/generate_fig4.py: Figure 4 generation script"
  - "figures/fig4_difficulty_curves.{pdf,png}: publication-quality Figure 4"
affects: [04-02-figure-generation]

# Tech tracking
tech-stack:
  added: []
  patterns: [publication-quality-matplotlib-rcParams, dual-format-pdf-png-export, paper-baselines-import-pattern]

key-files:
  created:
    - scripts/paper_baselines.py
    - scripts/generate_fig4.py
    - figures/fig4_difficulty_curves.pdf
    - figures/fig4_difficulty_curves.png

key-decisions:
  - "Paper baselines centralized in single PAPER_BASELINES dict for all 3 figures"
  - "CI bands rendered as narrow fill_between rectangles at discrete bucket centers (not continuous)"
  - "AUC/retention comparison shown as text annotation box, not reference lines on y-axis"

patterns-established:
  - "Figure script pattern: read JSON metrics, import PAPER_BASELINES, generate dual-format output"
  - "Publication rcParams: seaborn-v0_8-whitegrid style, 300 DPI, pdf.fonttype=42 (TrueType)"
  - "Color scheme: solo=#4878CF, coop_comm=#6ACC65, coop_nocomm=#D65F5F, paper=#888888"

requirements-completed: [FIG4-07, COMP-01]

# Metrics
duration: 2min
completed: 2026-02-18
---

# Phase 4 Plan 1: Paper Baselines and Figure 4 Summary

**Shared paper_baselines.py module with centralized CooperBench paper constants, plus publication-quality Figure 4 showing 3-bucket difficulty-stratified success curves with CI bands and paper AUC comparison overlay**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-18T23:34:36Z
- **Completed:** 2026-02-18T23:36:53Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Created `paper_baselines.py` with all paper baseline constants needed by Figures 4, 5, and 6, including the paper-to-project figure number mapping documentation
- Generated publication-quality Figure 4 as both PDF (TrueType fonts) and PNG (300 DPI) showing difficulty-stratified success curves for 3 populated buckets
- Paper comparison overlay shows pooled AUC values (solo=0.338 vs our 0.150, coop=0.200 vs our 0.000) and retention (0.59 vs our 0.00)
- Established figure generation pattern (rcParams, color scheme, dual-format export) for Figures 5 and 6

## Task Commits

Each task was committed atomically:

1. **Task 1: Create paper_baselines.py shared module** - `29713ca` (feat)
2. **Task 2: Generate Figure 4 difficulty-stratified success curves** - `fa436d9` (feat)

## Files Created/Modified
- `scripts/paper_baselines.py` - Centralized paper baseline constants (fig4/fig5/fig6) with figure number mapping docstring
- `scripts/generate_fig4.py` - Figure 4 generation script reading fig4_metrics.json and producing dual-format output
- `figures/fig4_difficulty_curves.pdf` - Vector format Figure 4 (16 KB, TrueType fonts)
- `figures/fig4_difficulty_curves.png` - Raster format Figure 4 (225 KB, 2065x1496 px at 300 DPI)

## Decisions Made
- Paper baselines centralized in single `PAPER_BASELINES` dict rather than per-script constants -- enables consistency across all 3 figure scripts
- CI bands rendered as narrow `fill_between` rectangles (center +/- 0.03) at discrete bucket centers rather than continuous bands -- honest representation of sparse 3-bucket data
- AUC/retention comparison shown as text annotation box (top-left) rather than horizontal reference lines on y-axis -- AUC is area under curve over [0,1] difficulty range, not a point value on the success rate axis
- Unpopulated bucket regions shaded light gray (#f0f0f0, alpha=0.3) to visually distinguish from populated data points

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Paper baselines module ready for import by generate_fig5.py and generate_fig6.py
- Figure generation pattern established (rcParams, colors, dual-format export)
- Ready for 04-02-PLAN.md (Figures 5 and 6)

## Self-Check: PASSED

All files verified present. All commit hashes verified in git log.

---
*Phase: 04-figure-generation-and-paper-comparison*
*Completed: 2026-02-18*
