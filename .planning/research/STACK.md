# Stack Research

**Domain:** Statistical analysis & figure generation for CooperBench paper reproduction (Figures 4, 5, 6)
**Researched:** 2026-02-14
**Confidence:** HIGH

## Context

Reproducing CooperBench paper figures with internal Cohere models. The analysis pipeline needs:
- Wilson confidence intervals for binomial pass/fail rates per difficulty bucket
- Trapezoidal AUC integration over difficulty-stratified success curves
- 10-bucket difficulty stratification (decile binning of tasks by difficulty score)
- 3-panel publication figures with error bands (Figures 4, 5, 6)
- LLM-based communication error taxonomy classification (C1a/C1b/C2/C3b/C4a/C4b)
- Qualitative transcript analysis: Plan:Question ratios, first-turn planning effects, specificity counts

The project already uses Python 3.11+, poetry/hatch, litellm, pydantic, instructor, cohere SDK, and matplotlib (lazy-imported in cooperbench-eval report generator).

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| numpy | >=2.2.0,<3.0 | Array operations, numerical backbone | Already implicitly depended on via other packages. Pin >=2.2 for `numpy.trapezoid` (renamed from `numpy.trapz` in 2.0). Avoid bleeding-edge 2.4 unless other deps are ready. **Confidence: HIGH** (PyPI verified: 2.4.2 released 2026-01-31) |
| pandas | >=2.2.0,<3.0 | DataFrames for task results, difficulty scores, aggregation | Already in project (`viewer` extras). Pin `<3.0` to avoid pandas 3.0 breaking changes (Copy-on-Write default, PyArrow-backed strings, NaN/NA unification). Pandas 3.0.0 released 2026-01-21 -- too fresh for production analysis code. **Confidence: HIGH** (PyPI verified) |
| scipy | >=1.14.0 | `scipy.integrate.trapezoid` for AUC, `scipy.stats` utilities | Stable trapezoidal integration API. `scipy.integrate.trapezoid` is the canonical name (replaces deprecated `trapz`). Latest: 1.17.0 (2026-01-10). Pin >=1.14 for Python 3.11 compat. **Confidence: HIGH** (official docs verified) |
| statsmodels | >=0.14.0 | `proportion_confint(method='wilson')` for Wilson score CIs | The only Python library with a one-call Wilson CI implementation. `statsmodels.stats.proportion.proportion_confint(count, nobs, alpha=0.05, method='wilson')` returns (lower, upper). Latest: 0.14.6 (2025-12-05). **Confidence: HIGH** (official docs verified) |
| matplotlib | >=3.9.0 | Figure generation: 3-panel layouts, error bands, bar charts | Already lazy-imported in `cooperbench-eval/src/report/generator.py`. The standard for publication figures. Latest: 3.10.8 (2025-12-10). Pin >=3.9 for stable `subplot_mosaic` and improved constrained_layout. **Confidence: HIGH** (PyPI verified) |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| seaborn | >=0.13.0 | Statistical plot aesthetics: `sns.set_theme()`, color palettes, heatmaps for error taxonomy matrices | Use for setting global plot theme and for any heatmap/distribution plots. NOT for core 3-panel figures (use raw matplotlib for pixel-precise control). Latest: 0.13.2 (2024-01-25). **Confidence: HIGH** |
| SciencePlots | >=2.2.0 | Pre-built matplotlib style sheets for publication formatting (`plt.style.use(['science', 'no-latex'])`) | Use for consistent publication styling without manual rcParams. Handles font sizes, grid, line widths. Use `no-latex` style to avoid LaTeX dependency. Latest: 2.2.0 (2025-11-20). **Confidence: MEDIUM** (PyPI verified, but optional -- can replicate with manual rcParams) |
| instructor | >=1.14.0 | Structured LLM output for communication error taxonomy classification (C1a-C4b) | Already a project dependency. Use with Cohere client for taxonomy classification. Pydantic models define the output schema, instructor handles validation + retries. Latest: 1.14.5. **Confidence: HIGH** (already in pyproject.toml) |
| litellm | >=1.81.0 | Unified LLM API for taxonomy classification calls | Already a project dependency. Use if you need to test taxonomy classification with non-Cohere models (GPT-4, Claude). **Confidence: HIGH** (already in pyproject.toml) |
| pydantic | >=2.12.0 | Schema definitions for analysis results, taxonomy classifications, report structures | Already a core dependency. Use for all data models in the analysis pipeline. **Confidence: HIGH** |
| loguru | >=0.7 | Logging throughout analysis pipeline | Already a project dependency. Prefer over stdlib logging for consistency. **Confidence: HIGH** |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| pytest >=8.2 | Testing analysis functions (CI computation, AUC, bucketing) | Already in dev dependencies. Write unit tests with known inputs/outputs for statistical functions. |
| ruff >=0.8.5 | Linting/formatting | Already configured in pyproject.toml. |
| mypy >=1.13 | Type checking analysis code | Already configured with pydantic plugin. |

## Installation

```bash
# Add to pyproject.toml [project.optional-dependencies] section:
# analysis = [...]
poetry add --group analysis numpy "pandas>=2.2.0,<3.0" scipy statsmodels matplotlib seaborn SciencePlots

# Or add directly to pyproject.toml:
[project.optional-dependencies]
analysis = [
    "numpy>=2.2.0,<3.0",
    "pandas>=2.2.0,<3.0",
    "scipy>=1.14.0",
    "statsmodels>=0.14.0",
    "matplotlib>=3.9.0",
    "seaborn>=0.13.0",
    "SciencePlots>=2.2.0",
]
```

Note: `instructor`, `litellm`, `pydantic`, and `cohere` are already core dependencies and do NOT need to be added again.

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| statsmodels `proportion_confint(method='wilson')` | Hand-rolled Wilson CI formula | Never. The formula is simple but error-prone with edge cases (0/0, n/n). Statsmodels handles all edge cases and is well-tested. |
| `scipy.integrate.trapezoid` | `numpy.trapezoid` | Either works identically for our use case. Prefer scipy because we already need scipy for stats, and scipy's version supports array API backends. |
| `scipy.integrate.trapezoid` | `sklearn.metrics.auc` | Use sklearn only if you need ROC-AUC. For arbitrary difficulty-curve AUC, scipy is cleaner (no sklearn dependency needed just for one function). |
| matplotlib (raw) for 3-panel figures | plotly | Never for this use case. Plotly produces interactive HTML; paper figures need static vector PDF/PNG. Plotly is already in `viewer` extras for dashboards -- do not mix into analysis pipeline. |
| matplotlib + SciencePlots | matplotlib + manual rcParams | Use manual rcParams if SciencePlots introduces unwanted styling. SciencePlots is a convenience, not a hard dependency. |
| seaborn `sns.heatmap` for error taxonomy matrix | matplotlib `imshow` + manual annotation | Use matplotlib `imshow` if you need precise control over cell formatting. Seaborn heatmap is faster to implement for standard cases. |
| pandas `<3.0` | pandas `3.0` | Use pandas 3.0 only after all deps (statsmodels, seaborn) confirm compatibility. As of 2026-02-14, pandas 3.0 is 3 weeks old -- too risky for a reproduction pipeline where correctness is paramount. |
| pandas DataFrame | Pure numpy arrays | Use pure numpy only for hot inner loops. DataFrames are better for labeled data (task_id, difficulty, pass/fail, model_name). |
| instructor for taxonomy classification | Raw Cohere API + JSON parsing | The project already has raw Cohere API calling in `cooperbench-eval/src/llm_judge/base.py`. Use instructor for NEW taxonomy classifiers (C1a-C4b) because structured output validation is critical for 6-class taxonomy. Keep existing LLM judge code as-is. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `numpy.trapz` | Deprecated since NumPy 2.0. Will be removed. | `scipy.integrate.trapezoid` or `numpy.trapezoid` |
| `scipy.integrate.simps` | Renamed to `scipy.integrate.simpson` in SciPy 1.12. Old name deprecated. | `scipy.integrate.trapezoid` (trapezoidal rule is standard for AUC; Simpson's adds no value for 10-point curves) |
| plotnine / ggplot2-style | Adds R-ggplot dependency chain. Niche in Python ML community. Harder to debug figure layout. | matplotlib + seaborn |
| bokeh | Interactive dashboards, not static figures. Wrong tool for paper figures. | matplotlib |
| scikit-learn (for AUC only) | Pulling in sklearn just for `sklearn.metrics.auc` is wasteful. sklearn is 200+ MB. | `scipy.integrate.trapezoid` (2 lines of code) |
| `pandas>=3.0` | Breaking changes with Copy-on-Write, string dtype inference, PyArrow requirement. Released 2026-01-21 -- ecosystem hasn't fully adapted. statsmodels 0.14.6 not tested against it. | `pandas>=2.2.0,<3.0` |
| Manual Wilson CI formula | Easy to get wrong at boundary cases (p=0, p=1, small n). No reason to reimplement when statsmodels provides it. | `statsmodels.stats.proportion.proportion_confint(method='wilson')` |
| LangChain for taxonomy classification | Massive dependency, framework lock-in, abstraction overhead. The project already uses instructor which is lighter and more appropriate for single-call structured output. | `instructor` (already in deps) |
| Custom confidence interval bootstrap | Slow (requires many resamples), unnecessary for binomial proportions where Wilson CI has closed-form solution. | Wilson CI via statsmodels |

## Stack Patterns by Use Case

**For Wilson Confidence Intervals (Figures 4, 5):**
```python
from statsmodels.stats.proportion import proportion_confint

# count = number of successes, nobs = total trials
lower, upper = proportion_confint(count=k, nobs=n, alpha=0.05, method='wilson')
```
- Use `method='wilson'` specifically. Do NOT use `method='normal'` (the default) -- normal approximation is poor for small n or extreme proportions.
- Returns (lower_bound, upper_bound) tuple.

**For AUC over Difficulty Curve (Figure 5):**
```python
from scipy.integrate import trapezoid

# x = difficulty bucket centers (0.05, 0.15, ..., 0.95 for 10 decile buckets)
# y = success rate per bucket
auc = trapezoid(y=success_rates, x=bucket_centers)
```
- Normalize x to [0, 1] range before computing AUC so results are comparable across configurations.

**For 10-Bucket Difficulty Stratification:**
```python
import pandas as pd
import numpy as np

# pd.qcut for equal-count decile buckets
df['difficulty_bucket'] = pd.qcut(df['difficulty_score'], q=10, labels=False)
# Or pd.cut for equal-width buckets (matches paper methodology)
df['difficulty_bucket'] = pd.cut(df['difficulty_score'], bins=10, labels=False)
```
- Check the paper: CooperBench uses equal-width bins (pd.cut), not quantile bins (pd.qcut). This matters for bucket sizes.

**For 3-Panel Figures (Figure 4/5/6):**
```python
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401 -- registers styles on import

plt.style.use(['science', 'no-latex'])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# Panel 1: overall pass rates with Wilson CIs
# Panel 2: difficulty-stratified curves with error bands
# Panel 3: failure mode taxonomy comparison
fig.savefig('figure.pdf', dpi=300, bbox_inches='tight')
```
- Use `no-latex` to avoid requiring a LaTeX installation.
- Save as PDF for vector graphics (journal submission), PNG at 300 DPI for preprint.

**For LLM Taxonomy Classification (C1a-C4b):**
```python
import instructor
from pydantic import BaseModel, Field
from cohere import Client

class CommunicationError(BaseModel):
    category: Literal["C1a", "C1b", "C2", "C3b", "C4a", "C4b", "none"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str

client = instructor.from_cohere(Client())
result = client.chat.completions.create(
    model="command-a-03-2025",
    response_model=CommunicationError,
    messages=[{"role": "user", "content": transcript_prompt}],
)
```
- Use instructor's Cohere integration since the project already uses the Cohere SDK.
- Pydantic validation ensures the LLM returns a valid category from the enum.

**For Qualitative Transcript Analysis (Plan:Question ratios, specificity):**
```python
import re

# Count plan statements vs questions in transcript
plan_count = len(re.findall(r'\b(plan|approach|strategy|will|going to)\b', transcript, re.I))
question_count = len(re.findall(r'\?', transcript))
ratio = plan_count / max(question_count, 1)

# Count specificity indicators
file_path_count = len(re.findall(r'[\w/]+\.\w{1,4}', transcript))
line_number_count = len(re.findall(r'line\s+\d+|L\d+', transcript, re.I))
```
- These are regex-based heuristics -- no additional libraries needed beyond stdlib `re`.
- For more sophisticated analysis, use the LLM taxonomy classifier above.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| statsmodels>=0.14.0 | numpy>=1.22,<3.0; scipy>=1.8; pandas>=1.4,<3.0 | statsmodels 0.14.6 is built against numpy 2.x. Confirmed compatible with scipy 1.14-1.17 and pandas 2.2. |
| matplotlib>=3.9.0 | numpy>=1.23 | No known issues with numpy 2.x. |
| seaborn>=0.13.0 | matplotlib>=3.4; pandas>=1.2; numpy>=1.20 | Seaborn 0.13.2 not yet tested against pandas 3.0. |
| SciencePlots>=2.2.0 | matplotlib>=3.4 | Pure style sheets, no compiled extensions. Low compatibility risk. |
| scipy>=1.14.0 | numpy>=1.23.5,<2.5 | SciPy 1.17 requires numpy<2.5. Our pin of numpy<3.0 is compatible. |
| instructor>=1.14.0 | pydantic>=2.0; cohere>=5.0 | Already resolved in existing lockfile. |

## Dependency Graph (New Analysis Stack)

```
analysis pipeline
  +-- numpy >=2.2,<3.0       (array ops)
  +-- pandas >=2.2,<3.0      (DataFrames)
  +-- scipy >=1.14            (trapezoid AUC)
  +-- statsmodels >=0.14      (Wilson CI)
  |     +-- numpy, scipy, pandas (transitive, compatible)
  +-- matplotlib >=3.9        (figures)
  +-- seaborn >=0.13          (themes, heatmaps)
  |     +-- matplotlib, pandas (transitive)
  +-- SciencePlots >=2.2      (publication styles)
  |     +-- matplotlib (transitive)
  +-- instructor >=1.14       (ALREADY in core deps -- reuse)
  +-- cohere >=5.20           (ALREADY in core deps -- reuse)
  +-- pydantic >=2.12         (ALREADY in core deps -- reuse)
```

## Key Decisions & Rationale

1. **Why `pandas<3.0`:** Pandas 3.0 (released 2026-01-21) introduced Copy-on-Write as default, PyArrow-backed strings, and NaN/NA unification. These are correctness-critical changes for statistical analysis. Statsmodels 0.14.6 and seaborn 0.13.2 have not been validated against pandas 3.0. For a paper reproduction where numerical correctness is paramount, we pin to the battle-tested 2.x series.

2. **Why statsmodels over hand-rolling Wilson CI:** The Wilson score interval formula is straightforward, but edge cases at p=0 and p=1 require special handling, and the continuity correction variant adds complexity. Statsmodels' implementation is well-tested across thousands of academic uses. One function call vs. potential bugs.

3. **Why NOT scikit-learn for AUC:** The project has no other need for scikit-learn. Adding a ~200MB dependency for `sklearn.metrics.auc` (which internally just calls `numpy.trapz` anyway) is wasteful. `scipy.integrate.trapezoid(y, x)` does the same thing in one line.

4. **Why instructor for taxonomy, not raw API:** The existing `cooperbench-eval/src/llm_judge/base.py` uses raw httpx calls to the Cohere API with manual JSON parsing. This works for binary classifiers. For the 6-class communication error taxonomy (C1a/C1b/C2/C3b/C4a/C4b), structured output validation via instructor + pydantic prevents silent misclassification (e.g., LLM returning "C3a" which isn't in the taxonomy).

5. **Why SciencePlots is optional:** It provides convenience (`plt.style.use(['science', 'no-latex'])`) but can be replicated with ~15 lines of `rcParams` configuration. Include it for developer ergonomics but don't make any code depend on it being present.

## Sources

- [statsmodels proportion_confint (0.14.6 stable)](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html) -- Verified Wilson method signature and return values. **HIGH confidence.**
- [scipy.integrate.trapezoid (1.17.0)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.trapezoid.html) -- Verified function name and API. **HIGH confidence.**
- [matplotlib 3.10.8 rcParams](https://matplotlib.org/stable/users/explain/customizing.html) -- Verified publication styling approach. **HIGH confidence.**
- [SciencePlots 2.2.0 (PyPI)](https://pypi.org/project/SciencePlots/) -- Verified latest version and release date. **MEDIUM confidence** (not verified against our specific matplotlib version).
- [pandas 3.0 What's New](https://pandas.pydata.org/docs/whatsnew/v3.0.0.html) -- Verified breaking changes and migration path. **HIGH confidence.**
- [instructor (PyPI)](https://python.useinstructor.com/) -- Verified Cohere integration and Pydantic model support. **HIGH confidence.**
- [numpy.trapezoid deprecation discussion](https://github.com/numpy/numpy/issues/25586) -- Verified `trapz` -> `trapezoid` rename. **HIGH confidence.**
- [sklearn.metrics.auc](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html) -- Verified it uses trapezoidal rule internally. **HIGH confidence.**

---
*Stack research for: CooperBench paper reproduction analysis pipeline*
*Researched: 2026-02-14*
