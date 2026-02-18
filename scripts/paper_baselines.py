#!/usr/bin/env python3
"""Paper baseline numbers from CooperBench (arXiv:2601.13295).

All values extracted from the paper's tables and text for overlay comparison
on our reproduction figures (COMP-01 requirement).

IMPORTANT -- Figure Number Mapping:
    The paper's figure numbering does NOT match our project's figure numbering.

    | Our Fig # | Our Content                        | Paper Fig # | Paper Content                    |
    |-----------|------------------------------------|-------------|----------------------------------|
    | Figure 4  | Difficulty-stratified success curves | Figure 6  | Difficulty curves + Wilson CIs   |
    | Figure 5  | Communication effects (3-panel)    | Figure 4    | Communication effects (3-panel)  |
    | Figure 6  | Communication error taxonomy       | Figure 5    | Communication error detection    |

    When looking up paper data for our Figure N, use the TOPIC mapping above,
    not the figure number.

Data sources:
    fig4 baselines: Paper Table 5 (per-model AUC and retention values)
    fig5 baselines: Paper Section 5.2 text ("~20% overhead", "~1/3 each", conflict rates)
    fig6 baselines: Paper Figure 5 caption and text (3 high-level categories)
"""

PAPER_BASELINES = {
    # ---------------------------------------------------------------
    # Figure 4: Difficulty-stratified success curves
    # Source: Paper Table 5 (AUC and retention per model)
    # Note: Our Fig4 = Paper's Fig6 content
    # ---------------------------------------------------------------
    "fig4": {
        "models": {
            "GPT-5":       {"solo_auc": 0.506, "coop_auc": 0.325, "retention": 0.64},
            "Claude 4.5":  {"solo_auc": 0.469, "coop_auc": 0.283, "retention": 0.60},
            "MiniMax-M2":  {"solo_auc": 0.374, "coop_auc": 0.171, "retention": 0.46},
            "Qwen-Coder":  {"solo_auc": 0.236, "coop_auc": 0.148, "retention": 0.63},
            "Qwen":        {"solo_auc": 0.106, "coop_auc": 0.072, "retention": 0.68},
        },
        "pooled": {
            "solo_auc": 0.338,
            "coop_auc": 0.200,
            "delta_auc": 0.138,
            "retention": 0.59,
        },
    },

    # ---------------------------------------------------------------
    # Figure 5: Communication effects
    # Source: Paper Section 5.2 text
    # Note: Our Fig5 = Paper's Fig4 content
    # ---------------------------------------------------------------
    "fig5": {
        "comm_overhead_pct": 20.0,  # "as much as 20% of the steps"
        "speech_acts": {
            "plan": 33.3,       # "each almost takes up 1/3"
            "question": 33.3,
            "update": 33.3,
        },
        "first_turn_planning": {
            "conflict_with": 29.4,     # conflict rate with first-turn planning
            "conflict_without": 51.5,  # conflict rate without first-turn planning
        },
    },

    # ---------------------------------------------------------------
    # Figure 6: Communication error taxonomy
    # Source: Paper Figure 5 description
    # Note: Our Fig6 = Paper's Fig5 content
    # ---------------------------------------------------------------
    "fig6": {
        # Paper uses 3 high-level categories, NOT our 6-category C1a-C4b taxonomy
        "paper_categories": ["Repetition", "Unresponsiveness", "Hallucination"],
        # Mapping from paper categories to our taxonomy:
        #   Repetition      = C4a (Spammy - Same Info) + C4b (Spammy - Near-duplicate)
        #   Unresponsiveness = C1a (Unanswered - No Reply) + C1b (Unanswered - Ignored)
        #   Hallucination    = C2 (Non-answer/Vague) + C3b (Incorrect Claim)
        "category_mapping": {
            "Repetition": ["C4a", "C4b"],
            "Unresponsiveness": ["C1a", "C1b"],
            "Hallucination": ["C2", "C3b"],
        },
        # Note: Exact percentages for the 3 paper categories are NOT available
        # from the paper text. The paper's Figure 5 shows a bar chart but does
        # not report numeric values. Do NOT fabricate numbers.
    },

    # ---------------------------------------------------------------
    # Table 1: Failure symptoms (for reference, not directly plotted)
    # ---------------------------------------------------------------
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
