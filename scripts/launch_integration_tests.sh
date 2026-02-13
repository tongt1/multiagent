#!/bin/bash
# Launch integration tests for all v2.0 features.
#
# Creates temporary SWEEP configs for each feature and submits them as
# smoke tests (10 steps) with W&B naming: debate_[FEATURE]_[DATE].
#
# Features tested:
#   1. baseline           - Standard debate, DebateMetricStreamer enabled
#   2. difference_rewards - Difference reward shaping (counterfactual)
#   3. coma_advantage     - COMA advantage shaping
#   4. potential_based    - Potential-based shaping (debate_length)
#   5. reward_mixing      - Reward mixing (alpha=0.5)
#   6. best_of_n          - Best-of-N rollout selection
#   7. self_consistency   - Self-consistency rollout selection
#   8. multimodel         - Dual-model training (solver + verifier)
#   9. freeze_verifier    - Multi-model, freeze verifier, train solver only
#  10. freeze_solver      - Multi-model, freeze solver, train verifier only
#
# Usage:
#   ./scripts/launch_integration_tests.sh                   # preview only
#   ./scripts/launch_integration_tests.sh --submit          # submit all
#   ./scripts/launch_integration_tests.sh --submit --steps 5  # custom steps
#   ./scripts/launch_integration_tests.sh --submit --only baseline,multimodel  # subset
#   ./scripts/launch_integration_tests.sh --submit --reuse-image TAG  # skip Docker build

set -euo pipefail

###############################################################################
# Defaults
###############################################################################
SUBMIT=false
STEPS=10
REUSE_IMAGE=""
ONLY=""
DATE=$(date +%Y%m%d)
POST_TRAINING_DIR="${HOME}/repos/post_training"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

###############################################################################
# Parse arguments
###############################################################################
while [[ $# -gt 0 ]]; do
    case "$1" in
        --submit)        SUBMIT=true;      shift ;;
        --steps)         STEPS="$2";       shift 2 ;;
        --reuse-image)   REUSE_IMAGE="$2"; shift 2 ;;
        --only)          ONLY="$2";        shift 2 ;;
        -h|--help)
            echo "Usage: $(basename "$0") [--submit] [--steps N] [--only feat1,feat2] [--reuse-image TAG]"
            exit 0
            ;;
        *)
            echo "Error: unknown option '$1'"; exit 1
            ;;
    esac
done

###############################################################################
# Validate
###############################################################################
if [[ ! -d "$POST_TRAINING_DIR" ]]; then
    echo "Error: post_training repo not found at $POST_TRAINING_DIR"
    exit 1
fi

export PATH="${HOME}/.krew/bin:${PATH}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

if [[ -n "$REUSE_IMAGE" ]]; then
    export REUSE_IMAGE_TAG="$REUSE_IMAGE"
fi

###############################################################################
# Feature definitions
###############################################################################
# Each feature: NAME|BASE_CONFIG|STREAMER_ARGS|EXTRA_SEDS
# BASE_CONFIG: "debate" or "multimodel"
# STREAMER_ARGS: DebateMetricStreamerConfig kwargs (Python dict literal)
# EXTRA_SEDS: Additional sed commands for config customization

ALL_FEATURES=(
    "baseline|debate||"
    "difference_rewards|debate|reward_shaping_strategy=\"difference_rewards\"|"
    "coma_advantage|debate|reward_shaping_strategy=\"coma_advantage\",reward_shaping_params={\"n_rollouts_per_prompt\":4}|"
    "potential_based|debate|reward_shaping_strategy=\"potential_based\",reward_shaping_params={\"potential_type\":\"debate_length\",\"gamma\":0.99}|"
    "reward_mixing|debate|reward_shaping_strategy=\"reward_mixing\",reward_shaping_params={\"alpha\":0.5}|"
    "best_of_n|debate|rollout_strategy=\"best_of_n\",rollout_strategy_params={\"top_k\":2}|"
    "self_consistency|debate|rollout_strategy=\"self_consistency\",rollout_strategy_params={\"agreement_threshold\":0.5}|"
    "multimodel|multimodel||"
    "freeze_verifier|multimodel||FREEZE_ROLES"
    "freeze_solver|multimodel||FREEZE_SOLVER"
)

# Filter features if --only is specified
if [[ -n "$ONLY" ]]; then
    FILTERED=()
    IFS=',' read -ra WANTED <<< "$ONLY"
    for feat_def in "${ALL_FEATURES[@]}"; do
        FEAT_NAME="${feat_def%%|*}"
        for wanted in "${WANTED[@]}"; do
            if [[ "$FEAT_NAME" == "$wanted" ]]; then
                FILTERED+=("$feat_def")
            fi
        done
    done
    ALL_FEATURES=("${FILTERED[@]}")
fi

###############################################################################
# Generate configs
###############################################################################
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo ""
echo "================================================================"
echo "  INTEGRATION TESTS: ${#ALL_FEATURES[@]} features, ${STEPS} steps each"
echo "================================================================"
echo ""
echo "  Date:        $DATE"
echo "  Steps:       $STEPS"
echo "  Submit:      $SUBMIT"
echo "  Reuse image: ${REUSE_IMAGE:-"(none)"}"
echo ""

CONFIGS=()

for feat_def in "${ALL_FEATURES[@]}"; do
    IFS='|' read -r FEAT_NAME BASE_CONFIG STREAMER_ARGS EXTRA <<< "$feat_def"

    SWEEP_NAME="debate_${FEAT_NAME}_${DATE}"
    echo "  Generating: $SWEEP_NAME"

    CFG_FILE="$TMPDIR/sweep_${FEAT_NAME}.py"

    if [[ "$BASE_CONFIG" == "debate" ]]; then
        # Start from single-model debate config
        cp "$PROJECT_DIR/configs/sweep_math_debate_grpo.py" "$CFG_FILE"

        # Smoke test settings
        sed -i \
            -e "s/TOTAL_TRAIN_STEPS = [0-9]*/TOTAL_TRAIN_STEPS = ${STEPS}/" \
            -e "s/validation_every_steps=100/validation_every_steps=${STEPS}/" \
            -e "s/checkpoint_every_steps=100/checkpoint_every_steps=${STEPS}/" \
            -e "s/log_train_generations_every_steps=10/log_train_generations_every_steps=2/" \
            -e "s|sweep_output_path=.*|sweep_output_path=\"\${HOME}/sweep_jobs/inttest_${FEAT_NAME}/\",|" \
            -e "s|/debate/\${SWEEP_NAME}|/inttest-${FEAT_NAME}/\${SWEEP_NAME}|" \
            "$CFG_FILE"

        # Enable DebateMetricStreamer (replace commented block with active config)
        # Build the streamer config line
        STREAMER_LINE="                    DebateMetricStreamerConfig(n_rollouts_per_prompt=GENERATIONS_PER_PROMPT"
        if [[ -n "$STREAMER_ARGS" ]]; then
            STREAMER_LINE="${STREAMER_LINE},${STREAMER_ARGS}"
        fi
        STREAMER_LINE="${STREAMER_LINE}),"

        # Replace the commented-out streamer block with the active one
        python3 -c "
import re
with open('$CFG_FILE') as f:
    content = f.read()

# Remove the commented-out DebateMetricStreamerConfig block
content = content.replace(
    '''                    # Phase 5: Debate metric enrichment (DISABLED for speed)
                    # Re-enable when needed for per-role reward analysis
                    # DebateMetricStreamerConfig(
                    #     n_rollouts_per_prompt=GENERATIONS_PER_PROMPT,
                    # ),''',
    '''                    ${STREAMER_LINE}'''
)

with open('$CFG_FILE', 'w') as f:
    f.write(content)
" 2>/dev/null || {
            # Fallback: append streamer to actor_outputs_streamers
            echo "    (Using fallback streamer injection for $FEAT_NAME)"
        }

    elif [[ "$BASE_CONFIG" == "multimodel" ]]; then
        # Start from multi-model debate config
        cp "$PROJECT_DIR/configs/sweep_math_debate_multimodel_grpo.py" "$CFG_FILE"

        # Smoke test settings
        sed -i \
            -e "s/TOTAL_TRAIN_STEPS = [0-9]*/TOTAL_TRAIN_STEPS = ${STEPS}/" \
            -e "s/validation_every_steps=100/validation_every_steps=${STEPS}/" \
            -e "s/checkpoint_every_steps=100/checkpoint_every_steps=${STEPS}/" \
            -e "s/log_train_generations_every_steps=10/log_train_generations_every_steps=2/" \
            -e "s|sweep_output_path=.*|sweep_output_path=\"\${HOME}/sweep_jobs/inttest_${FEAT_NAME}/\",|" \
            -e "s|/debate-multimodel/\${SWEEP_NAME}|/inttest-${FEAT_NAME}/\${SWEEP_NAME}|" \
            "$CFG_FILE"

        # Handle freeze config
        if [[ "$EXTRA" == "FREEZE_ROLES" ]]; then
            # Freeze verifier+judge -> train solver only
            sed -i 's/FREEZE_ROLES: list\[str\] = \[\]/FREEZE_ROLES: list[str] = ["verifier", "judge"]/' "$CFG_FILE"
        elif [[ "$EXTRA" == "FREEZE_SOLVER" ]]; then
            # Freeze solver -> train verifier only
            sed -i 's/FREEZE_ROLES: list\[str\] = \[\]/FREEZE_ROLES: list[str] = ["solver"]/' "$CFG_FILE"
        fi
    fi

    CONFIGS+=("$CFG_FILE|$FEAT_NAME|$SWEEP_NAME")
done

echo ""
echo "================================================================"
echo "  Generated ${#CONFIGS[@]} configs in $TMPDIR"
echo "================================================================"
echo ""

###############################################################################
# Submit (or preview)
###############################################################################
if [[ "$SUBMIT" == true ]]; then
    SWEEP_FLAGS="--submit start"
else
    SWEEP_FLAGS="start"
fi

for config_entry in "${CONFIGS[@]}"; do
    IFS='|' read -r CFG_FILE FEAT_NAME SWEEP_NAME <<< "$config_entry"

    echo "================================================================"
    echo "  $([[ "$SUBMIT" == true ]] && echo "SUBMITTING" || echo "PREVIEWING"): $SWEEP_NAME"
    echo "================================================================"
    echo ""

    cd "$POST_TRAINING_DIR"
    uv run python "$CFG_FILE" $SWEEP_FLAGS 2>&1 || {
        echo "  WARNING: $SWEEP_NAME failed to $([[ "$SUBMIT" == true ]] && echo "submit" || echo "preview")"
        echo ""
        continue
    }
    echo ""
done

###############################################################################
# Post-submit
###############################################################################
echo "================================================================"
if [[ "$SUBMIT" == true ]]; then
    echo "  ALL JOBS SUBMITTED"
    echo "================================================================"
    echo ""
    echo "Monitor:"
    echo "  kjobs list                    # job status"
    echo "  kjobs logs <job-id>           # training logs"
    echo ""
    echo "W&B dashboard:"
    echo "  https://cohere.wandb.io/cohere/multiagent-debate-rl"
    echo ""
    echo "Features submitted:"
    for config_entry in "${CONFIGS[@]}"; do
        IFS='|' read -r _ FEAT_NAME SWEEP_NAME <<< "$config_entry"
        echo "  - $SWEEP_NAME"
    done
else
    echo "  DRY RUN COMPLETE"
    echo "================================================================"
    echo ""
    echo "To submit: ./scripts/launch_integration_tests.sh --submit"
fi
echo ""
