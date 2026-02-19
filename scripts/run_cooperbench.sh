#!/usr/bin/env bash
# run_cooperbench.sh -- Orchestrator for CooperBench benchmark runs
#
# Runs cooperbench across all 3 experimental settings (solo, coop-comm,
# coop-nocomm) with retry logic, infra_error tagging, and cost reporting.
#
# Usage:
#   bash scripts/run_cooperbench.sh                         # Run all 3 settings
#   bash scripts/run_cooperbench.sh --setting solo           # Run solo only
#   bash scripts/run_cooperbench.sh --smoke 5 --setting solo # Smoke test: 5 pairs, solo
#   bash scripts/run_cooperbench.sh --dry-run                # Print commands without running

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COOPERBENCH_DIR="$PROJECT_DIR/repos/CooperBench"
VENV_ACTIVATE="$COOPERBENCH_DIR/.venv/bin/activate"

# Benchmark parameters
AGENT="mini_swe_agent"
MODEL="command-a-03-2025"
SUBSET="lite"
BACKEND="docker"
CONCURRENCY=4
MAX_ATTEMPTS=3

# Run names
SOLO_NAME="command-a-solo"
COOP_COMM_NAME="command-a-coop-comm"
COOP_NOCOMM_NAME="command-a-coop-nocomm"

# Environment variables (set defaults if not already exported)
export MSWEA_MODEL_API_BASE="${MSWEA_MODEL_API_BASE:-https://stg.api.cohere.com/v2/chat}"
export MSWEA_COST_TRACKING="${MSWEA_COST_TRACKING:-ignore_errors}"
export LITELLM_LOG="${LITELLM_LOG:-ERROR}"

# ============================================================================
# Argument parsing
# ============================================================================

SETTING="all"
SMOKE_PAIRS=0
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --setting)
            SETTING="$2"
            shift 2
            ;;
        --smoke)
            SMOKE_PAIRS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --setting SETTING   Which setting to run: solo, coop-comm, coop-nocomm, all (default: all)"
            echo "  --smoke N           Smoke test: limit to first N pairs (creates temp subset)"
            echo "  --dry-run           Print commands without executing"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

# Validate setting
case "$SETTING" in
    solo|coop-comm|coop-nocomm|all) ;;
    *)
        echo "ERROR: Invalid setting '$SETTING'. Must be: solo, coop-comm, coop-nocomm, all"
        exit 1
        ;;
esac

# ============================================================================
# Pre-flight checks
# ============================================================================

preflight_check() {
    echo "=== Pre-flight checks ==="

    # Check cooperbench venv exists
    if [[ ! -f "$VENV_ACTIVATE" ]]; then
        echo "ERROR: Virtualenv not found at $VENV_ACTIVATE"
        echo "Run: bash scripts/setup_cooperbench.sh"
        exit 1
    fi

    # Check COHERE_API_KEY
    if [[ -z "${COHERE_API_KEY:-}" ]]; then
        echo "ERROR: COHERE_API_KEY is not set"
        echo "Run: export COHERE_API_KEY='your-key-here'"
        exit 1
    fi

    # Check Docker is running
    if ! docker info >/dev/null 2>&1; then
        echo "ERROR: Docker is not running"
        exit 1
    fi

    echo "  Virtualenv: OK"
    echo "  COHERE_API_KEY: set"
    echo "  Docker: running"
    echo "  MSWEA_MODEL_API_BASE: $MSWEA_MODEL_API_BASE"
    echo ""
}

# ============================================================================
# Smoke subset creation
# ============================================================================

TEMP_SUBSET_FILE=""
EFFECTIVE_SUBSET="$SUBSET"

create_smoke_subset() {
    local num_pairs="$1"
    echo "=== Creating smoke subset with $num_pairs pairs ==="

    TEMP_SUBSET_FILE=$(mktemp /tmp/cooperbench-smoke-XXXXXX.json)

    # Use Python from the venv to create the subset
    (
        cd "$COOPERBENCH_DIR"
        source "$VENV_ACTIVATE"
        python -c "
import json, sys

num_pairs = int(sys.argv[1])
output_file = sys.argv[2]

with open('dataset/subsets/lite.json') as f:
    data = json.load(f)

# Take pairs from tasks until we have enough
smoke_tasks = []
pairs_collected = 0

for task in data['tasks']:
    if pairs_collected >= num_pairs:
        break
    remaining = num_pairs - pairs_collected
    pairs_to_take = task['pairs'][:remaining]
    if pairs_to_take:
        smoke_tasks.append({
            'repo': task['repo'],
            'task_id': task['task_id'],
            'pairs': pairs_to_take
        })
        pairs_collected += len(pairs_to_take)

smoke_data = {
    'name': 'smoke',
    'description': f'Smoke test subset with {pairs_collected} pairs',
    'stats': {
        'tasks': len(smoke_tasks),
        'pairs': pairs_collected,
        'repos': len(set(t['repo'] for t in smoke_tasks))
    },
    'tasks': smoke_tasks
}

with open(output_file, 'w') as f:
    json.dump(smoke_data, f, indent=2)

print(f'  Created smoke subset: {pairs_collected} pairs from {len(smoke_tasks)} tasks')
print(f'  Written to: {output_file}')
" "$num_pairs" "$TEMP_SUBSET_FILE"
    )

    # Copy the temp file into the subsets directory so cooperbench can find it
    cp "$TEMP_SUBSET_FILE" "$COOPERBENCH_DIR/dataset/subsets/smoke.json"
    EFFECTIVE_SUBSET="smoke"
    echo ""
}

cleanup_smoke_subset() {
    if [[ -n "$TEMP_SUBSET_FILE" && -f "$TEMP_SUBSET_FILE" ]]; then
        rm -f "$TEMP_SUBSET_FILE"
    fi
    if [[ -f "$COOPERBENCH_DIR/dataset/subsets/smoke.json" ]]; then
        rm -f "$COOPERBENCH_DIR/dataset/subsets/smoke.json"
    fi
}

# Always clean up temp files
trap cleanup_smoke_subset EXIT

# ============================================================================
# Run function with retry logic
# ============================================================================

run_setting() {
    local run_name="$1"
    local setting_args="$2"
    local attempt=1

    echo "=== Running: $run_name ==="
    echo "  Setting args: $setting_args"
    echo "  Subset: $EFFECTIVE_SUBSET"
    echo "  Max attempts: $MAX_ATTEMPTS"
    echo ""

    while [[ $attempt -le $MAX_ATTEMPTS ]]; do
        local timestamp
        timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        echo "--- Attempt $attempt/$MAX_ATTEMPTS (started: $timestamp) ---"

        local cmd="cooperbench run -n $run_name -m $MODEL -a $AGENT -s $EFFECTIVE_SUBSET --backend $BACKEND -c $CONCURRENCY $setting_args --no-auto-eval"

        if [[ "$DRY_RUN" == "true" ]]; then
            echo "[DRY RUN] cd $COOPERBENCH_DIR && source $VENV_ACTIVATE && $cmd"
            return 0
        fi

        # Run cooperbench (allow failure -- we handle retries)
        local exit_code=0
        (
            cd "$COOPERBENCH_DIR"
            source "$VENV_ACTIVATE"
            eval "$cmd"
        ) || exit_code=$?

        if [[ $exit_code -ne 0 ]]; then
            echo "WARNING: cooperbench exited with code $exit_code"
        fi

        # Count completed vs error results
        local log_dir="$COOPERBENCH_DIR/logs/$run_name"
        if [[ -d "$log_dir" ]]; then
            local total_results error_results completed_results
            total_results=$(find "$log_dir" -name "result.json" | wc -l)
            error_results=$(find "$log_dir" -name "result.json" -exec grep -l '"status": "Error"' {} \; 2>/dev/null | wc -l)
            completed_results=$((total_results - error_results))

            echo "  Results: $completed_results completed, $error_results errors, $total_results total"

            if [[ $error_results -eq 0 ]]; then
                echo "  All tasks completed successfully. No retry needed."
                break
            fi
        else
            echo "  WARNING: No log directory found at $log_dir"
        fi

        if [[ $attempt -lt $MAX_ATTEMPTS ]]; then
            echo "  Retrying (Error tasks will be re-run, completed tasks will be skipped)..."
        else
            echo "  Max attempts reached. Some tasks may still have errors."
        fi

        attempt=$((attempt + 1))
    done

    echo ""
}

# ============================================================================
# Post-processing: infra_error tagging
# ============================================================================

tag_infra_errors() {
    local run_name="$1"
    local log_dir="$COOPERBENCH_DIR/logs/$run_name"

    if [[ ! -d "$log_dir" ]]; then
        return
    fi

    echo "--- Tagging infra errors for $run_name ---"

    (
        cd "$COOPERBENCH_DIR"
        source "$VENV_ACTIVATE"
        python -c "
import json, re, sys
from pathlib import Path

log_dir = Path(sys.argv[1])
run_name = sys.argv[2]

infra_patterns = [
    # Docker patterns
    r'OOM', r'out of memory', r'killed', r'container',
    r'docker', r'timeout', r'timed out',
    # API patterns
    r'429', r'rate.?limit', r'connection', r'502', r'503',
    r'504', r'service.?unavailable', r'too many requests',
    r'connection.?refused', r'connection.?reset',
]
pattern = re.compile('|'.join(infra_patterns), re.IGNORECASE)

pass_count = 0
fail_count = 0
infra_count = 0
total = 0

for result_file in sorted(log_dir.rglob('result.json')):
    total += 1
    with open(result_file) as f:
        data = json.load(f)

    agent_status = data.get('agent', {}).get('status', '')
    agent_error = data.get('agent', {}).get('error', '') or ''

    if agent_status == 'Error':
        if pattern.search(agent_error):
            data['infra_error'] = True
            with open(result_file, 'w') as f:
                json.dump(data, f, indent=2)
            infra_count += 1
        else:
            fail_count += 1
    else:
        pass_count += 1

print(f'  {run_name}: {pass_count} pass, {fail_count} fail, {infra_count} infra_error (total: {total})')
" "$log_dir" "$run_name"
    )
}

# ============================================================================
# Cost summary
# ============================================================================

print_cost_summary() {
    echo "=== Cost Summary ==="

    (
        cd "$COOPERBENCH_DIR"
        source "$VENV_ACTIVATE"
        python -c "
import json, sys
from pathlib import Path

logs_dir = Path('logs')
if not logs_dir.exists():
    print('  No logs directory found.')
    sys.exit(0)

run_names = sys.argv[1:]

grand_total = 0.0
for run_name in run_names:
    log_dir = logs_dir / run_name
    if not log_dir.exists():
        continue

    run_cost = 0.0
    task_count = 0
    for result_file in log_dir.rglob('result.json'):
        with open(result_file) as f:
            data = json.load(f)
        # Check both cost locations
        cost = data.get('total_cost', 0) or data.get('agent', {}).get('cost', 0) or 0
        run_cost += cost
        task_count += 1

    grand_total += run_cost
    print(f'  {run_name}: \${run_cost:.4f} ({task_count} tasks)')

print(f'  --------')
print(f'  Total: \${grand_total:.4f}')
" "$SOLO_NAME" "$COOP_COMM_NAME" "$COOP_NOCOMM_NAME"
    )
    echo ""
}

# ============================================================================
# Main execution
# ============================================================================

main() {
    preflight_check

    # Create smoke subset if requested
    if [[ $SMOKE_PAIRS -gt 0 ]]; then
        create_smoke_subset "$SMOKE_PAIRS"
    fi

    local start_time
    start_time=$(date +%s)

    # Run the requested settings
    case "$SETTING" in
        solo)
            run_setting "$SOLO_NAME" "--setting solo"
            ;;
        coop-comm)
            run_setting "$COOP_COMM_NAME" "--setting coop"
            ;;
        coop-nocomm)
            run_setting "$COOP_NOCOMM_NAME" "--setting coop --no-messaging"
            ;;
        all)
            run_setting "$SOLO_NAME" "--setting solo"
            run_setting "$COOP_COMM_NAME" "--setting coop"
            run_setting "$COOP_NOCOMM_NAME" "--setting coop --no-messaging"
            ;;
    esac

    # Post-processing (skip for dry-run)
    if [[ "$DRY_RUN" != "true" ]]; then
        echo "=== Post-processing ==="

        # Tag infra errors
        case "$SETTING" in
            solo)       tag_infra_errors "$SOLO_NAME" ;;
            coop-comm)  tag_infra_errors "$COOP_COMM_NAME" ;;
            coop-nocomm) tag_infra_errors "$COOP_NOCOMM_NAME" ;;
            all)
                tag_infra_errors "$SOLO_NAME"
                tag_infra_errors "$COOP_COMM_NAME"
                tag_infra_errors "$COOP_NOCOMM_NAME"
                ;;
        esac
        echo ""

        # Print cost summary
        print_cost_summary
    fi

    local end_time elapsed_mins
    end_time=$(date +%s)
    elapsed_mins=$(( (end_time - start_time) / 60 ))
    echo "=== Done (${elapsed_mins} minutes) ==="
}

main
