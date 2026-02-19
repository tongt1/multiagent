#!/usr/bin/env bash
# setup_cooperbench.sh -- Install cooperbench CLI in a Python 3.12 virtualenv
# and configure environment variables for Command A on staging endpoint.
#
# Usage:
#   bash scripts/setup_cooperbench.sh
#
# After running, source the environment exports printed at the end,
# or set COHERE_API_KEY manually.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COOPERBENCH_DIR="$PROJECT_ROOT/repos/CooperBench"

echo "=== CooperBench CLI Setup ==="
echo "Project root: $PROJECT_ROOT"
echo "CooperBench dir: $COOPERBENCH_DIR"

# --- Step 1: Verify CooperBench repo exists ---
if [ ! -d "$COOPERBENCH_DIR" ]; then
    echo "ERROR: CooperBench repo not found at $COOPERBENCH_DIR"
    echo "Clone it first: git clone <url> $COOPERBENCH_DIR"
    exit 1
fi

# --- Step 2: Create Python 3.12 virtualenv ---
echo ""
echo "--- Creating Python 3.12 virtualenv ---"
cd "$COOPERBENCH_DIR"

if [ -d ".venv" ]; then
    echo "Virtualenv already exists at $COOPERBENCH_DIR/.venv"
    EXISTING_VERSION=$(.venv/bin/python --version 2>&1 || echo "unknown")
    echo "Existing Python version: $EXISTING_VERSION"
else
    echo "Creating new virtualenv with Python 3.12..."
    uv venv --python 3.12 .venv
    echo "Virtualenv created at $COOPERBENCH_DIR/.venv"
fi

# Verify Python version
PYTHON_VERSION=$(.venv/bin/python --version 2>&1)
echo "Python version: $PYTHON_VERSION"

# Check Python >= 3.12
PYTHON_MINOR=$(.venv/bin/python -c "import sys; print(sys.version_info.minor)")
if [ "$PYTHON_MINOR" -lt 12 ]; then
    echo "ERROR: Python 3.12+ required, got $PYTHON_VERSION"
    exit 1
fi

# --- Step 3: Install cooperbench in editable mode ---
echo ""
echo "--- Installing cooperbench (editable mode) ---"
.venv/bin/python -m pip install --upgrade pip 2>/dev/null || true
uv pip install -e ".[dev]" --python .venv/bin/python
echo "cooperbench installed."

# --- Step 4: Verify installation ---
echo ""
echo "--- Verifying cooperbench CLI ---"
if .venv/bin/cooperbench --help > /dev/null 2>&1; then
    echo "OK: cooperbench --help succeeds"
else
    echo "ERROR: cooperbench --help failed"
    exit 1
fi

# --- Step 5: Print environment variable exports ---
echo ""
echo "=== Environment Configuration ==="
echo ""
echo "Add these to your shell or .env file:"
echo ""
echo "  # Required: Cohere API key (set your actual key)"
echo "  export COHERE_API_KEY=\"your-cohere-api-key-here\""
echo ""
echo "  # Staging endpoint for Command A"
echo "  export MSWEA_MODEL_API_BASE=\"https://stg.api.cohere.com/v2\""
echo ""
echo "  # Suppress cost tracking warnings"
echo "  export MSWEA_COST_TRACKING=\"ignore_errors\""
echo ""
echo "  # Suppress litellm debug noise"
echo "  export LITELLM_LOG=\"ERROR\""
echo ""
echo "=== Setup Complete ==="
echo "CLI location: $COOPERBENCH_DIR/.venv/bin/cooperbench"
echo "Python: $PYTHON_VERSION"
echo ""
echo "To activate the virtualenv:"
echo "  source $COOPERBENCH_DIR/.venv/bin/activate"
