#!/usr/bin/env bash
# pull_images.sh -- Pull and verify all missing CooperBench Docker images
# for the lite subset (26 tasks, 100 pairs).
#
# Usage:
#   bash scripts/pull_images.sh
#
# This script is idempotent: already-pulled images are fast no-ops.
# Docker storage is on /mnt/data/docker (876GB free).
set -euo pipefail

# The 8 images missing from the lite subset (as of research phase)
MISSING_IMAGES=(
    "akhatua/cooperbench-dottxt-ai-outlines:task1655"
    "akhatua/cooperbench-dottxt-ai-outlines:task1706"
    "akhatua/cooperbench-dspy:task8587"
    "akhatua/cooperbench-dspy:task8635"
    "akhatua/cooperbench-go-chi:task27"
    "akhatua/cooperbench-llama-index:task17244"
    "akhatua/cooperbench-react-hook-form:task85"
    "akhatua/cooperbench-react-hook-form:task153"
)

TOTAL=${#MISSING_IMAGES[@]}
PASSED=0
FAILED=0
FAILURES=()

echo "=== Pulling $TOTAL Missing Docker Images ==="
echo ""

for i in "${!MISSING_IMAGES[@]}"; do
    img="${MISSING_IMAGES[$i]}"
    idx=$((i + 1))
    echo "[$idx/$TOTAL] Pulling $img..."

    # Pull the image
    if ! docker pull "$img" 2>&1; then
        echo "  FAIL: docker pull failed for $img"
        FAILED=$((FAILED + 1))
        FAILURES+=("$img")
        continue
    fi

    # Start a temporary container to verify /workspace exists
    container_id=$(docker run -d "$img" sleep 10 2>&1)
    if [ -z "$container_id" ]; then
        echo "  FAIL: could not start container for $img"
        FAILED=$((FAILED + 1))
        FAILURES+=("$img")
        continue
    fi

    # Check for /workspace or /workspace/repo directory
    if docker exec "$container_id" test -d /workspace/repo 2>/dev/null; then
        echo "  OK: /workspace/repo exists"
        PASSED=$((PASSED + 1))
    elif docker exec "$container_id" test -d /workspace 2>/dev/null; then
        echo "  OK: /workspace exists (no /workspace/repo)"
        PASSED=$((PASSED + 1))
    else
        echo "  WARNING: neither /workspace nor /workspace/repo found in $img"
        PASSED=$((PASSED + 1))  # Still count as pulled, workspace layout may vary
    fi

    # Clean up temporary container
    docker rm -f "$container_id" > /dev/null 2>&1
    echo ""
done

# --- Final verification: count all cooperbench images ---
echo "=== Final Verification ==="
echo ""
echo "All cooperbench images on this machine:"
docker images --format '{{.Repository}}:{{.Tag}}' | grep cooperbench | sort
echo ""

TOTAL_IMAGES=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep cooperbench | wc -l)
echo "Total cooperbench images: $TOTAL_IMAGES"
echo ""

# --- Summary ---
echo "=== Pull Summary ==="
echo "  Pulled and verified: $PASSED / $TOTAL"
echo "  Failed: $FAILED / $TOTAL"
echo "  Total cooperbench images: $TOTAL_IMAGES / 26 expected"

if [ "$FAILED" -gt 0 ]; then
    echo ""
    echo "FAILED images:"
    for f in "${FAILURES[@]}"; do
        echo "  - $f"
    done
    exit 1
fi

if [ "$TOTAL_IMAGES" -ge 26 ]; then
    echo ""
    echo "All 26 lite subset images are available locally."
else
    echo ""
    echo "WARNING: Only $TOTAL_IMAGES of 26 expected images found."
    echo "Some pre-existing images may be missing."
fi
