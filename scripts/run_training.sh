#!/bin/bash
# MARTI Training Job Submission
# Usage: bash scripts/run_training.sh

set -e

echo "========================================="
echo "MARTI Training Job Submission"
echo "========================================="

cd "$(dirname "$0")/.."

# Pull latest code
echo "Pulling latest code from GitHub..."
git pull origin main

# Check kubectl context
echo ""
echo "Current kubectl context:"
kubectl config current-context 2>/dev/null || echo "No context set"

# Check if kjobs auth works
echo ""
echo "Testing cluster connection..."
if ! timeout 10 kubectl get ns --no-headers 2>/dev/null | head -3; then
    echo ""
    echo "ERROR: Cluster authentication failed."
    echo "Run this to refresh OIDC login:"
    echo "  kubectl config use-context cw-ca-east-01-prod"
    echo "  kubectl get ns"
    echo "  (This will open a browser for OIDC login)"
    echo ""
    echo "After auth succeeds, run this script again."
    exit 1
fi

# Submit the job
echo ""
echo "Submitting kjobs training job..."
kjobs submit -f kjobs-compute.yaml

echo ""
echo "========================================="
echo "Job submitted! Check status with:"
echo "  kjobs list"
echo "  kjobs logs <job-name>"
echo ""
echo "W&B dashboard: https://cohere.wandb.io/marti-training"
echo "========================================="
