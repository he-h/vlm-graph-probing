#!/usr/bin/env bash
# Step 7: Cross-modality (visual-text) token correlation analysis.
#
# For each layer, computes the mean token-token correlation within
# vision tokens (VV), within text tokens (TT), and across modalities (VT).
# Saves results to results/modality_correlation/.
#
# Usage:
#   bash scripts/examples/06_modality_corr.sh

set -euo pipefail

MODEL="OpenGVLab/InternVL3-1B-hf"
DATASET="clevr"
CATEGORY="color"
NUM_SAMPLES=1000
DEVICE="cuda:0"

python -u -m probing.modality_corr \
    --dataset "$DATASET" \
    --category "$CATEGORY" \
    --num_samples "$NUM_SAMPLES" \
    --model_ckpt "$MODEL" \
    --device "$DEVICE"
