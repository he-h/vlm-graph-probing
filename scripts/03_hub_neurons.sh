#!/usr/bin/env bash
# Step 3: Identify hub neurons by degree and activation frequency.
#
# For each selected layer, tracks the top-k neurons that appear most
# frequently as high-degree or high-activation neurons across samples.
# Saves results to results/hub_neurons/.
#
# Usage:
#   bash scripts/examples/03_analyze_hub_neurons.sh

set -euo pipefail

MODEL="OpenGVLab/InternVL3-1B-hf"
DATASET="clevr"
CATEGORY="color"
NUM_SAMPLES=1000
DEVICE="cuda:0"
TOP_K=10

python -u -m probing.hub_neurons \
    --dataset "$DATASET" \
    --category "$CATEGORY" \
    --num_samples "$NUM_SAMPLES" \
    --model_ckpt "$MODEL" \
    --device "$DEVICE" \
    --layer_slices 4 \
    --top_k "$TOP_K"
