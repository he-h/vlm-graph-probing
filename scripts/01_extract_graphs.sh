#!/usr/bin/env bash
# Step 1: Extract hidden states and build sparse correlation graphs.
#
# This produces per-layer graph artifacts (edge lists, last-token activations)
# stored under the output directory.
#
# Usage:
#   bash scripts/examples/01_prepare_data.sh
#
# Customize MODEL, DATASET, CATEGORY, NUM_SAMPLES, DEVICE below.

set -euo pipefail

MODEL="OpenGVLab/InternVL3-1B-hf"   # smallest supported model
DATASET="clevr"                       # clevr | tdiuc | coco
CATEGORY="color"                      # color | counting | existence | comparison | shape
NUM_SAMPLES=1000
DEVICE="cuda:0"
SPARSE_LEVEL=0.9
LAYER_SLICES=4                        # K slices -> K+1 evenly spaced layers
OUTPUT_ROOT="data"

python -u -m probing.extract_graphs \
    --dataset "$DATASET" \
    --category "$CATEGORY" \
    --num_samples "$NUM_SAMPLES" \
    --model_ckpt "$MODEL" \
    --output_root "$OUTPUT_ROOT" \
    --device "$DEVICE" \
    --sparse_level "$SPARSE_LEVEL" \
    --layer_slices "$LAYER_SLICES" \
    --verbose
