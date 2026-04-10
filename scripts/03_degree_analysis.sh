#!/usr/bin/env bash
# Step 3: Layer-wise node-degree and activation analysis across ALL layers.
#
# Computes per-neuron average degree (from sparse correlation graphs) and
# average |last-token activation| for every layer. Saves two .npy files
# of shape [num_layers, hidden_dim].
#
# Usage:
#   bash scripts/examples/02_get_degree.sh

set -euo pipefail

MODEL="OpenGVLab/InternVL3-1B-hf"
DATASET="clevr"
CATEGORY="color"
NUM_SAMPLES=1000
DEVICE="cuda:0"

python -u -m probing.degree_analysis \
    --dataset "$DATASET" \
    --category "$CATEGORY" \
    --num_samples "$NUM_SAMPLES" \
    --model_ckpt "$MODEL" \
    --output_root data \
    --device "$DEVICE" \
    --sparse_level 0.9
