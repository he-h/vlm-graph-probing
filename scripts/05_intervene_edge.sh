#!/usr/bin/env bash
# Step 5: Edge-level (neuron-pair) intervention.
#
# Two-pass experiment:
#   Pass 1 -- accumulate edge importance (sum of |correlation|) across samples.
#   Pass 2 -- intervene on top-k edges by forcing correlations:
#             identical (corr=1), opposite (corr=-1), random (corr~0).
#
# Usage:
#   bash scripts/examples/05_intervention_edge.sh

set -euo pipefail

MODEL="OpenGVLab/InternVL3-1B-hf"
DATASET="clevr"
CATEGORY="color"
NUM_SAMPLES=1000
DEVICE="cuda:0"
TOP_K_EDGES=10

python -u -m probing.intervene_edge \
    --dataset "$DATASET" \
    --category "$CATEGORY" \
    --num_samples "$NUM_SAMPLES" \
    --model_ckpt "$MODEL" \
    --device "$DEVICE" \
    --top_k_edges "$TOP_K_EDGES" \
    --layer_slices 1
