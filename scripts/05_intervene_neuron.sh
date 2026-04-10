#!/usr/bin/env bash
# Step 5: Neuron-level intervention (ablation / scaling).
#
# Requires hub neuron JSON from step 4. For each selected layer, scales
# the top-N hub neurons and measures the accuracy change vs. baseline.
#
# Usage:
#   bash scripts/examples/04_intervention_hub_neuron.sh
#
# Adjust CRITERION to test different neuron selection strategies:
#   full_degree  -- neurons with highest graph degree (full hidden state)
#   text_degree  -- neurons with highest graph degree (text tokens only)
#   last_token   -- neurons with highest |last-token activation|
#   random       -- random baseline

set -euo pipefail

MODEL="OpenGVLab/InternVL3-1B-hf"
DATASET="clevr"
CATEGORY="color"
NUM_SAMPLES=1000
DEVICE="cuda:0"

HUB_JSON="results/hub_neurons/hub_neurons_InternVL3-1B_clevr_color_top10.json"
CRITERION="full_degree"
NUM_INTERVENE=10
SCALE=0.0                # 0.0 = ablate, 0.5 = suppress, 1.0 = no-op, 2.0 = amplify

python -u -m probing.intervene_neuron \
    --dataset "$DATASET" \
    --category "$CATEGORY" \
    --num_samples "$NUM_SAMPLES" \
    --model_ckpt "$MODEL" \
    --device "$DEVICE" \
    --hub_neuron_json "$HUB_JSON" \
    --criterion "$CRITERION" \
    --num_intervene "$NUM_INTERVENE" \
    --scale "$SCALE" \
    --layer_slices 4
