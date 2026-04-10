#!/usr/bin/env bash
# Step 2: Train and evaluate GCNPredictor on pre-extracted correlation graphs.
#
# Reads artifacts produced by scripts/01_extract_graphs.sh and runs a
# binary classification experiment: predicting whether the VLM answered
# each VQA question correctly from its neuron-correlation graph.
#
# Usage:
#   bash scripts/07_predict_gcn.sh
#
# Customize MODEL, DATASET, CATEGORY, LAYER, DEVICE, and training
# hyperparameters below. DATA_DIR must point to the output directory
# created by 01_extract_graphs.sh.

set -euo pipefail

MODEL="OpenGVLab/InternVL3-1B-hf"
DATASET="clevr"
CATEGORY="color"
SPARSE_LEVEL=0.9
SPARSITY_PCT=90     # integer version of SPARSE_LEVEL * 100, used in dir name

# Output directory produced by 01_extract_graphs.sh
MODEL_NAME="InternVL3-1B"   # must match model_ckpt2name() in utils.py
DATA_DIR="data/${MODEL_NAME}_${DATASET}_${CATEGORY}_sparsity_${SPARSITY_PCT}_probing_dataset"

LAYER=-1            # -1 = auto-select first available layer
GRAPH_TYPE="full"   # full | text
DEVICE="cuda:0"

# --- training ---
EPOCHS=50
LR=1e-3
BATCH_SIZE=32
WEIGHT_DECAY=0.0
SEED=42
TEST_SIZE=0.2

# --- model architecture ---
EMBEDDING_DIM=64
HIDDEN_DIM=128
FC_HIDDEN_DIM=128
NUM_GCN_LAYERS=2
DROPOUT=0.0
ACTIVATION="relu"

OUTPUT_DIR="results/gcn_predictor"

python -u -m probing.predict_gcn \
    --data_dir     "$DATA_DIR" \
    --layer        "$LAYER" \
    --graph_type   "$GRAPH_TYPE" \
    --seed         "$SEED" \
    --test_size    "$TEST_SIZE" \
    --epochs       "$EPOCHS" \
    --lr           "$LR" \
    --batch_size   "$BATCH_SIZE" \
    --weight_decay "$WEIGHT_DECAY" \
    --embedding_dim "$EMBEDDING_DIM" \
    --hidden_dim   "$HIDDEN_DIM" \
    --fc_hidden_dim "$FC_HIDDEN_DIM" \
    --num_layers   "$NUM_GCN_LAYERS" \
    --dropout      "$DROPOUT" \
    --activation   "$ACTIVATION" \
    --device       "$DEVICE" \
    --output_dir   "$OUTPUT_DIR" \
    --save_model \
    --log_every    10
