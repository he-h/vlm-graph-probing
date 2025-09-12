export HF_TOKEN="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"
export HF_HOME="../cache/huggingface"

python -m probing.vqa.prepare_data \
    --model_name Qwen/Qwen2.5-VL-3B-Instruct \
    --batch_size 16 \
    --num_samples 1000 \
    --device cuda:0 \
    --sparse_level 0.9 \
    --log_every 50 \
    --task color
