export HF_TOKEN="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"
export HF_HOME="../cache/huggingface"

# samples=5000
samples=1000
log_every=100

python -m probing.caption.prepare_data \
    --model_ckpt OpenGVLab/InternVL3-1B-hf \
    --prompt_choice 1 \
    --num_samples $samples \
    --device cuda:0 \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose

python -m probing.caption.prepare_data \
    --model_ckpt llava-hf/llava-1.5-7b-hf \
    --prompt_choice 1 \
    --num_samples $samples \
    --device cuda:0 \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose

python -m probing.caption.prepare_data \
    --model_ckpt Qwen/Qwen2.5-VL-3B-Instruct \
    --prompt_choice 1 \
    --num_samples $samples \
    --device cuda:0 \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose

# python -m probing.caption.prepare_data \
#     --model_ckpt llava-hf/llava-1.5-13b-hf \
#     --prompt_choice 1 \
#     --num_samples $samples \
#     --device cuda:0 \
#     --log_every $log_every \
#     --sparse_level 0.9 \
#     --layer_slices 4 \
#     --verbose

# python -m probing.caption.prepare_data \
#     --model_ckpt Qwen/Qwen2.5-VL-7B-Instruct \
#     --prompt_choice 1 \
#     --num_samples $samples \
#     --device cuda:0 \
#     --log_every $log_every \
#     --sparse_level 0.9 \
#     --layer_slices 4 \
#     --verbose

# python -m probing.caption.prepare_data \
#     --model_ckpt OpenGVLab/InternVL3-4B-hf \
#     --prompt_choice 1 \
#     --num_samples $samples \
#     --device cuda:0 \
#     --log_every $log_every \
#     --sparse_level 0.9 \
#     --layer_slices 4 \
#     --verbose

# python -m probing.caption.prepare_data \
#     --model_ckpt OpenGVLab/InternVL3-14B-hf \
#     --prompt_choice 1 \
#     --num_samples $samples \
#     --device cuda:0 \
#     --log_every $log_every \
#     --sparse_level 0.9 \
#     --layer_slices 4 \
#     --verbose



