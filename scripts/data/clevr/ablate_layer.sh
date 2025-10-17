export HF_TOKEN="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"
export HF_HOME="../cache/huggingface"

# python -m probing.vqa.ablate_layer \
#     --model_name Qwen/Qwen2.5-VL-3B-Instruct \
#     --num_samples 100 \
#     --ablation_layer 18 \
#     --ablation_method random \
#     --device cuda:0 \
#     --sparse_level 0.9 \
#     --log_every 50 \
#     --task counting 

# python -m probing.vqa.ablate_layer \
#     --model_name Qwen/Qwen2.5-VL-3B-Instruct \
#     --num_samples 100 \
#     --ablation_layer 18 \
#     --ablation_method random \
#     --ablation_percentage 0.1 \
#     --device cuda:0 \
#     --sparse_level 0.9 \
#     --log_every 50 \
#     --task color 

python -m probing.vqa.ablate_layer \
    --model_name Qwen/Qwen2.5-VL-3B-Instruct \
    --num_samples 100 \
    --ablation_layer 0 \
    --ablation_method top_activation \
    --ablation_percentage 0.1 \
    --device cuda:6 \
    --sparse_level 0.9 \
    --log_every 10 \
    --task color 


# python -m probing.vqa.ablate_layer \
#     --model_name Qwen/Qwen2.5-VL-3B-Instruct \
#     --num_samples 100 \
#     --ablation_layer 18 \
#     --ablation_method top_activation \
#     --ablation_percentage 0.01 \
#     --device cuda:0 \
#     --sparse_level 0.9 \
#     --log_every 50 \
#     --task color 