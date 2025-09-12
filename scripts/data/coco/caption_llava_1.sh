export HF_TOKEN="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"
export HF_HOME="../cache/huggingface"

python -m probing.caption.prepare_data \
    --model_name llava-hf/llava-1.5-7b-hf \
    --prompt_choice 1 \
    --batch_size 8 \
    --num_samples 2500 \
    --device cuda:0 \

# python -m probing.caption.prepare_data \
#     --model_name llava-hf/llava-1.5-13b-hf \
#     --prompt_choice 1 \
#     --batch_size 4 \
#     --num_samples 2500 \
#     --device cuda:1 \
#     --sparse_level 0.9
