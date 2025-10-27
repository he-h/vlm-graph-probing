export HF_TOKEN="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"
export HF_HOME="../cache/huggingface"

# samples=5000
samples=1000
log_every=1000
cuda_device=cuda:1

python -m probing.vqa.prepare_data \
    --model_ckpt OpenGVLab/InternVL3-1B-hf \
    --task color \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose

python -m probing.vqa.prepare_data \
    --model_ckpt llava-hf/llava-1.5-7b-hf \
    --task color \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose

python -m probing.vqa.prepare_data \
    --model_ckpt Qwen/Qwen2.5-VL-3B-Instruct \
    --task color \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose

python -m probing.vqa.prepare_data \
    --model_ckpt llava-hf/llava-1.5-13b-hf \
    --task color \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose

python -m probing.vqa.prepare_data \
    --model_ckpt Qwen/Qwen2.5-VL-7B-Instruct \
    --task color \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose

python -m probing.vqa.prepare_data \
    --model_ckpt OpenGVLab/InternVL3-4B-hf \
    --task color \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose

python -m probing.vqa.prepare_data \
    --model_ckpt OpenGVLab/InternVL3-14B-hf \
    --task color \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose


# ==========================

python -m probing.vqa.prepare_data \
    --model_ckpt OpenGVLab/InternVL3-1B-hf \
    --task counting \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose

python -m probing.vqa.prepare_data \
    --model_ckpt llava-hf/llava-1.5-7b-hf \
    --task counting \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose

python -m probing.vqa.prepare_data \
    --model_ckpt Qwen/Qwen2.5-VL-3B-Instruct \
    --task counting \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose

python -m probing.vqa.prepare_data \
    --model_ckpt llava-hf/llava-1.5-13b-hf \
    --task counting \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose

python -m probing.vqa.prepare_data \
    --model_ckpt Qwen/Qwen2.5-VL-7B-Instruct \
    --task counting \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose

python -m probing.vqa.prepare_data \
    --model_ckpt OpenGVLab/InternVL3-4B-hf \
    --task counting \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose

python -m probing.vqa.prepare_data \
    --model_ckpt OpenGVLab/InternVL3-14B-hf \
    --task counting \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose
