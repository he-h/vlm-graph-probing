
# samples=5000
samples=1000
log_every=1000
cuda_device=cuda:1

python -m probing.prepare_data \
    --model_ckpt OpenGVLab/InternVL3-1B-hf \
    --category color \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose

python -m probing.prepare_data \
    --model_ckpt llava-hf/llava-1.5-7b-hf \
    --category color \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose

python -m probing.prepare_data \
    --model_ckpt Qwen/Qwen2.5-VL-3B-Instruct \
    --category color \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose

python -m probing.prepare_data \
    --model_ckpt llava-hf/llava-1.5-13b-hf \
    --category color \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose

python -m probing.prepare_data \
    --model_ckpt Qwen/Qwen2.5-VL-7B-Instruct \
    --category color \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose

python -m probing.prepare_data \
    --model_ckpt OpenGVLab/InternVL3-4B-hf \
    --category color \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose

python -m probing.prepare_data \
    --model_ckpt OpenGVLab/InternVL3-14B-hf \
    --category color \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose


# ==========================

python -m probing.prepare_data \
    --model_ckpt OpenGVLab/InternVL3-1B-hf \
    --category counting \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose

python -m probing.prepare_data \
    --model_ckpt llava-hf/llava-1.5-7b-hf \
    --category counting \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose

python -m probing.prepare_data \
    --model_ckpt Qwen/Qwen2.5-VL-3B-Instruct \
    --category counting \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose

python -m probing.prepare_data \
    --model_ckpt llava-hf/llava-1.5-13b-hf \
    --category counting \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose

python -m probing.prepare_data \
    --model_ckpt Qwen/Qwen2.5-VL-7B-Instruct \
    --category counting \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose

python -m probing.prepare_data \
    --model_ckpt OpenGVLab/InternVL3-4B-hf \
    --category counting \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose

python -m probing.prepare_data \
    --model_ckpt OpenGVLab/InternVL3-14B-hf \
    --category counting \
    --num_samples $samples \
    --device $cuda_device \
    --log_every $log_every \
    --sparse_level 0.9 \
    --layer_slices 4 \
    --verbose
