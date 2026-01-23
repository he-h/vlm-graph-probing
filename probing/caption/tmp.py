import torch
from datasets import load_dataset
from PIL import Image
import time

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    LlavaForConditionalGeneration,
    Gemma3ForConditionalGeneration,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

QWEN_ID  = "Qwen/Qwen2.5-VL-3B-Instruct"
LLAVA_ID = "llava-hf/llava-1.5-7b-hf"
GEMMA_ID = "google/gemma-3-4b-it" 

def count_tokens_total_text(processor, prompt: str, inputs):
    total_len = inputs["input_ids"].shape[1]
    text_len  = processor.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    vision_len = total_len - text_len
    return total_len, text_len, vision_len

def run_qwen(image: Image.Image):
    print("\n=== Qwen2.5-VL-3B-Instruct ===")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(QWEN_ID, use_fast=True)

    # Manual ChatML with Qwen vision tokens (no apply_chat_template)
    prompt = (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        "Provide a caption for this image in one sentence.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    input_ids = inputs["input_ids"]
    total_len = input_ids.shape[1]
    image_token_id = model.config.image_token_id
    vision_len = int((input_ids == image_token_id).sum().item())
    text_len = total_len - vision_len

    print(f"Total input length: {total_len}")
    print(f"  Vision token length: {vision_len}")
    print(f"  Text prompt token length: {text_len}")

    eos_id = processor.tokenizer.eos_token_id
    pad_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else eos_id

    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            min_new_tokens=8,     # encourage continuation in case of early EOS
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
    dt = time.time() - t0

    seq = out.sequences
    new_ids = seq[:, total_len:]
    caption = processor.tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

    print(f"New tokens: {new_ids.shape[1]} | Time: {dt:.2f}s")
    print("Caption:", caption if caption else "[EMPTY]")

def run_llava(image: Image.Image):
    print("\n=== LLaVA-1.5-7B-HF ===")
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(LLAVA_ID, use_fast=True)

    prompt = "Provide a caption for this image in one sentence."
    prompt = f"USER: <image>\n{prompt} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    total_len, text_len, vision_len = count_tokens_total_text(processor, prompt, inputs)
    print(f"Total input length: {total_len}")
    print(f"  Vision token length (heuristic): {vision_len}")
    print(f"  Text prompt token length: {text_len}")

    eos_id = processor.tokenizer.eos_token_id
    pad_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else eos_id

    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            return_dict_in_generate=True,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
    dt = time.time() - t0

    seq = out.sequences
    new_ids = seq[:, total_len:]
    caption = processor.tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

    print(f"New tokens: {new_ids.shape[1]} | Time: {dt:.2f}s")
    print("Caption:", caption if caption else "[EMPTY]")

def run_gemma_it(image: Image.Image):
    print("\n=== Gemma-3-4B-IT ===")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        GEMMA_ID
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(GEMMA_ID, use_fast=True)

    # Instruction-tuned Gemma: plain language is fine; no manual <start_of_image> needed
    prompt = "<start_of_image> Provide a caption for this image in one sentence."

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    total_len, text_len, vision_len = count_tokens_total_text(processor, prompt, inputs)
    print(f"Total input length: {total_len}")
    print(f"  Vision token length (heuristic): {vision_len}")
    print(f"  Text prompt token length: {text_len}")

    eos_id = processor.tokenizer.eos_token_id
    pad_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else eos_id

    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            return_dict_in_generate=True,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            min_new_tokens=1,
        )
    dt = time.time() - t0

    seq = out.sequences
    new_ids = seq[:, total_len:]
    caption = processor.tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

    print(f"New tokens: {new_ids.shape[1]} | Time: {dt:.2f}s")
    print("Caption:", caption if caption else "[EMPTY]")

if __name__ == "__main__":
    coco = load_dataset("lmms-lab/COCO-Caption2017", split="val")
    sample = coco[0]
    image = sample["image"].convert("RGB")

    print("Reference captions (first 2):")
    for r in sample["answer"][:2]:
        print(" -", r)

    run_qwen(image)
    run_gemma_it(image)
    run_llava(image)
    
