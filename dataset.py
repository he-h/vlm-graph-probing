import json
from datasets import load_dataset
from PIL import Image

from utils import *

CLEVR_COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
CLEVR_SHAPES = ['cube', 'sphere', 'cylinder']
CLEVR_COUNTS = [str(i) for i in range(10)]

def split_clevr_question_answer(qa_str):
    list_ = qa_str.split("?")
    if len(list_) != 2:
        raise ValueError(f"Unexpected QA format, cannot split question and answer: {qa_str}")
    question = list_[0] + "?"
    answer = list_[1].strip().lower()
    return question, answer

def constrain_clevr_prompt(question: str, task: str) -> str:
    if task == "color":
        choices = ", ".join(CLEVR_COLORS)
        return f"{question} Answer with one word from: {choices}. Output exactly one word."
    elif task == "counting":
        return f"{question} Answer with a single integer 0-9. Output only the number."
    elif task == "existence":
        return f"{question} Answer with 'yes' or 'no' only. Output exactly one word."
    elif task == "comparison":
        return f"{question} Answer with 'more', 'fewer', or 'equal' only. Output exactly one word."
    elif task == "shape":
        choices = ", ".join(CLEVR_SHAPES)
        return f"{question} Answer with one word from: {choices}. Output exactly one word."
    else:
        return question

def candidate_answers(task: str):
    if task == "color":
        return CLEVR_COLORS
    elif task == "counting":
        return [str(i) for i in range(10)]
    elif task == "existence":
        return ['yes', 'no']
    elif task == "comparison":
        return ['more', 'fewer', 'equal']
    elif task == "shape":
        return CLEVR_SHAPES
    else:
        return []

def classify_clevr_question(question, answer):
    question = question.lower()
    answer = answer.lower()
    try:
        num = int(answer)
        if 0 <= num < 10:
            return 'counting'
    except:
        pass
    if "color" in question and answer in CLEVR_COLORS:
        return 'color'
    elif answer in CLEVR_SHAPES:
        return 'shape'
    else:
        return 'unknown'


def caption_prompt(choice=0, add_1sent_constraint=False):
    '''Return the caption prompt based on the choice. From short to long. range 0-2'''
    prompts = [
        "Describe the image.",
        "Provide a caption for this image in one sentence.",
        "Provide a detailed caption describing the objects, colors, and relationships in the image."
    ]
    
    candidates = [
        "Caption the image focusing on spatial positions (left, right, top, bottom).",
        "Write a caption using only what is clearly visible. Do not guess or infer.",
    ]

    if add_1sent_constraint:
        return prompts[choice] + " Respond with only one sentence, nothing else."
    return prompts[choice]


def prepare_vlm_data(dataset: str, num_samples: int, task: str = "color", prompt_choice: int = 1):
    """
    Prepare a test set for Vision-Language Model evaluation (VQA or captioning).

    Args:
        dataset (str): Which dataset to load. One of:
            - "coco"   → COCO Caption 2017 (captioning)
            - "clevr"  → CLEVR webdataset (VQA-style)
        num_samples (int): Number of samples to load.
        task (str): For CLEVR only; specify question type
                    ('color', 'counting', 'existence', 'comparison', 'shape').

    Returns:
        list[list]: Each element is [image, prompt, response]
            - image (PIL.Image): RGB image
            - prompt (str): Text input (caption prompt or question)
            - response (str | list[str]): Ground-truth answer(s)
    """
    dataset = dataset.lower()
    samples = []

    print(f"Preparing {num_samples} samples from {dataset.upper()} dataset...")

    if dataset == "coco":
        ds = load_dataset("lmms-lab/COCO-Caption2017", split="val")
        ds = ds.select(range(min(num_samples, len(ds))))
        for s in ds:
            img = s["image"].convert("RGB")
            prompt = caption_prompt(prompt_choice, add_1sent_constraint=False)
            refs = s.get("answer", [])
            samples.append([img, prompt, refs])

    elif dataset == "clevr":
        ds = load_dataset("laion/clevr-webdataset", split="validation")
        cnt = 0
        for s in ds:
            try:
                q, a = split_clevr_question_answer(s["txt"])
                if classify_clevr_question(q, a) == task:
                    img = s["jpg"].convert("RGB")
                    prompt = constrain_clevr_prompt(q, task)
                    samples.append([img, prompt, a])
                    cnt += 1
                    if cnt >= num_samples:
                        break
            except Exception:
                continue
    else:
        raise ValueError("Dataset must be 'coco' or 'clevr'.")

    print(f"Loaded {len(samples)} samples from {dataset.upper()}.")
    return samples

def get_candidate_answers(dataset, task="color"):
    """
    Get candidate answers for VQA tasks.

    Args:
        dataset (str): Which dataset to load. One of:
            - "clevr"  → CLEVR webdataset (VQA-style)
        task (str): For CLEVR only; specify question type
                    ('color', 'counting', 'existence', 'comparison', 'shape').

    Returns:
        list[str]: Candidate answers
    """
    dataset = dataset.lower()

    if dataset == "clevr":
        if task == "color":
            return CLEVR_COLORS
        elif task == "counting":
            return CLEVR_COUNTS
        elif task == "existence":
            return ['yes', 'no']
    else:
        raise ValueError("Dataset must be 'clevr' for candidate answers.")