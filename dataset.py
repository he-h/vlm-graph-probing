import argparse
import json
import os

from datasets import load_dataset
from PIL import Image


CLEVR_COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
TDIUC_COLORS = ['white', 'blue', 'red', 'green', 'black', 'yellow', 'brown', 'gray', 'silver', 'orange']
TDIUC_COUNTS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
CLEVR_SHAPES = ['cube', 'sphere', 'cylinder']
COUNTS = [str(i) for i in range(10)]


def resolve_data_roots(data_root=None, tdiuc_root=None, coco_val_root=None):
    """Resolve dataset locations from explicit args first, then environment variables, then local defaults."""
    base_data_root = data_root or os.getenv("VLM_GRAPH_DATA_ROOT", "data")
    resolved_tdiuc_root = tdiuc_root or os.getenv("VLM_GRAPH_TDIUC_ROOT", os.path.join(base_data_root, "TDIUC"))
    resolved_coco_val_root = coco_val_root or os.getenv("VLM_GRAPH_COCO_VAL_ROOT", os.path.join(base_data_root, "val2014"))
    return {
        "data_root": base_data_root,
        "tdiuc_root": resolved_tdiuc_root,
        "coco_val_root": resolved_coco_val_root,
    }


def split_clevr_question_answer(qa_str):
    list_ = qa_str.split("?")
    if len(list_) != 2:
        raise ValueError(f"Unexpected QA format, cannot split question and answer: {qa_str}")
    question = list_[0] + "?"
    answer = list_[1].strip().lower()
    return question, answer

def constrain_vqa_prompt(dataset, question: str, category: str) -> str:
    if category == "color":
        if dataset == "tdiuc":
            choices = ", ".join(TDIUC_COLORS)
        elif dataset == "clevr":
            choices = ", ".join(CLEVR_COLORS)
        return f"{question} Answer with one word from: {choices}. Output exactly one word."
    elif category == "counting":
        return f"{question} Answer with a single integer 0-9. Output only the number."
    elif category == "existence":
        return f"{question} Answer with 'yes' or 'no' only. Output exactly one word."
    elif category == "comparison":
        return f"{question} Answer with 'more', 'fewer', or 'equal' only. Output exactly one word."
    elif category == "shape":
        choices = ", ".join(CLEVR_SHAPES)
        return f"{question} Answer with one word from: {choices}. Output exactly one word."
    else:
        return question


def classify_clevr_question(question, answer):
    question = question.lower()
    answer = answer.lower()
    try:
        num = int(answer)
        if 0 <= num < 10:
            return 'counting'
    except (ValueError, TypeError):
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


def prepare_vlm_data(
    dataset: str,
    num_samples: int,
    category: str = "color",
    prompt_choice: int = 1,
    balance: bool = False,
    data_root: str | None = None,
    tdiuc_root: str | None = None,
    coco_val_root: str | None = None,
) -> list:
    """
    Prepare a test set for Vision-Language Model evaluation (VQA or captioning).

    Args:
        dataset (str): Which dataset to load. One of:
            - "coco"   → COCO Caption 2017 (captioning)
            - "clevr"  → CLEVR webdataset (VQA-style)
        num_samples (int): Number of samples to load.
        category (str): For CLEVR only; specify question type
                    ('color', 'counting', 'existence', 'comparison', 'shape').

    Returns:
        list[list]: Each element is [image, prompt, response]
            - image (PIL.Image): RGB image
            - prompt (str): Text input (caption prompt or question)
            - response (str | list[str]): Ground-truth answer(s)
    """
    dataset = dataset.lower()
    samples = []
    paths = resolve_data_roots(data_root=data_root, tdiuc_root=tdiuc_root, coco_val_root=coco_val_root)

    balance_str = " (balanced)" if balance else ""
    print(f"Preparing {num_samples} samples from {dataset.upper()} dataset{balance_str}...")
    
    if balance:
        candidates = get_candidate_answers(dataset, category)
        samples_per_class = num_samples // len(candidates)
        class_counts = {ans: 0 for ans in candidates}
        print(f"Balancing: {samples_per_class} samples per class")

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
                if classify_clevr_question(q, a) == category:
                    if balance:
                        if class_counts.get(a, 0) >= samples_per_class:
                            continue
                        class_counts[a] = class_counts.get(a, 0) + 1
                    img = s["jpg"].convert("RGB")
                    prompt = constrain_vqa_prompt(dataset, q, category)
                    samples.append([img, prompt, a])
                    cnt += 1
                    if cnt >= num_samples:
                        break
            except Exception:  # skip malformed samples
                continue
    elif dataset == "tdiuc":
        questions_path = os.path.join(paths["tdiuc_root"], "Questions", "OpenEnded_mscoco_val2014_questions.json")
        annotations_path = os.path.join(paths["tdiuc_root"], "Annotations", "mscoco_val2014_annotations.json")
        if not os.path.exists(questions_path):
            raise FileNotFoundError(f"TDIUC questions file not found: {questions_path}")
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"TDIUC annotations file not found: {annotations_path}")

        with open(questions_path, "r") as f:
            data = json.load(f)
        with open(annotations_path, "r") as f:
            annote_data = json.load(f)

        image_id_to_questions = {}
        for item in data["questions"]:
            image_id_to_questions[item["question_id"]] = item

        cnt = 0
        for ann in annote_data["annotations"]:
            if ann["question_type"] != category:
                continue
            if ann["answers"][0]['answer_confidence'] != "yes":
                continue
            question_item = image_id_to_questions.get(ann["question_id"], None)
            if question_item is None:
                continue
            image_path = os.path.join(paths["coco_val_root"], f"COCO_val2014_{ann['image_id']:012d}.jpg")
            try:
                img = Image.open(image_path).convert("RGB")
            except Exception:
                continue
            prompt = constrain_vqa_prompt(dataset, question_item["question"], category)
            ref = ann['answers'][0]['answer'].strip().lower()
            if category == "color" and ref.lower() not in TDIUC_COLORS:
                continue
            elif category == "counting":
                if ref not in TDIUC_COUNTS:
                    continue
                ref = number_word_to_digit(ref)
            
            # Check balance constraint
            if balance and class_counts.get(str(ref), 0) >= samples_per_class:
                continue
            
            samples.append([img, prompt, ref])
            
            if balance:
                class_counts[str(ref)] = class_counts.get(str(ref), 0) + 1
            cnt += 1
            if cnt >= num_samples:
                break
    else:
        raise ValueError(f"Unsupported dataset: '{dataset}'.")

    print(f"Loaded {len(samples)} samples from {dataset.upper()}.")
    return samples

def get_candidate_answers(dataset, category="color"):
    """
    Get candidate answers for VQA categorys.

    Args:
        dataset (str): Which dataset to load. One of:
            - "clevr"  → CLEVR webdataset (VQA-style)
        category (str): For CLEVR only; specify question type
                    ('color', 'counting', 'existence', 'comparison', 'shape').

    Returns:
        list[str]: Candidate answers
    """
    dataset = dataset.lower()

    if dataset == "clevr":
        if category == "color":
            return CLEVR_COLORS
        elif category == "counting":
            return COUNTS
        elif category == "existence":
            return ['yes', 'no']
    elif dataset == "tdiuc":
        if category == "color":
            return TDIUC_COLORS
        elif category == "counting":
            return COUNTS
    else:
        raise ValueError(f"Unsupported dataset for candidate answers: '{dataset}'. Supported: 'clevr', 'tdiuc'.")

def number_word_to_digit(word):
    word = word.lower()
    word_to_digit = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9
    }
    return word_to_digit.get(word, word)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset utilities for VLM graph probing.")
    parser.add_argument("--self-test", action="store_true", help="Run a tiny path-resolution self-test.")
    args = parser.parse_args()
    if args.self_test:
        print(resolve_data_roots())
    else:
        print("dataset.py is a library module. Use --self-test for a tiny smoke check.")
