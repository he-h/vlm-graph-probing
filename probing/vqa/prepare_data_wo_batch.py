import torch
import numpy as np
from datasets import load_dataset
from PIL import Image
import json
from tqdm import tqdm
import pickle
import scipy.sparse as sp
import os
import warnings
import argparse

from utils import *
from metrics import spice_scores, meteor_scores, rougeL_scores, bertscore_f1, sanitize_preds_refs
from model import NeuronGraphExtractor as GraphExtractor
from model import corr_graph_torch

def split_clevr_question_answer(qa_str):
    list_ = qa_str.split("?")
    if len(list_) != 2:
        raise ValueError(f"Unexpected QA format, cannot split question and answer: {qa_str}")
    question = list_[0] + "?"
    answer = list_[1].strip().lower()
    return question, answer

CLEVR_COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
CLEVR_SHAPES = ['cube', 'sphere', 'cylinder']

def constrain_clevr_prompt(question: str, task: str) -> str:
    if task == "color":
        choices = ", ".join(CLEVR_COLORS)
        return f"{question} Answer with one word from: {choices}. Output exactly one word."
    elif task == "counting":
        return f"{question} Answer with a single integer 0-10. Output only the number."
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
        return [str(i) for i in range(11)]
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
        if 0 <= num <= 10:
            return 'counting'
    except:
        pass
    if "color" in question and answer in CLEVR_COLORS:
        return 'color'
    elif answer in CLEVR_SHAPES:
        return 'shape'
    else:
        return 'unknown'


def create_clevr_dataset(
    num_samples=1000,
    model_name="llava-hf/llava-1.5-7b-hf",
    output_dir="probing_dataset",
    verbose=False,
    device="cuda:0",
    task='color',
    sparse_level=0.9,
    log_every=5,
):
    model_prefix = model_path2name(model_name)
    output_dir = f"data/{model_prefix}_clevr_{task}_sparsity_{int(sparse_level * 100)}_{output_dir}"
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("Loading CLEVR dataset from HuggingFace...")
    clevr_samples = load_dataset("laion/clevr-webdataset", split='validation')
    if not clevr_samples:
        print("ERROR: Failed to load CLEVR dataset!")
        return []

    valid_sample_counts = 0
    all_images, all_questions, all_answers = [], [], []
    for sample in clevr_samples:
        txt = sample['txt']
        try:
            question, answer = split_clevr_question_answer(txt)
            if classify_clevr_question(question, answer) == task:
                valid_sample_counts += 1
                all_questions.append(question)
                all_answers.append(answer)
                all_images.append(sample['jpg'].convert("RGB"))
        except Exception:
            continue
        if valid_sample_counts >= num_samples:
            break

    print(f"Loaded {valid_sample_counts} samples for task={task}")
    print("="*60)
    print(f"Initializing {model_name} model...")
    extractor = GraphExtractor(model_name=model_name, device=device)

    # Match COCO: 0/25/50/75/100 percentiles across layers
    layer_indices = {
        'layer_0': 0,
        'layer_25': extractor.num_layers // 4,
        'layer_50': extractor.num_layers // 2,
        'layer_75': (3 * extractor.num_layers) // 4,
        'layer_100': extractor.num_layers - 1
    }

    all_samples = []
    correct, total = 0, 0
    all_preds, all_refs = [], []

    for sample_idx in tqdm(range(valid_sample_counts), desc="Processing samples"):
        image = all_images[sample_idx]
        question = all_questions[sample_idx]
        answer = all_answers[sample_idx]

        try:
            prompt = constrain_clevr_prompt(question, task)
            hidden_states_all, gen = extractor.process_single(image, prompt)
            # move all to CPU and detach for safe storage
            hidden_states_all = [h.detach().cpu() for h in hidden_states_all]  # [num_layers, 1, seq, hidden]

            graphs = {}
            for lname, lidx in layer_indices.items():
                lidx = min(lidx, len(hidden_states_all) - 1)
                hs = hidden_states_all[lidx][0]  # [seq, hidden]
                graphs[lname] = corr_graph_torch(hs, sparse_level=sparse_level)

            # Last-token states at each percentile layer (store tensors on CPU, consistent with COCO)
            sample_data = {
                "graph_layer_0":    graphs["layer_0"],
                "graph_layer_25":   graphs["layer_25"],
                "graph_layer_50":   graphs["layer_50"],
                "graph_layer_75":   graphs["layer_75"],
                "graph_layer_100":  graphs["layer_100"],
                "last_token_layer_0":    hidden_states_all[layer_indices['layer_0']][0, -1, :],
                "last_token_layer_25":   hidden_states_all[layer_indices['layer_25']][0, -1, :],
                "last_token_layer_50":   hidden_states_all[layer_indices['layer_50']][0, -1, :],
                "last_token_layer_75":   hidden_states_all[layer_indices['layer_75']][0, -1, :],
                "last_token_layer_100":  hidden_states_all[layer_indices['layer_100']][0, -1, :],
                "predicted_answer": gen.lower().strip() if isinstance(gen, str) else gen,
                "reference_answer": answer,
            }
            all_samples.append(sample_data)

            # accuracy
            if isinstance(gen, str) and gen.lower().strip() == answer:
                correct += 1
            total += 1
            all_preds.append(gen)
            all_refs.append(answer)

            if sample_idx % log_every == 0 and sample_idx > 0:
                print(f"\n--- Sample {sample_idx} ---")
                print(f"Q: {question}")
                print(f"Pred: {gen} | Ref: {answer}")

        except Exception as e:
            print(f"Error processing sample {sample_idx}: {str(e)}")
            continue

        # clear GPU
        del hidden_states_all
        torch.cuda.empty_cache()

    print(f"Total samples processed: {len(all_samples)}")

    if all_samples:
        print(f"\n{'='*60}")
        print("FINAL STATISTICS")
        print("="*60)
        print(f"Samples processed: {len(all_samples)}")
        print(f"Accuracy: {correct}/{total} = {correct/total*100:.2f}%")

        # quick structure check
        sc = all_samples[0]
        print("\nSample structure verification keys:")
        print(list(sc.keys()))

        # save
        final_path = os.path.join(output_dir, "complete_dataset.pkl")
        with open(final_path, "wb") as f:
            pickle.dump(all_samples, f)

        metadata = {
            "num_samples": len(all_samples),
            "model": model_name,
            "model_type": extractor.model_type,
            "num_layers": extractor.num_layers,
            "hidden_dim": extractor.hidden_dim,
            "layers_extracted": list(layer_indices.keys()),
            "layer_indices": layer_indices,
            "sample_keys": list(all_samples[0].keys()) if all_samples else [],
            "sparse_level": sparse_level,
            "task": task,
            "accuracy": correct / total if total > 0 else 0.0,
            "candidate_answers": candidate_answers(task),
            "num_classes": len(candidate_answers(task)),
            "processing_mode": "single_sample"
        }

        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nDataset saved to: {output_dir}")
        print("Files created:\n  - complete_dataset.pkl\n  - metadata.json\n" + "="*60)

    return all_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run probing CLEVR dataset creation (single sample, 0/25/50/75/100 layers)")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--num_samples", type=int, default=2500)
    parser.add_argument("--sparse_level", type=float, default=0.9)
    parser.add_argument("--task", type=str, default="color")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="probing_dataset")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log_every", type=int, default=5)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Creating CLEVR dataset (single sample, 0/25/50/75/100)")
    print(f"Model: {args.model_name}")
    print(f"Task: {args.task}")
    print(f"{'='*60}\n")

    dataset = create_clevr_dataset(
        num_samples=args.num_samples,
        model_name=args.model_name,
        output_dir=args.output_dir,
        sparse_level=args.sparse_level,
        task=args.task,
        verbose=args.verbose,
        device=args.device,
        log_every=args.log_every,
    )
