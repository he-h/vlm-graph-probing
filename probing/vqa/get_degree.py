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


def compute_node_degrees(corr_graph):
    """
    Compute node degrees from correlation graph in COO format.
    
    Args:
        corr_graph: dict with keys:
            - "num_nodes": int
            - "edge_index": np.ndarray [2, E] 
            - "edge_weight": np.ndarray [E]
    
    Returns:
        degrees: np.ndarray [num_nodes] - degree of each node
    """
    num_nodes = corr_graph["num_nodes"]
    edge_index = corr_graph["edge_index"]  # [2, E]
    
    # Initialize degree array
    degrees = np.zeros(num_nodes, dtype=np.float64)
    
    # Count outgoing edges for each node (row indices)
    source_nodes = edge_index[0, :]  # [E]
    np.add.at(degrees, source_nodes, 1)
    
    # Count incoming edges for each node (column indices)
    target_nodes = edge_index[1, :]  # [E]
    np.add.at(degrees, target_nodes, 1)
    
    
    return degrees


def create_clevr_layer_analysis(
    num_samples=1000,
    model_name="llava-hf/llava-1.5-7b-hf",
    output_dir="layer_analysis",
    verbose=False,
    device="cuda:0",
    task='color',
    sparse_level=0.9,
    log_every=5,
):
    """
    Process CLEVR dataset and compute:
    1. Average node degree for each neuron across all layers (shape: [num_layers, hidden_dim])
    2. Average absolute activation of last token across all layers (shape: [num_layers, hidden_dim])
    """
    model_prefix = model_path2name(model_name)
    output_dir = f"data/{model_prefix}_clevr_{task}_sparsity_{int(sparse_level * 100)}_{output_dir}"
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("Loading CLEVR dataset from HuggingFace...")
    clevr_samples = load_dataset("laion/clevr-webdataset", split='validation')
    if not clevr_samples:
        print("ERROR: Failed to load CLEVR dataset!")
        return

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

    num_layers = extractor.num_layers
    hidden_dim = extractor.hidden_dim
    
    # Initialize accumulators for all layers
    # Shape: [num_layers, hidden_dim]
    degree_accumulator = np.zeros((num_layers, hidden_dim), dtype=np.float64)
    activation_accumulator = np.zeros((num_layers, hidden_dim), dtype=np.float64)
    
    correct, total = 0, 0
    samples_processed = 0

    print(f"\nProcessing {valid_sample_counts} samples across ALL {num_layers} layers...")
    print(f"Output shapes will be: [{num_layers}, {hidden_dim}]")
    print("="*60)

    for sample_idx in tqdm(range(valid_sample_counts), desc="Processing samples"):
        image = all_images[sample_idx]
        question = all_questions[sample_idx]
        answer = all_answers[sample_idx]

        try:
            prompt = constrain_clevr_prompt(question, task)
            hidden_states_all, gen = extractor.process_single(image, prompt)
            # hidden_states_all: list of [1, seq, hidden] tensors, one per layer
            
            # Process each layer
            for layer_idx in range(num_layers):
                if layer_idx >= len(hidden_states_all):
                    continue
                    
                hs = hidden_states_all[layer_idx][0]  # [seq, hidden]
                
                # 1. Compute correlation graph and extract degrees
                corr_graph = corr_graph_torch(hs, sparse_level=sparse_level)
                degrees = compute_node_degrees(corr_graph)
                
                # Ensure correct shape
                if len(degrees) != hidden_dim:
                    print(f"Warning: degree shape mismatch at layer {layer_idx}: {len(degrees)} vs {hidden_dim}")
                    continue
                
                degree_accumulator[layer_idx] += degrees
                
                # 2. Extract last token activation (absolute value)
                last_token = hs[-1, :].detach().cpu().numpy()  # [hidden]
                activation_accumulator[layer_idx] += np.abs(last_token)
            
            # Track accuracy
            if isinstance(gen, str) and gen.lower().strip() == answer:
                correct += 1
            total += 1
            samples_processed += 1

            if sample_idx % log_every == 0 and sample_idx > 0:
                print(f"\n--- Sample {sample_idx} ---")
                print(f"Q: {question}")
                print(f"Pred: {gen} | Ref: {answer}")
                print(f"Current accuracy: {correct}/{total} = {correct/total*100:.2f}%")

        except Exception as e:
            print(f"Error processing sample {sample_idx}: {str(e)}")
            if verbose:
                import traceback
                traceback.print_exc()
            continue

        # Clear GPU memory
        del hidden_states_all
        torch.cuda.empty_cache()

    # Compute averages
    if samples_processed > 0:
        avg_degree = degree_accumulator / samples_processed
        avg_activation = activation_accumulator / samples_processed
    else:
        print("ERROR: No samples processed successfully!")
        return

    print(f"\n{'='*60}")
    print("FINAL STATISTICS")
    print("="*60)
    print(f"Samples processed: {samples_processed}")
    print(f"Accuracy: {correct}/{total} = {correct/total*100:.2f}%")
    print(f"\nAverage degree matrix shape: {avg_degree.shape}")
    print(f"Average activation matrix shape: {avg_activation.shape}")
    print(f"Average degree range: [{avg_degree.min():.4f}, {avg_degree.max():.4f}]")
    print(f"Average activation range: [{avg_activation.min():.4f}, {avg_activation.max():.4f}]")

    # Save the arrays
    degree_path = os.path.join(output_dir, "avg_node_degree_all_layers.npy")
    activation_path = os.path.join(output_dir, "avg_last_token_activation_all_layers.npy")
    
    np.save(degree_path, avg_degree)
    np.save(activation_path, avg_activation)

    # Save metadata
    metadata = {
        "num_samples": samples_processed,
        "model": model_name,
        "model_type": extractor.model_type,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "sparse_level": sparse_level,
        "task": task,
        "accuracy": correct / total if total > 0 else 0.0,
        "candidate_answers": candidate_answers(task),
        "num_classes": len(candidate_answers(task)),
        "output_shapes": {
            "avg_node_degree": list(avg_degree.shape),
            "avg_activation": list(avg_activation.shape)
        },
        "statistics": {
            "degree_min": float(avg_degree.min()),
            "degree_max": float(avg_degree.max()),
            "degree_mean": float(avg_degree.mean()),
            "degree_std": float(avg_degree.std()),
            "activation_min": float(avg_activation.min()),
            "activation_max": float(avg_activation.max()),
            "activation_mean": float(avg_activation.mean()),
            "activation_std": float(avg_activation.std()),
        }
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nData saved to: {output_dir}")
    print("Files created:")
    print(f"  - avg_node_degree_all_layers.npy (shape: {avg_degree.shape})")
    print(f"  - avg_last_token_activation_all_layers.npy (shape: {avg_activation.shape})")
    print(f"  - metadata.json")
    print("="*60)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run layer-wise analysis on CLEVR dataset (all layers)")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--num_samples", type=int, default=2500)
    parser.add_argument("--sparse_level", type=float, default=0.9)
    parser.add_argument("--task", type=str, default="color")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="layer_analysis")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log_every", type=int, default=5)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Creating CLEVR Layer Analysis (ALL layers)")
    print(f"Model: {args.model_name}")
    print(f"Task: {args.task}")
    print(f"{'='*60}\n")

    create_clevr_layer_analysis(
        num_samples=args.num_samples,
        model_name=args.model_name,
        output_dir=args.output_dir,
        sparse_level=args.sparse_level,
        task=args.task,
        verbose=args.verbose,
        device=args.device,
        log_every=args.log_every,
    )