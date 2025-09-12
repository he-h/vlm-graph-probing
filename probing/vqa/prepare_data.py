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

# TODO: before run dataset, select question first 2, save correct and predicted answers

'''Exist, Count, Compare Integer, Query Attribute and Compare Attribute'''

def split_clevr_question_answer(qa_str):
    list_ = qa_str.split("?")
    if len(list_) != 2:
        raise ValueError(f"Unexpected QA format, cannot split question and answer: {qa_str}")
    question = list_[0] + "?"
    answer = list_[1].strip().lower()
    return question, answer

CLEVR_COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
CLEVR_SHAPES = ['cube', 'sphere', 'cylinder']
CLEVR_RELATIONS = ['left', 'right', 'front', 'behind', 'above', 'below']
CLEVR_EXISTENCE = ['Is there', 'Are there']
CLEVR_COUNTING = ['How many']

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
    batch_size=8,
    sparse_level=0.9,
    log_every=5,
):
    """
    Create graph probing dataset for VLM with CLEVR captions (optimized batched processing).
    """
    model_prefix = model_path2name(model_name)
    output_dir = f"data/{model_prefix}_clevr_{task}_sparsity_{int(sparse_level * 100)}_{output_dir}"
    os.makedirs(output_dir, exist_ok=True)

    # Load CLEVR dataset
    print("="*60)
    print("Loading CLEVR dataset from HuggingFace...")
    clevr_samples = load_dataset("laion/clevr-webdataset", split='validation')
    if not clevr_samples:
        print("ERROR: Failed to load CLEVR dataset from HuggingFace!")
        return []

    valid_sample_counts = 0
    all_images = []
    all_questions = []
    all_answers = []
    for sample in clevr_samples:
        txt = sample['txt']
        try:
            question, answer = split_clevr_question_answer(txt)
            if classify_clevr_question(question, answer) == task:
                valid_sample_counts += 1
                all_questions.append(question)
                all_answers.append(answer)
                all_images.append(sample['jpg'].convert("RGB"))
        except Exception as e:
            continue
        if valid_sample_counts >= num_samples:
            break

    print(f"Loaded {valid_sample_counts} samples from CLEVR validation dataset")
    print("="*60)
    print(f"Initializing {model_name} model...")
    extractor = GraphExtractor(model_name=model_name, device=device)

    layer_indices = {
        'layer_0': 0,
        'layer_middle': extractor.num_layers // 2,
        'layer_last': extractor.num_layers - 1 
    }

    all_samples = []
    print(f"Batch size: {batch_size}")

    # Process in batches with progress bar
    total_batches = (valid_sample_counts + batch_size - 1) // batch_size

    all_preds = []
    all_refs  = []
    
    for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(clevr_samples))
        images = all_images[start_idx:end_idx]
        questions = all_questions[start_idx:end_idx]
        answers = all_answers[start_idx:end_idx]
        valid_indices = list(range(start_idx, end_idx))

        if not images:
            continue

        try:
            graph_extraction_prompts = questions
            hidden_states_all, generations = extractor.process(
                images, graph_extraction_prompts
            )
            print(generations)
            
            # Process each item in batch
            for b_idx, (gen, global_idx) in enumerate(
                zip(generations, valid_indices)
            ):
                
                # compute correlation graph for selected layers
                graphs = {}
                for layer_name, layer_idx in layer_indices.items():
                    layer_idx = min(layer_idx, len(hidden_states_all) - 1)
                    
                    # Extract hidden states for this sample and layer
                    hs = hidden_states_all[layer_idx][b_idx]  # shape [seq, hidden]
                    graph = corr_graph_torch(hs, sparse_level=sparse_level)
                    graphs[layer_name] = graph

                # Last token hidden state from the last layer
                last_token_state = hidden_states_all[-1][b_idx, -1, :].detach().cpu().numpy()

                # Store sample
                sample = {
                    "graph_layer_0": graphs["layer_0"],
                    "graph_layer_middle": graphs["layer_middle"],
                    "graph_layer_last": graphs["layer_last"],
                    "last_token_state": last_token_state,
                    "predicted_answer": gen,
                    "reference_answer": batch_answers[b_idx],
                }
                all_samples.append(sample)

            if batch_idx % log_every == 0:
                print(f"Processed sample {global_idx}")
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {str(e)}")
            continue
        
        del hidden_states_all
        torch.cuda.empty_cache()

    print(f"Total samples processed: {len(all_samples)}")


    # Final statistics
    if all_samples:
        print(f"\n{'='*60}")
        print("FINAL STATISTICS")
        print("="*60)
        print(f"Samples processed: {len(all_samples)}")

        
        # Verify data structure
        sample_check = all_samples[0]
        print(f"\nSample structure verification:")
        print(f"  Keys in sample: {list(sample_check.keys())}")
        print(f"  Graph layer 0 keys: {list(sample_check['graph_layer_0'].keys())}")
        print(f"  Last token state shape: {sample_check['last_token_state'].shape}")
        # print sample edge index and weights for layer 0, first 5 edges
        edge_index = sample_check['graph_layer_0']['edge_index']
        edge_weight = sample_check['graph_layer_0']['edge_weight']
        print(f"  Graph layer 0 edges (first 5): {edge_index[:, :5]}")
        print(f"  Graph layer 0 weights (first 5): {edge_weight[:5]}")

    #     # Save results
    #     final_path = os.path.join(output_dir, "complete_dataset.pkl")
    #     with open(final_path, "wb") as f:
    #         pickle.dump(all_samples, f)
            
    #     metadata = {
    #         "num_samples": len(all_samples),
    #         "model": model_name,
    #         "model_type": extractor.model_type,
    #         "num_layers": extractor.num_layers,
    #         "hidden_dim": extractor.hidden_dim,
    #         "layers_extracted": list(layer_indices.keys()),
    #         "layer_indices": layer_indices,
    #         "batch_size": batch_size,
    #         "sample_keys": list(all_samples[0].keys()) if all_samples else [],
    #         "sparse_level": sparse_level,
    #         "num_classes": 8 if task == 'color' else None,
    #     }
        
    #     with open(os.path.join(output_dir, "metadata.json"), "w") as f:
    #         json.dump(metadata, f, indent=2)

    #     print(f"\nDataset saved to: {output_dir}")
    #     print("Files created:")
    #     print("  - complete_dataset.pkl")
    #     print("  - metadata.json")
    #     print("="*60)

    return all_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run probing clevr dataset creation")

    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf",
                        help="Model checkpoint name (HuggingFace repo ID or path)")
    parser.add_argument("--num_samples", type=int, default=2500,
                        help="Number of samples from CLEVR val set")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for model forward")
    parser.add_argument("--sparse_level", type=float, default=0.9,
                        help="Quantile threshold for sparsifying correlation graph")
    parser.add_argument("--task", type=str, default="color", choices=['color', 'counting', 'existence', 'comparison'],
                        help="Type of VQA task to filter on")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device string, e.g., 'cuda:0' or 'cpu'")
    parser.add_argument("--output_dir", type=str, default="probing_dataset",
                        help="Output directory for results")
    parser.add_argument("--verbose", action="store_true",
                        help="Print extra info for first sample")
    parser.add_argument("--log_every", type=int, default=5,
                        help="Log progress every N batches")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Creating dataset with model={args.model_name}, task={args.task}")
    print(f"{'='*60}")

    dataset = create_clevr_dataset(
        num_samples=args.num_samples,
        model_name=args.model_name,
        output_dir=args.output_dir,
        sparse_level=args.sparse_level,
        task=args.task,
        verbose=args.verbose,
        device=args.device,
        batch_size=args.batch_size,
        log_every=args.log_every,
    )
