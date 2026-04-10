"""
Edge-based intervention experiment for VLMs.

This script:
1. First pass: Accumulate edge importance (sum of |correlation| across samples)
2. Second pass: Intervene on top-k edges by forcing specific correlations:
   - corr=1: Make neuron_j identical to neuron_i
   - corr=-1: Make neuron_j opposite of neuron_i  
   - corr=0: Make neuron_j random (uncorrelated with neuron_i)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from tqdm import tqdm
import os
import json
import argparse

from utils import model_ckpt2name, evenly_spaced_layers, get_blocks
from model import NeuronGraphExtractor as GraphExtractor
from model import compute_corr_matrix
from dataset import prepare_vlm_data, get_candidate_answers


# -------------------------------
# Edge Intervention Hook
# -------------------------------

class EdgeInterventionHook:
    """
    Hook to intervene on specific neuron pairs (edges) by forcing correlation.
    
    Intervention types:
    - 'identical' (corr=1): neuron_target = neuron_source
    - 'opposite' (corr=-1): neuron_target = -neuron_source
    - 'random' (corr=0): neuron_target = random noise with same mean/std
    """
    
    def __init__(self, edge_pairs, intervention_type='identical', direction='i_to_j', seed=42):
        """
        Args:
            edge_pairs: List of (neuron_i, neuron_j) tuples to intervene on
            intervention_type: 'identical', 'opposite', or 'random'
            direction: 'i_to_j' (fix i, intervene j) or 'j_to_i' (fix j, intervene i)
            seed: Random seed for 'random' intervention
        """
        self.edge_pairs = edge_pairs
        self.intervention_type = intervention_type
        self.direction = direction
        self.rng = np.random.default_rng(seed)
        self.handle = None
    
    def hook_fn(self, module, input, output):
        """Hook function that modifies hidden states to force correlations."""
        # Extract hidden states tensor
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        # hidden_states shape: [batch, seq, hidden]
        if len(hidden_states.shape) != 3:
            return output
        
        # Apply intervention for each edge pair
        for neuron_i, neuron_j in self.edge_pairs:
            # Determine source (fixed) and target (intervened) based on direction
            if self.direction == 'i_to_j':
                neuron_source = neuron_i
                neuron_target = neuron_j
            else:  # 'j_to_i'
                neuron_source = neuron_j
                neuron_target = neuron_i
            
            if self.intervention_type == 'identical':
                # Force correlation = 1: make target identical to source
                hidden_states[:, :, neuron_target] = hidden_states[:, :, neuron_source].clone()
            
            elif self.intervention_type == 'opposite':
                # Force correlation = -1: make target opposite of source
                hidden_states[:, :, neuron_target] = -hidden_states[:, :, neuron_source].clone()
            
            elif self.intervention_type == 'random':
                # Force correlation ≈ 0: replace target with random noise
                # Match the statistics (mean, std) of the original activation
                original = hidden_states[:, :, neuron_target]
                mean_val = original.mean().item()
                std_val = original.std().item() + 1e-6
                
                # Generate random tensor with same shape and statistics
                random_vals = torch.randn_like(original) * std_val + mean_val
                hidden_states[:, :, neuron_target] = random_vals
        
        # Return in same format
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        else:
            return hidden_states
    
    def register(self, layer):
        """Register this hook to a layer."""
        self.handle = layer.register_forward_hook(self.hook_fn)
    
    def remove(self):
        """Remove the hook."""
        if self.handle is not None:
            self.handle.remove()


# -------------------------------
# Pass 1: Accumulate edge importance
# -------------------------------

def accumulate_edge_importance(
    dataset,
    num_samples=500,
    model_ckpt="llava-hf/llava-1.5-7b-hf",
    device="cuda:0",
    category="color",
    layer_indices=None,
    layer_slices=4,
    top_k_edges=100,
):
    """
    First pass: Run through samples and accumulate edge importance.
    
    Edge importance = sum of |correlation weight| across all samples.
    
    Returns:
        dict: {layer: [(neuron_i, neuron_j, total_weight), ...]} sorted by importance
    """
    print("=" * 60)
    print("PASS 1: Accumulating Edge Importance")
    print("=" * 60)
    
    # Load data
    data = prepare_vlm_data(dataset=dataset, num_samples=num_samples, category=category, balance=True)
    if not data:
        raise ValueError("No samples loaded!")
    print(f"Loaded {len(data)} samples", flush=True)
    
    # Initialize model
    print(f"Loading model {model_ckpt}... (this may take a minute)", flush=True)
    extractor = GraphExtractor(model_ckpt=model_ckpt, device=device)
    print("Model loaded!", flush=True)
    num_layers = extractor.num_layers
    
    # Determine layers
    selected_layers = layer_indices if layer_indices else evenly_spaced_layers(num_layers, layer_slices)
    print(f"Selected layers: {selected_layers}")
    
    # Get hidden dim from model
    hidden_dim = extractor.hidden_dim
    print(f"Hidden dim: {hidden_dim}")
    
    # Storage: accumulated |correlation| matrix per layer (on GPU for speed)
    accumulated_corr = {L: torch.zeros(hidden_dim, hidden_dim, device=device) for L in selected_layers}
    
    # Process samples
    for sample_idx, (image, prompt, answer) in enumerate(tqdm(data, desc="Accumulating edges")):
        try:
            hidden_states_all, gen, [img_start, txt_start] = extractor.process_single(image, prompt)
            
            for L in selected_layers:
                hs = hidden_states_all[L][0].detach()  # [seq, hidden]
                
                # Compute correlation matrix
                corr_matrix = compute_corr_matrix(hs)  # [hidden, hidden]
                
                # Fast tensor accumulation of absolute correlations
                accumulated_corr[L] += corr_matrix.abs()
            
            del hidden_states_all
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")
            continue
    
    # Extract top-k edges from accumulated matrix (upper triangle only)
    top_edges = {}
    for L in selected_layers:
        acc_mat = accumulated_corr[L].cpu().numpy()
        
        # Get upper triangle indices (exclude diagonal)
        triu_indices = np.triu_indices(hidden_dim, k=1)
        weights = acc_mat[triu_indices]
        
        # Get top-k indices
        top_k_idx = np.argpartition(weights, -top_k_edges)[-top_k_edges:]
        top_k_idx = top_k_idx[np.argsort(-weights[top_k_idx])]  # Sort descending
        
        # Convert to (neuron_i, neuron_j, weight) tuples
        top_edges[L] = [
            (int(triu_indices[0][idx]), int(triu_indices[1][idx]), float(weights[idx]))
            for idx in top_k_idx
        ]
        
        print(f"\nLayer {L} top 5 edges:")
        for i, (n1, n2, w) in enumerate(top_edges[L][:5]):
            print(f"  {i+1}. ({n1}, {n2}): {w:.4f}")
    
    return top_edges, extractor, selected_layers


# -------------------------------
# Pass 2: Run intervention experiments
# -------------------------------

def run_edge_intervention(
    dataset,
    num_samples=500,
    model_ckpt="llava-hf/llava-1.5-7b-hf",
    device="cuda:0",
    category="color",
    layer_indices=None,
    layer_slices=4,
    top_k_edges=10,
    output_dir="results/edge_intervention",
):
    """
    Full experiment: accumulate edge importance then run interventions.
    """
    # Pass 1: Get important edges
    top_edges, extractor, selected_layers = accumulate_edge_importance(
        dataset=dataset,
        num_samples=num_samples,
        model_ckpt=model_ckpt,
        device=device,
        category=category,
        layer_indices=layer_indices,
        layer_slices=layer_slices,
        top_k_edges=top_k_edges,
    )
    
    # Reload data for pass 2
    data = prepare_vlm_data(dataset=dataset, num_samples=num_samples, category=category, balance=True)
    candidates = get_candidate_answers(dataset, category)
    
    print("\n" + "=" * 60)
    print("PASS 2: Running Edge Interventions")
    print("=" * 60)
    
    intervention_types = ['identical', 'opposite', 'random']
    directions = ['i_to_j', 'j_to_i']  # Test both directions
    results = {
        'metadata': {
            'model': model_ckpt,
            'dataset': dataset,
            'category': category,
            'num_samples': len(data),
            'top_k_edges': top_k_edges,
            'selected_layers': selected_layers,
        },
        'top_edges': {str(L): [(int(a), int(b), float(w)) for a, b, w in edges] 
                      for L, edges in top_edges.items()},
        'baseline': {},
        'interventions': {},
    }
    
    # Get model blocks
    blocks = get_blocks(extractor.model, extractor.model_family)
    
    # Run baseline first (no intervention)
    print("\n--- Running BASELINE ---")
    baseline_correct = 0
    baseline_total = 0
    
    for sample_idx, (image, prompt, answer) in enumerate(tqdm(data, desc="Baseline")):
        try:
            _, gen, logits, _ = extractor.process_single(image, prompt, max_new_tokens=1, output_logits=True)
            
            probs = torch.nn.functional.log_softmax(logits[:, -1, :], dim=-1)
            tok = extractor.processor.tokenizer
            ids = [tok(c, add_special_tokens=False)["input_ids"][0] for c in candidates]
            scores = [probs[0, i].item() for i in ids]
            pred = candidates[int(np.argmax(scores))]
            
            ref = answer.lower().strip() if isinstance(answer, str) else str(answer).lower().strip()
            baseline_correct += int(pred == ref)
            baseline_total += 1
            
        except Exception as e:
            continue
    
    baseline_acc = baseline_correct / baseline_total if baseline_total > 0 else 0
    results['baseline']['accuracy'] = baseline_acc
    print(f"Baseline Accuracy: {baseline_acc * 100:.2f}%")
    
    # Run interventions for each layer, direction, and type
    for L in selected_layers:
        results['interventions'][str(L)] = {}
        
        # Get edge pairs for this layer (just the neuron indices)
        edge_pairs = [(e[0], e[1]) for e in top_edges[L]]
        
        for direction in directions:
            results['interventions'][str(L)][direction] = {}
            
            for intervention_type in intervention_types:
                print(f"\n--- Layer {L}, Direction: {direction}, Intervention: {intervention_type} ---")
                
                correct = 0
                total = 0
                
                for sample_idx, (image, prompt, answer) in enumerate(tqdm(data, desc=f"L{L}-{direction}-{intervention_type}", disable=False)):
                    hook = None
                    try:
                        hook = EdgeInterventionHook(
                            edge_pairs, 
                            intervention_type=intervention_type, 
                            direction=direction,
                            seed=42 + sample_idx
                        )
                        hook.register(blocks[L])
                        
                        # Run inference
                        _, gen, logits, _ = extractor.process_single(image, prompt, max_new_tokens=1, output_logits=True)
                        
                        # Remove hook
                        hook.remove()
                        
                        # Compute prediction
                        probs = torch.nn.functional.log_softmax(logits[:, -1, :], dim=-1)
                        tok = extractor.processor.tokenizer
                        ids = [tok(c, add_special_tokens=False)["input_ids"][0] for c in candidates]
                        scores = [probs[0, i].item() for i in ids]
                        pred = candidates[int(np.argmax(scores))]
                        
                        ref = answer.lower().strip() if isinstance(answer, str) else str(answer).lower().strip()
                        correct += int(pred == ref)
                        total += 1
                        
                    except Exception as e:
                        if hook is not None:
                            hook.remove()
                        continue
                
                acc = correct / total if total > 0 else 0
                results['interventions'][str(L)][direction][intervention_type] = {
                    'accuracy': acc,
                    'correct': correct,
                    'total': total,
                    'delta_from_baseline': acc - baseline_acc,
                }
                print(f"Accuracy: {acc * 100:.2f}% (Δ = {(acc - baseline_acc) * 100:+.2f}%)")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    model_name = model_ckpt2name(model_ckpt)
    output_file = f"{output_dir}/edge_intervention_{model_name}_{dataset}_{category}_top{top_k_edges}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Baseline: {baseline_acc * 100:.2f}%")
    for L in selected_layers:
        print(f"\nLayer {L}:")
        for direction in directions:
            print(f"  Direction: {direction}")
            for itype in intervention_types:
                acc = results['interventions'][str(L)][direction][itype]['accuracy']
                delta = results['interventions'][str(L)][direction][itype]['delta_from_baseline']
                print(f"    {itype:10s}: {acc * 100:.2f}% (Δ = {delta * 100:+.2f}%)")
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge-based intervention experiment")
    parser.add_argument("--dataset", type=str, default="clevr")
    parser.add_argument("--model_ckpt", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--category", type=str, default="color")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--layer_slices", type=int, default=4)
    parser.add_argument("--layer_indices", nargs="+", type=int, default=None)
    parser.add_argument("--top_k_edges", type=int, default=10,
                       help="Number of top edges to intervene on per layer")
    parser.add_argument("--output_dir", type=str, default="results/edge_intervention")
    
    args = parser.parse_args()
    
    print(f"\n{'=' * 60}")
    print("Edge-Based Intervention Experiment")
    print(f"Model: {args.model_ckpt}")
    print(f"Dataset: {args.dataset}")
    print(f"Category: {args.category}")
    print(f"Top-k edges: {args.top_k_edges}")
    print(f"{'=' * 60}\n")
    
    run_edge_intervention(
        dataset=args.dataset,
        num_samples=args.num_samples,
        model_ckpt=args.model_ckpt,
        device=args.device,
        category=args.category,
        layer_slices=args.layer_slices,
        layer_indices=args.layer_indices,
        top_k_edges=args.top_k_edges,
        output_dir=args.output_dir,
    )
