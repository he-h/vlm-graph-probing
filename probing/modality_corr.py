import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from tqdm import tqdm
import os, argparse

from utils import model_ckpt2name
from model import NeuronGraphExtractor as GraphExtractor
from model import compute_corr_matrix
from dataset import prepare_vlm_data


def compute_modality_correlations(
    dataset,
    num_samples=1000,
    model_ckpt="llava-hf/llava-1.5-7b-hf",
    device="cuda:0",
    category="color",
    output_dir="results/modality_correlation",
    data_root=None,
    tdiuc_root=None,
    coco_val_root=None,
):
    """
    Compute token-token correlations within and across modalities for all layers.
    
    For each layer, computes:
    - Visual-Visual correlation: correlation between image tokens
    - Text-Text correlation: correlation between text tokens  
    - Visual-Text correlation: cross-correlation between image and text tokens
    
    Returns per-sample correlations and computes mean + std across samples.
    """
    model_prefix = model_ckpt2name(model_ckpt)
    
    # ---- Load data ----
    print("=" * 60)
    print(f"Preparing {dataset} samples, category={category}")
    data = prepare_vlm_data(
        dataset=dataset, num_samples=num_samples, category=category, balance=True,
        data_root=data_root, tdiuc_root=tdiuc_root, coco_val_root=coco_val_root,
    )
    if not data:
        print("ERROR: No samples loaded!")
        return

    print("=" * 60)
    print(f"Initializing {model_ckpt} model...")
    extractor = GraphExtractor(model_ckpt=model_ckpt, device=device)
    num_layers = extractor.num_layers
    
    print(f"Computing correlations for all {num_layers} layers...")
    
    # Storage for per-sample correlations
    # Each will be a list of length num_samples, where each element is the correlation value for that sample
    visual_visual_all = {L: [] for L in range(num_layers)}
    text_text_all = {L: [] for L in range(num_layers)}
    visual_text_all = {L: [] for L in range(num_layers)}
    
    # ---- Main loop ----
    for sample_idx, (image, prompt, answer) in enumerate(tqdm(data, desc="Processing samples", disable=True)):
        try:
            hidden_states_all, gen, [image_token_start, text_token_start] = extractor.process_single(image, prompt)
            hidden_states_all = [h.detach() for h in hidden_states_all]
            
            # Process each layer
            for L in range(num_layers):
                hs = hidden_states_all[L][0]  # [seq, hidden]
                
                # Split into visual and text tokens
                visual_tokens = hs[image_token_start:text_token_start, :]  # [n_visual, hidden]
                text_tokens = hs[text_token_start:, :]  # [n_text, hidden]
                
                # Compute token-level correlations by transposing
                # Original: [seq, hidden] -> correlation gives [hidden, hidden] (neuron-neuron)
                # Transposed: [hidden, seq] -> correlation gives [seq, seq] (token-token)
                
                # 1. Visual-Visual: correlation between visual tokens
                if visual_tokens.shape[0] > 1:
                    visual_corr_matrix = compute_corr_matrix(visual_tokens.T)  # [n_visual, n_visual]
                    # Take mean of off-diagonal elements (exclude diagonal which is always 1)
                    mask = ~torch.eye(visual_corr_matrix.shape[0], dtype=bool, device=visual_corr_matrix.device)
                    vv_corr = visual_corr_matrix[mask].mean().item()
                else:
                    vv_corr = 0.0
                visual_visual_all[L].append(vv_corr)
                
                # 2. Text-Text: correlation between text tokens
                if text_tokens.shape[0] > 1:
                    text_corr_matrix = compute_corr_matrix(text_tokens.T)  # [n_text, n_text]
                    mask = ~torch.eye(text_corr_matrix.shape[0], dtype=bool, device=text_corr_matrix.device)
                    tt_corr = text_corr_matrix[mask].mean().item()
                else:
                    tt_corr = 0.0
                text_text_all[L].append(tt_corr)
                
                # 3. Visual-Text: cross-correlation
                if visual_tokens.shape[0] > 0 and text_tokens.shape[0] > 0:
                    # Concatenate and compute full correlation matrix
                    all_tokens = torch.cat([visual_tokens, text_tokens], dim=0)  # [n_visual+n_text, hidden]
                    full_corr_matrix = compute_corr_matrix(all_tokens.T)  # [n_visual+n_text, n_visual+n_text]
                    
                    # Extract cross-correlation block
                    n_visual = visual_tokens.shape[0]
                    n_text = text_tokens.shape[0]
                    cross_corr_block = full_corr_matrix[:n_visual, n_visual:]  # [n_visual, n_text]
                    vt_corr = cross_corr_block.mean().item()
                else:
                    vt_corr = 0.0
                visual_text_all[L].append(vt_corr)
            
            del hidden_states_all
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing sample {sample_idx}: {str(e)}")
            continue
    
    # ---- Compute statistics ----
    print(f"\n{'='*60}")
    print(f"Computing statistics across {len(data)} samples")
    print(f"{'='*60}")
    
    # Convert to numpy arrays and compute mean/std
    vv_mean = np.zeros(num_layers)
    vv_std = np.zeros(num_layers)
    tt_mean = np.zeros(num_layers)
    tt_std = np.zeros(num_layers)
    vt_mean = np.zeros(num_layers)
    vt_std = np.zeros(num_layers)
    
    for L in range(num_layers):
        vv_array = np.array(visual_visual_all[L])
        tt_array = np.array(text_text_all[L])
        vt_array = np.array(visual_text_all[L])
        
        vv_mean[L] = vv_array.mean()
        vv_std[L] = vv_array.std()
        tt_mean[L] = tt_array.mean()
        tt_std[L] = tt_array.std()
        vt_mean[L] = vt_array.mean()
        vt_std[L] = vt_array.std()
        
        if L % 5 == 0 or L == num_layers - 1:
            print(f"Layer {L:2d}: VV={vv_mean[L]:.4f}±{vv_std[L]:.4f}, "
                  f"TT={tt_mean[L]:.4f}±{tt_std[L]:.4f}, "
                  f"VT={vt_mean[L]:.4f}±{vt_std[L]:.4f}")
    
    # ---- Save results as single .npy file ----
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/modality_corr_{model_prefix}_{dataset}_{category}.npy"
    
    results = {
        'vv_mean': vv_mean,
        'vv_std': vv_std,
        'tt_mean': tt_mean,
        'tt_std': tt_std,
        'vt_mean': vt_mean,
        'vt_std': vt_std,
        'num_layers': np.array(num_layers),
        'num_samples': np.array(len(data)),
    }
    
    np.savez(output_file, **results)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute modality correlations across layers")
    parser.add_argument("--dataset", type=str, default="clevr")
    parser.add_argument("--model_ckpt", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--category", type=str, default="color")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="results/modality_correlation")
    parser.add_argument("--data_root", type=str, default=None, help="Base data directory.")
    parser.add_argument("--tdiuc_root", type=str, default=None, help="Path to TDIUC dataset root.")
    parser.add_argument("--coco_val_root", type=str, default=None, help="Path to COCO val2014 images.")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Modality Correlation Analysis")
    print(f"Model: {args.model_ckpt}")
    print(f"Dataset: {args.dataset}")
    print(f"Category: {args.category}")
    print(f"{'='*60}\n")
    
    compute_modality_correlations(
        dataset=args.dataset,
        num_samples=args.num_samples,
        model_ckpt=args.model_ckpt,
        device=args.device,
        category=args.category,
        output_dir=args.output_dir,
        data_root=args.data_root,
        tdiuc_root=args.tdiuc_root,
        coco_val_root=args.coco_val_root,
    )
