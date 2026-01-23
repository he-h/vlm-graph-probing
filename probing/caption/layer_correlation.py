import torch
import numpy as np
from transformers import LlavaForConditionalGeneration, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from datasets import load_dataset
from PIL import Image
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import argparse

from utils import *
from model import CaptionGraphExtractor as GraphExtractor


def analyze_correlations(
    num_samples=50,
    model_name="llava-hf/llava-1.5-7b-hf",
    prompt_choice=0,
    device="cuda:0",
):
    """
    Analyze token correlations across layers for VLM.
    """
    # Load COCO dataset
    print("="*60)
    print("Loading COCO dataset from HuggingFace...")
    coco_samples = load_dataset("lmms-lab/COCO-Caption2017", split='val')
    coco_samples = coco_samples.select(range(min(num_samples, len(coco_samples))))
    print(f"Loaded {len(coco_samples)} samples")
    print("="*60)

    print(f"Initializing {model_name} model...")
    extractor = GraphExtractor(model_name=model_name, device=device)
    
    graph_extraction_prompt = caption_prompt(prompt_choice)
    print(f"Using prompt: {graph_extraction_prompt}")
    
    # Get number of prompt tokens
    prompt_tokens = extractor.processor.tokenizer(
        graph_extraction_prompt,
        return_tensors="pt"
    )
    num_prompt_tokens = prompt_tokens['input_ids'].shape[1]
    
    # Determine number of visual tokens based on model type
    if extractor.model_family == "llava":
        num_visual_tokens = 576  # Standard for ViT-L/14
    elif extractor.model_family == "qwen2_5_vl":
        num_visual_tokens = 256  # Adjust based on your model
    else:
        num_visual_tokens = 256
    
    # Process one sample first to get actual number of layers
    img = coco_samples[0]["image"]
    if img.mode != "RGB":
        img = img.convert("RGB")
    hidden_states_test, _ = extractor.process([img], graph_extraction_prompt)
    actual_num_layers = len(hidden_states_test)
    del hidden_states_test
    torch.cuda.empty_cache()
    
    print(f"Model type: {extractor.model_family}")
    print(f"Model.num_layers: {extractor.num_layers}")
    print(f"Actual hidden states layers: {actual_num_layers}")
    print(f"Expected visual tokens: {num_visual_tokens}")
    print(f"Prompt tokens: {num_prompt_tokens}")
    
    # Store correlations for all samples
    all_correlations = {
        'visual_visual': [],
        'prompt_prompt': [],
        'visual_prompt': []
    }
    
    # Process each sample
    for idx in tqdm(range(len(coco_samples)), desc="Processing samples"):
        img = coco_samples[idx]["image"]
        
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        try:
            # Process single image
            hidden_states_all, _ = extractor.process(
                [img], graph_extraction_prompt
            )
            
            # Calculate correlations for each layer
            sample_correlations = {
                'visual_visual': [],
                'prompt_prompt': [],
                'visual_prompt': []
            }
            
            # Use actual number of hidden state layers
            num_layers_to_process = len(hidden_states_all)
            
            for layer_idx in range(num_layers_to_process):
                hs = hidden_states_all[layer_idx][0].detach().cpu().numpy()  # [0] for first batch item
                
                # Ensure we don't exceed sequence length
                seq_len = hs.shape[0]
                actual_visual_tokens = min(num_visual_tokens, seq_len)
                actual_prompt_end = min(actual_visual_tokens + num_prompt_tokens, seq_len)
                
                # Split tokens
                visual_tokens = hs[:actual_visual_tokens]
                prompt_tokens = hs[actual_visual_tokens:actual_prompt_end]
                
                # Visual-Visual correlation
                if len(visual_tokens) > 1:
                    visual_corr = np.corrcoef(visual_tokens)
                    visual_corr_mean = np.mean(np.abs(visual_corr[np.triu_indices_from(visual_corr, k=1)]))
                else:
                    visual_corr_mean = 0.0
                
                # Prompt-Prompt correlation
                if len(prompt_tokens) > 1:
                    prompt_corr = np.corrcoef(prompt_tokens)
                    prompt_corr_mean = np.mean(np.abs(prompt_corr[np.triu_indices_from(prompt_corr, k=1)]))
                else:
                    prompt_corr_mean = 0.0
                
                # Visual-Prompt cross-correlation
                if len(visual_tokens) > 0 and len(prompt_tokens) > 0:
                    combined = np.vstack([visual_tokens, prompt_tokens])
                    full_corr = np.corrcoef(combined)
                    cross_corr_block = full_corr[:len(visual_tokens), len(visual_tokens):]
                    visual_prompt_corr_mean = np.mean(np.abs(cross_corr_block))
                else:
                    visual_prompt_corr_mean = 0.0
                
                sample_correlations['visual_visual'].append(visual_corr_mean)
                sample_correlations['prompt_prompt'].append(prompt_corr_mean)
                sample_correlations['visual_prompt'].append(visual_prompt_corr_mean)
            
            # Add to all samples
            all_correlations['visual_visual'].append(sample_correlations['visual_visual'])
            all_correlations['prompt_prompt'].append(sample_correlations['prompt_prompt'])
            all_correlations['visual_prompt'].append(sample_correlations['visual_prompt'])
            
            # Print progress
            if idx % 10 == 0:
                print(f"\nSample {idx}:")
                print(f"  Layers processed: {len(sample_correlations['visual_visual'])}")
                print(f"  First layer - V-V: {sample_correlations['visual_visual'][0]:.3f}, "
                      f"P-P: {sample_correlations['prompt_prompt'][0]:.3f}, "
                      f"V-P: {sample_correlations['visual_prompt'][0]:.3f}")
                print(f"  Last layer - V-V: {sample_correlations['visual_visual'][-1]:.3f}, "
                      f"P-P: {sample_correlations['prompt_prompt'][-1]:.3f}, "
                      f"V-P: {sample_correlations['visual_prompt'][-1]:.3f}")
                
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            continue
        
        # Clear memory
        del hidden_states_all
        torch.cuda.empty_cache()
    
    # Calculate averages across samples
    avg_correlations = {
        'visual_visual': np.mean(all_correlations['visual_visual'], axis=0),
        'prompt_prompt': np.mean(all_correlations['prompt_prompt'], axis=0),
        'visual_prompt': np.mean(all_correlations['visual_prompt'], axis=0)
    }
    
    std_correlations = {
        'visual_visual': np.std(all_correlations['visual_visual'], axis=0),
        'prompt_prompt': np.std(all_correlations['prompt_prompt'], axis=0),
        'visual_prompt': np.std(all_correlations['visual_prompt'], axis=0)
    }
    
    # Use actual number of layers for plotting
    actual_layers = len(avg_correlations['visual_visual'])
    
    # Plot results
    plot_correlations(avg_correlations, std_correlations, actual_layers, model_name)
    
    return avg_correlations, std_correlations


def plot_correlations(avg_correlations, std_correlations, num_layers, model_name):
    """
    Plot correlation trends across layers.
    """
    layers = list(range(num_layers))
    
    # Verify dimensions match
    print(f"\nPlotting {num_layers} layers")
    print(f"Visual-Visual shape: {len(avg_correlations['visual_visual'])}")
    print(f"Prompt-Prompt shape: {len(avg_correlations['prompt_prompt'])}")
    print(f"Visual-Prompt shape: {len(avg_correlations['visual_prompt'])}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Visual-Visual correlation
    axes[0].plot(layers, avg_correlations['visual_visual'], 'b-', linewidth=2, label='Mean')
    axes[0].fill_between(layers, 
                         avg_correlations['visual_visual'] - std_correlations['visual_visual'],
                         avg_correlations['visual_visual'] + std_correlations['visual_visual'],
                         alpha=0.3, color='blue')
    axes[0].set_xlabel('Layer Index')
    axes[0].set_ylabel('Correlation')
    axes[0].set_title('Visual-Visual Token Correlation')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Prompt-Prompt correlation
    axes[1].plot(layers, avg_correlations['prompt_prompt'], 'g-', linewidth=2, label='Mean')
    axes[1].fill_between(layers,
                         avg_correlations['prompt_prompt'] - std_correlations['prompt_prompt'],
                         avg_correlations['prompt_prompt'] + std_correlations['prompt_prompt'],
                         alpha=0.3, color='green')
    axes[1].set_xlabel('Layer Index')
    axes[1].set_ylabel('Correlation')
    axes[1].set_title('Prompt-Prompt Token Correlation')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    # Visual-Prompt correlation
    axes[2].plot(layers, avg_correlations['visual_prompt'], 'r-', linewidth=2, label='Mean')
    axes[2].fill_between(layers,
                         avg_correlations['visual_prompt'] - std_correlations['visual_prompt'],
                         avg_correlations['visual_prompt'] + std_correlations['visual_prompt'],
                         alpha=0.3, color='red')
    axes[2].set_xlabel('Layer Index')
    axes[2].set_ylabel('Correlation')
    axes[2].set_title('Visual-Prompt Cross Correlation')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 1])
    
    model_short = model_name.split('/')[-1]
    plt.suptitle(f'Token Correlations Across Layers - {model_short}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_name = f"correlation_analysis_{model_short.replace('-', '_')}.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved as {output_name}")
    plt.show()
    
    # Also create a combined plot
    plt.figure(figsize=(10, 6))
    plt.plot(layers, avg_correlations['visual_visual'], 'b-', linewidth=2, label='Visual-Visual')
    plt.plot(layers, avg_correlations['prompt_prompt'], 'g-', linewidth=2, label='Prompt-Prompt')
    plt.plot(layers, avg_correlations['visual_prompt'], 'r-', linewidth=2, label='Visual-Prompt')
    
    plt.fill_between(layers,
                     avg_correlations['visual_visual'] - std_correlations['visual_visual'],
                     avg_correlations['visual_visual'] + std_correlations['visual_visual'],
                     alpha=0.2, color='blue')
    plt.fill_between(layers,
                     avg_correlations['prompt_prompt'] - std_correlations['prompt_prompt'],
                     avg_correlations['prompt_prompt'] + std_correlations['prompt_prompt'],
                     alpha=0.2, color='green')
    plt.fill_between(layers,
                     avg_correlations['visual_prompt'] - std_correlations['visual_prompt'],
                     avg_correlations['visual_prompt'] + std_correlations['visual_prompt'],
                     alpha=0.2, color='red')
    
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Average Correlation', fontsize=12)
    plt.title(f'All Token Correlations Across Layers - {model_short}', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    output_name_combined = f"correlation_analysis_combined_{model_short.replace('-', '_')}.png"
    plt.savefig(output_name_combined, dpi=300, bbox_inches='tight')
    print(f"Combined figure saved as {output_name_combined}")
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("CORRELATION SUMMARY")
    print("="*60)
    print(f"Visual-Visual: First layer={avg_correlations['visual_visual'][0]:.3f}, "
          f"Last layer={avg_correlations['visual_visual'][-1]:.3f}")
    print(f"Prompt-Prompt: First layer={avg_correlations['prompt_prompt'][0]:.3f}, "
          f"Last layer={avg_correlations['prompt_prompt'][-1]:.3f}")
    print(f"Visual-Prompt: First layer={avg_correlations['visual_prompt'][0]:.3f}, "
          f"Last layer={avg_correlations['visual_prompt'][-1]:.3f}")
    
    # Calculate trends
    vv_trend = avg_correlations['visual_visual'][-1] - avg_correlations['visual_visual'][0]
    pp_trend = avg_correlations['prompt_prompt'][-1] - avg_correlations['prompt_prompt'][0]
    vp_trend = avg_correlations['visual_prompt'][-1] - avg_correlations['visual_prompt'][0]
    
    print(f"\nTrends (Last - First):")
    print(f"  Visual-Visual: {vv_trend:+.3f}")
    print(f"  Prompt-Prompt: {pp_trend:+.3f}")
    print(f"  Visual-Prompt: {vp_trend:+.3f}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze token correlations in VLM")

    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf",
                        help="Model checkpoint name")
    parser.add_argument("--prompt_choice", type=int, default=0, choices=[0, 1, 2],
                        help="Prompt index (0, 1, or 2)")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of samples to analyze")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device string")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Analyzing correlations for {args.model_name}")
    print(f"Using {args.num_samples} samples with prompt {args.prompt_choice}")
    print(f"{'='*60}")

    avg_corr, std_corr = analyze_correlations(
        num_samples=args.num_samples,
        model_name=args.model_name,
        prompt_choice=args.prompt_choice,
        device=args.device
    )