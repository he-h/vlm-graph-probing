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


def create_coco_dataset(
    num_samples=1000,
    model_name="llava-hf/llava-1.5-7b-hf",
    output_dir="probing_dataset",
    prompt_choice=0,
    verbose=False,
    device="cuda:0",
    batch_size=8,
    sparse_level=0.9,
    log_every=5,
):
    """
    Create graph probing dataset for VLM with COCO captions (optimized batched processing).
    """
    model_prefix = model_path2name(model_name)
    output_dir = f"data/{model_prefix}_prompt_{prompt_choice}_sparsity_{int(sparse_level * 100)}_{output_dir}"
    os.makedirs(output_dir, exist_ok=True)

    # Load COCO dataset
    print("="*60)
    print("Loading COCO dataset from HuggingFace...")
    coco_samples = load_dataset("lmms-lab/COCO-Caption2017", split='val')
    if not coco_samples:
        print("ERROR: Failed to load COCO dataset from HuggingFace!")
        return []

    if num_samples:
        coco_samples = coco_samples.select(range(min(num_samples, len(coco_samples))))

    print(f"Loaded {len(coco_samples)} samples from COCO 2017 validation set")
    print("="*60)

    print(f"Initializing {model_name} model...")
    extractor = GraphExtractor(model_name=model_name, device=device)

    layer_indices = {
        'layer_0': 0,
        'layer_middle': extractor.num_layers // 2,
        'layer_last': extractor.num_layers - 1 
    }

    all_samples = []
    missing_captions = 0
    graph_extraction_prompt = caption_prompt(prompt_choice)
    print(f"Using graph extraction prompt:\n{graph_extraction_prompt}")
    print(f"Batch size: {batch_size}")

    # Process in batches with progress bar
    total_batches = (len(coco_samples) + batch_size - 1) // batch_size

    all_preds = []
    all_refs  = []
    
    for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(coco_samples))
        batch = coco_samples[start_idx:end_idx]   # returns a dict
        images = batch["image"]                   # list of PIL Images
        references = batch["answer"]              # list of refs (answers/captions)
        valid_indices = []
        
        for local_idx, (img, refs) in enumerate(zip(images, references)):
            if img.mode != "RGB":
                images[local_idx] = img.convert("RGB")
                print("Converted image to RGB mode")
            valid_indices.append(start_idx + local_idx)

        if not images:
            continue

        try:
            graph_extraction_prompts = [graph_extraction_prompt] * len(images)
            hidden_states_all, generated_captions = extractor.process(
                images, graph_extraction_prompts
            )
            # print(generated_captions)
            # for ref in references:
            #     print(ref)
            # print("-" * 40)
            
            # Process each item in batch
            for b_idx, (refs, gen_caption, global_idx) in enumerate(
                zip(references, generated_captions, valid_indices)
            ):
                if not gen_caption:
                    missing_captions += 1
                
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
                    "meteor_score": None,
                    "spice_score": None,
                    "rouge_l_score": None,
                    "bertscore_f1": None,
                }
                all_samples.append(sample)

                all_preds.append(gen_caption)
                all_refs.append(refs)

            if batch_idx % log_every == 0:
                print(f"Processed sample {global_idx} Number of captions: {len(refs)}")
                for i, ref in enumerate(refs):
                    print(f"  Caption {i+1}: {ref[:80]}")
                print(f"  Generated: {gen_caption[:80]}")
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {str(e)}")
            # print(f"Generated cations so far: {generated_captions}")
            # print(f"References: {references}")
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
        print(f"Missing captions: {missing_captions}")

        preds_s, refs_s = sanitize_preds_refs(all_preds, all_refs)

        import time
        start_time = time.time()
        M = meteor_scores(preds_s, refs_s)         # fast (python)
        print(f"METEOR computation time: {(time.time() - start_time)/60:.2f} minutes")
        start_time = time.time()
        R = rougeL_scores(preds_s, refs_s)         # fast
        print(f"ROUGE-L computation time: {(time.time() - start_time)/60:.2f} minutes")
        start_time = time.time()
        B = bertscore_f1(preds_s, refs_s)          # heavy but 1x init
        print(f"BERTScore computation time: {(time.time() - start_time)/60:.2f} minutes")
        start_time = time.time()
        S = spice_scores(preds_s, refs_s)          # slowest (Java) but 1x run
        print(f"SPICE computation time: {(time.time() - start_time)/60:.2f} minutes")

        for i in range(len(all_samples)):
            all_samples[i]["meteor_score"]  = M[i]
            all_samples[i]["rouge_l_score"] = R[i]
            all_samples[i]["bertscore_f1"]  = B[i]
            all_samples[i]["spice_score"]   = S[i]

        avg_meteor = np.mean([s.get("meteor_score", 0) for s in all_samples])
        avg_spice = np.mean([s.get("spice_score", 0) for s in all_samples])
        avg_rouge = np.mean([s.get("rouge_l_score", 0) for s in all_samples])
        avg_bertscore = np.mean([s.get("bertscore_f1", 0) for s in all_samples])

        print(f"Average METEOR: {avg_meteor:.4f}")
        print(f"Average SPICE: {avg_spice:.4f}")
        print(f"Average ROUGE-L: {avg_rouge:.4f}")
        print(f"Average BERTScore F1: {avg_bertscore:.4f}")
        
        # Verify data structure
        sample_check = all_samples[0]
        print(f"\nSample structure verification:")
        print(f"  Keys in sample: {list(sample_check.keys())}")
        print(f"  Graph layer 0 keys: {list(sample_check['graph_layer_0'].keys())}")
        print(f"  Last token state shape: {sample_check['last_token_state'].shape}")
        print(f"  METEOR: {sample_check.get('meteor_score', 0.0):.4f}")
        print(f"  SPICE: {sample_check.get('spice_score', 0.0):.4f}")
        print(f"  ROUGE-L: {sample_check.get('rouge_l_score', 0.0):.4f}")
        print(f"  BERTScore: {sample_check.get('bertscore_f1', 0.0):.4f}")
        # print sample edge index and weights for layer 0, first 5 edges
        edge_index = sample_check['graph_layer_0']['edge_index']
        edge_weight = sample_check['graph_layer_0']['edge_weight']
        print(f"  Graph layer 0 edges (first 5): {edge_index[:, :5]}")
        print(f"  Graph layer 0 weights (first 5): {edge_weight[:5]}")

        # Save results
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
            "prompt_choice": prompt_choice,
            "prompt_used": graph_extraction_prompt,
            "avg_meteor": avg_meteor,
            "avg_spice": avg_spice,
            "avg_rouge_l": avg_rouge,
            "avg_bertscore_f1": avg_bertscore,
            "batch_size": batch_size,
            "sample_keys": list(all_samples[0].keys()) if all_samples else [],
            "sparse_level": sparse_level
        }
        
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nDataset saved to: {output_dir}")
        print("Files created:")
        print("  - complete_dataset.pkl")
        print("  - metadata.json")
        print("="*60)

    return all_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run probing caption coco dataset creation")

    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf",
                        help="Model checkpoint name (HuggingFace repo ID or path)")
    parser.add_argument("--prompt_choice", type=int, default=0, choices=[0, 1, 2],
                        help="Prompt index (0, 1, or 2)")
    parser.add_argument("--num_samples", type=int, default=2500,
                        help="Number of samples from COCO val set")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for model forward")
    parser.add_argument("--sparse_level", type=float, default=0.9,
                        help="Quantile threshold for sparsifying correlation graph")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device string, e.g., 'cuda:0' or 'cpu'")
    parser.add_argument("--output_dir", type=str, default="probing_dataset",
                        help="Output directory for results")
    parser.add_argument("--verbose", action="store_true",
                        help="Print extra info for first sample")
    parser.add_argument("--log_every", type=int, default=25,
                        help="Log progress every N batches")

    # TODO: three layers vs all layers

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Creating dataset with model={args.model_name}, prompt={args.prompt_choice}")
    print(f"{'='*60}")

    dataset = create_coco_dataset(
        num_samples=args.num_samples,
        model_name=args.model_name,
        output_dir=args.output_dir,
        sparse_level=args.sparse_level,
        prompt_choice=args.prompt_choice,
        verbose=args.verbose,
        device=args.device,
        batch_size=args.batch_size,
        log_every=args.log_every,
    )
