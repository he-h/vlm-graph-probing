import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from tqdm import tqdm
import pickle, os, json, argparse
from collections import Counter

from utils import model_ckpt2name, evenly_spaced_layers
from model import NeuronGraphExtractor as GraphExtractor
from model import compute_corr_matrix, node_degrees
from dataset import prepare_vlm_data 


def analyze_hub_neurons(
    dataset,
    num_samples=1000,
    model_ckpt="llava-hf/llava-1.5-7b-hf",
    verbose=False,
    device="cuda:0",
    category="color",
    log_every=200,
    layer_slices=4,
    layer_indices=None,
    top_k=10  # Number of top neurons to track per sample
):
    """
    Analyze hub neurons across layers and save frequency counts to JSON.
    
    For each sample, track top-k neurons based on three criteria:
    1. Degree from full hidden state correlation
    2. Degree from text-only hidden state correlation
    3. Last token activation magnitude
    """
    model_prefix = model_ckpt2name(model_ckpt)

    # ---- Load data ----
    print("=" * 60)
    print(f"Preparing {dataset} samples, category={category}")
    data = prepare_vlm_data(dataset=dataset, num_samples=num_samples, category=category, balance=True)
    if not data:
        print("ERROR: No samples loaded!")
        return

    print("=" * 60)
    print(f"Initializing {model_ckpt} model...")
    extractor = GraphExtractor(model_ckpt=model_ckpt, device=device)

    # ---- Select layers ----
    selected_layers = layer_indices if layer_indices is not None else evenly_spaced_layers(extractor.num_layers, layer_slices)
    print(f"Selected layers (total {extractor.num_layers}): {selected_layers}")

    # Per-layer collectors for tracking top-k occurrences
    full_degree_counter = {L: Counter() for L in selected_layers}
    vision_degree_counter = {L: Counter() for L in selected_layers}
    text_degree_counter = {L: Counter() for L in selected_layers}
    last_token_counter = {L: Counter() for L in selected_layers}

    full_degree_sum = {L: np.zeros(extractor.hidden_dim) for L in selected_layers}
    vision_degree_sum = {L: np.zeros(extractor.hidden_dim) for L in selected_layers}
    text_degree_sum = {L: np.zeros(extractor.hidden_dim) for L in selected_layers}
    last_token_sum = {L: np.zeros(extractor.hidden_dim) for L in selected_layers}

    correct, total = 0, 0

    # ---- Main loop ----
    for sample_idx, (image, prompt, answer) in enumerate(tqdm(data, desc="Processing VQA samples", disable=True)):
        try:
            hidden_states_all, gen, [image_token_start, text_token_start] = extractor.process_single(image, prompt)
            hidden_states_all = [h.detach() for h in hidden_states_all]
            _, seq_len, hidden_dim = hidden_states_all[-1].shape

            # Process each selected layer
            for L in selected_layers:
                L_eff = min(L, len(hidden_states_all) - 1)
                
                hs = hidden_states_all[L_eff][0]  # [seq, hidden]
                
                # 1. Full hidden state degree
                hs_corr_matrix = compute_corr_matrix(hs)  # [hidden, hidden]
                hs_degree = node_degrees(hs_corr_matrix)  # [hidden]
                full_degree_sum[L] += hs_degree.float().cpu().numpy()
                top_full_neurons = torch.topk(hs_degree, k=top_k).indices.cpu().numpy()
                for neuron_idx in top_full_neurons:
                    full_degree_counter[L][int(neuron_idx)] += 1
                
                # 2. Text-only hidden state degree
                text_hs = hs[text_token_start:, :]  # [text_seq, hidden]
                text_hs_corr_matrix = compute_corr_matrix(text_hs)  # [hidden, hidden]
                text_hs_degree = node_degrees(text_hs_corr_matrix)  # [hidden]
                text_degree_sum[L] += text_hs_degree.float().cpu().numpy()
                top_text_neurons = torch.topk(text_hs_degree, k=top_k).indices.cpu().numpy()
                for neuron_idx in top_text_neurons:
                    text_degree_counter[L][int(neuron_idx)] += 1

                # 3. Vision-only hidden state degree
                vision_hs = hs[image_token_start:text_token_start, :]  # [vision_seq, hidden]
                vision_hs_corr_matrix = compute_corr_matrix(vision_hs)  # [hidden, hidden]
                vision_hs_degree = node_degrees(vision_hs_corr_matrix)
                vision_degree_sum[L] += vision_hs_degree.float().cpu().numpy()
                top_vision_neurons = torch.topk(vision_hs_degree, k=top_k).indices.cpu().numpy()
                for neuron_idx in top_vision_neurons:
                    vision_degree_counter[L][int(neuron_idx)] += 1
                
                # 4. Last token activation magnitude
                last_token_hs = hs[-1, :].float().abs()  # [hidden]
                top_last_neurons = torch.topk(last_token_hs, k=top_k).indices.cpu().numpy()
                last_token_sum[L] += last_token_hs.cpu().numpy()
                for neuron_idx in top_last_neurons:
                    last_token_counter[L][int(neuron_idx)] += 1

            # Accuracy
            pred = (gen.lower().strip() if isinstance(gen, str) else str(gen))
            ref  = (answer.lower().strip() if isinstance(answer, str) else str(answer))
            correct += int(pred == ref)
            total += 1

            # Logging
            if sample_idx > 0 and (sample_idx % log_every == 0):
                print(f"\n--- Sample {sample_idx} ---")
                print(f"Q: {prompt}")
                print(f"Pred: {gen} | Ref: {answer}")
                if verbose:
                    print(f"Image tokens starts at {image_token_start}, text tokens starts at {text_token_start}, total seq len {seq_len}")

            del hidden_states_all
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing sample {sample_idx}: {str(e)}")
            continue

    # ---- Hub Neuron Analysis ----
    print(f"\n{'='*60}")
    print(f"HUB NEURON ANALYSIS")
    print(f"{'='*60}")
    print(f"Total samples processed: {total}")
    acc = (correct / total) if total > 0 else 0.0
    print(f"Accuracy: {correct}/{total} = {acc * 100:.2f}%\n")

    summed_topk = {}

    for L in selected_layers:
        print(f"\n{'='*60}")
        print(f"Layer {L}")
        print(f"{'='*60}")
        
        # 1. Full hidden state degree
        print(f"\n[1] Top 10 Neurons by Full Hidden State Degree (appeared in top-{top_k}):")
        for rank, (neuron_idx, count) in enumerate(full_degree_counter[L].most_common(10), 1):
            percentage = (count / total) * 100
            print(f"  {rank}. Neuron {neuron_idx}: appeared {count}/{total} times ({percentage:.1f}%)")
        
        # 2. Text-only hidden state degree
        print(f"\n[2] Top 10 Neurons by Text-Only Hidden State Degree (appeared in top-{top_k}):")
        for rank, (neuron_idx, count) in enumerate(text_degree_counter[L].most_common(10), 1):
            percentage = (count / total) * 100
            print(f"  {rank}. Neuron {neuron_idx}: appeared {count}/{total} times ({percentage:.1f}%)")
        
        # 3. Vision-only hidden state degree
        print(f"\n[3] Top 10 Neurons by Vision-Only Hidden State Degree (appeared in top-{top_k}):")
        for rank, (neuron_idx, count) in enumerate(vision_degree_counter[L].most_common(10), 1):
            percentage = (count / total) * 100
            print(f"  {rank}. Neuron {neuron_idx}: appeared {count}/{total} times ({percentage:.1f}%)")
        
        # 4. Last token activation magnitude
        print(f"\n[4] Top 10 Neurons by Last Token Activation Magnitude (appeared in top-{top_k}):")
        for rank, (neuron_idx, count) in enumerate(last_token_counter[L].most_common(10), 1):
            percentage = (count / total) * 100
            print(f"  {rank}. Neuron {neuron_idx}: appeared {count}/{total} times ({percentage:.1f}%)")

        full_top = np.argsort(full_degree_sum[L])[::-1][:top_k]
        text_top = np.argsort(text_degree_sum[L])[::-1][:top_k]
        vision_top = np.argsort(vision_degree_sum[L])[::-1][:top_k]
        last_token_top = np.argsort(last_token_sum[L])[::-1][:top_k]

        summed_topk[L] = {
            "full_degree": full_top.tolist(),
            "text_degree": text_top.tolist(),
            "vision_degree": vision_top.tolist(),
            "last_token": last_token_top.tolist()
        }

        print("  [SUM] full: ", full_top)
        print("  [SUM] text: ", text_top)
        print("  [SUM] vision: ", vision_top)
        print("  [SUM] last_token: ", last_token_top)
        
    # ---- Save results ----
    os.makedirs("results/hub_neurons", exist_ok=True)
    output_file = f"results/hub_neurons/hub_neurons_{model_prefix}_{dataset}_{category}_top{top_k}.json"
    
    results = {
        "metadata": {
            "model": model_ckpt,
            "dataset": dataset,
            "category": category,
            "num_samples": total,
            "accuracy": acc,
            "top_k": top_k,
            "selected_layers": selected_layers
        },
        "full_degree": {str(L): dict(full_degree_counter[L]) for L in selected_layers},
        "text_degree": {str(L): dict(text_degree_counter[L]) for L in selected_layers},
        "vision_degree": {str(L): dict(vision_degree_counter[L]) for L in selected_layers},
        "last_token": {str(L): dict(last_token_counter[L]) for L in selected_layers},
        "summed_topk": summed_topk
    }
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze hub neurons and save results")
    parser.add_argument("--dataset", type=str, default="clevr") 
    parser.add_argument("--model_ckpt", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--category", type=str, default="color")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--layer_slices", type=int, default=4)
    parser.add_argument("--layer_indices", nargs="+", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=10, help="Number of top neurons to track per sample")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Analyzing Hub Neurons")
    print(f"Model: {args.model_ckpt}")
    print(f"Category: {args.category}")
    print(f"{'='*60}\n")

    analyze_hub_neurons(
        dataset=args.dataset,
        num_samples=args.num_samples,
        model_ckpt=args.model_ckpt,
        category=args.category,
        verbose=args.verbose,
        device=args.device,
        log_every=args.log_every,
        layer_slices=args.layer_slices,
        layer_indices=args.layer_indices if args.layer_indices else None,
        top_k=args.top_k
    )
