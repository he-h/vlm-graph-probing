import torch
import numpy as np
from tqdm import tqdm
import pickle, os, json, argparse

from utils import *
from model import NeuronGraphExtractor as GraphExtractor
from model import build_corr_graph
from dataset import prepare_vlm_data 


def create_vqa_dataset(
    dataset,
    num_samples=1000,
    model_ckpt="llava-hf/llava-1.5-7b-hf",
    output_dir="probing_dataset",
    verbose=False,
    device="cuda:0",
    category="color",                  # 'color' | 'counting' | 'existence' | 'comparison' | 'shape'
    sparse_level=0.9,
    log_every=200,
    layer_slices=4,                # K slices -> K+1 evenly spaced layers (incl. first & last)
    layer_indices=None,            # List of layer indices to save (overrides layer_slices if provided)
    save=True
):
    """
    Create a graph probing dataset for a VLM on VQA using prepare_vlm_data().

    Per-layer artifacts (mirror caption script):
      - graphs_layer_<L>.pkl : list of [num_nodes, edge_index(int64), edge_weight(float32)]
      - last_token_layer_<L>.npy : float32 array [N, H]
    """
    model_prefix = model_ckpt2name(model_ckpt)
    out_dir = f"data/{model_prefix}_{dataset}_{category}_sparsity_{int(sparse_level * 100)}_{output_dir}"
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load data ----
    print("=" * 60)
    print(f"Preparing {dataset} samples, category={category}")
    data = prepare_vlm_data(dataset=dataset, num_samples=num_samples, category=category, balance=True)
    if not data:
        print("ERROR: No samples loaded!")
        return []

    print("=" * 60)
    print(f"Initializing {model_ckpt} model...")
    extractor = GraphExtractor(model_ckpt=model_ckpt, device=device)

    # ---- Select layers ----
    selected_layers = layer_indices if layer_indices is not None else evenly_spaced_layers(extractor.num_layers, layer_slices)
    if verbose:
        print(f"Selected layers (total {extractor.num_layers}): {selected_layers}")

    # Per-layer collectors
    graphs_by_layer = {L: [] for L in selected_layers}
    text_graphs_by_layer = {L: [] for L in selected_layers}
    last_token_by_layer = {L: [] for L in selected_layers}

    correct, total = 0, 0
    preds, refs = [], []

    # ---- Main loop ----
    for sample_idx, (image, prompt, answer) in enumerate(tqdm(data, desc="Processing VQA samples", disable=True)):
        try:
            hidden_states_all, gen, [image_token_start, text_token_start] = extractor.process_single(image, prompt)
            # detach to be safe
            hidden_states_all = [h.detach() for h in hidden_states_all]
            _, seq_len, hidden_dim = hidden_states_all[-1].shape

            # Build graphs & last-token vectors for selected layers
            for L in selected_layers:
                L_eff = min(L, len(hidden_states_all) - 1)
                hs = hidden_states_all[L_eff][0]  # [seq, hidden]
                text_hs = hidden_states_all[L_eff][0, text_token_start:, :]  # [text_seq, hidden]
    
                g = build_corr_graph(hs, sparse_level=sparse_level)
                num_nodes = g["num_nodes"]
                edge_index = g["edge_index"].astype(np.int64)
                edge_weight = g["edge_weight"].astype(np.float32)
                graphs_by_layer[L].append([num_nodes, edge_index, edge_weight])
                
                # Text-only graph
                text_g = build_corr_graph(text_hs, sparse_level=sparse_level)
                t_num_nodes = text_g["num_nodes"]
                t_edge_index = text_g["edge_index"].astype(np.int64)
                t_edge_weight = text_g["edge_weight"].astype(np.float32)
                text_graphs_by_layer[L].append([t_num_nodes, t_edge_index, t_edge_weight])

                last_vec = hs[-1, :].contiguous().cpu().numpy().astype(np.float32)
                last_token_by_layer[L].append(last_vec)

            # Accuracy (simple normalization)
            pred = (gen.lower().strip() if isinstance(gen, str) else str(gen))
            ref  = (answer.lower().strip() if isinstance(answer, str) else str(answer))
            preds.append(pred)
            refs.append(ref)
            correct += int(pred == ref)
            total += 1

            # Logging
            if sample_idx > 0 and (sample_idx % log_every == 0):
                print(f"\n--- Sample {sample_idx} ---")
                print(f"Q: {prompt}")
                print(f"Pred: {gen} | Ref: {answer}")
                if verbose:
                    L0 = selected_layers[0]
                    print(f"Image tokens starts at {image_token_start}, text tokens starts at {text_token_start}, total seq len {seq_len}")
                    num_nodes, eidx, ew = graphs_by_layer[L0][-1]
                    print(f"  [Layer {L0}] edges: {eidx.shape[1]}, weights in [{ew.min():.4f}, {ew.max():.4f}]")
                    t_num_nodes, t_eidx, t_ew = text_graphs_by_layer[L0][-1]
                    print(f"  [Layer {L0} Text-only] edges: {t_eidx.shape[1]}, weights in [{t_ew.min():.4f}, {t_ew.max():.4f}]")
                    

            del hidden_states_all
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing sample {sample_idx}: {str(e)}")
            continue

    # ---- Summary ----
    print(f"\nTotal samples processed: {total}")
    acc = (correct / total) if total > 0 else 0.0
    print(f"Accuracy: {correct}/{total} = {acc * 100:.2f}%")

    # ---- SAVE per-layer artifacts ----
    if save:
        print(f"\n{'='*60}\nSAVING PER-LAYER ARTIFACTS\n{'='*60}")
        for L in selected_layers:
            graphs_path = os.path.join(out_dir, f"graphs_layer_{L}.pkl")
            with open(graphs_path, "wb") as f:
                pickle.dump(graphs_by_layer[L], f)
            
            text_graphs_path = os.path.join(out_dir, f"text_graphs_layer_{L}.pkl")
            with open(text_graphs_path, "wb") as f:
                pickle.dump(text_graphs_by_layer[L], f)

            if len(last_token_by_layer[L]) > 0:
                last_mat = np.stack(last_token_by_layer[L], axis=0)  # [N, H]
            else:
                last_mat = np.zeros((0, extractor.hidden_dim), dtype=np.float32)
            np.save(os.path.join(out_dir, f"last_token_layer_{L}.npy"), last_mat)

        with open(os.path.join(out_dir, "preds.json"), "w") as f:
            json.dump(preds, f, indent=2)

        with open(os.path.join(out_dir, "refs.json"), "w") as f:
            json.dump(refs, f, indent=2)
        
        # ---- Metadata ----
        metadata = {
            "num_samples_total": len(data),
            "num_samples_used": total,
            "model": model_ckpt,
            "model_family": extractor.model_family,
            "num_layers": extractor.num_layers,
            "hidden_dim": extractor.hidden_dim,
            "selected_layers": selected_layers,
            "category": category,
            "sparse_level": float(sparse_level),
            "accuracy": acc,
            "files": {
                "graphs": [f"graphs_layer_{L}.pkl" for L in selected_layers],
                "text_graphs": [f"text_graphs_layer_{L}.pkl" for L in selected_layers],
                "last_tokens": [f"last_token_layer_{L}.npy" for L in selected_layers],
            }
        }
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nOutput directory: {out_dir}")
    return graphs_by_layer, last_token_by_layer, preds, refs    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create VQA probing dataset (layer-sliced, per-layer saves)")
    parser.add_argument("--dataset", type=str, default="clevr") 
    parser.add_argument("--model_ckpt", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--num_samples", type=int, default=2500)
    parser.add_argument("--sparse_level", type=float, default=0.9)
    parser.add_argument("--category", type=str, default="color")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="probing_dataset")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--layer_slices", type=int, default=4)
    parser.add_argument("--layer_indices", nargs="+", type=int, default=None, help="List of layer indices to save (overrides layer_slices if provided)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Creating VQA dataset")
    print(f"Model: {args.model_ckpt}")
    print(f"category: {args.category}")
    print(f"{'='*60}\n")

    _ = create_vqa_dataset(
        dataset=args.dataset,
        num_samples=args.num_samples,
        model_ckpt=args.model_ckpt,
        output_dir=args.output_dir,
        sparse_level=args.sparse_level,
        category=args.category,
        verbose=args.verbose,
        device=args.device,
        log_every=args.log_every,
        layer_slices=args.layer_slices,
        layer_indices=args.layer_indices if args.layer_indices else None,
    )
