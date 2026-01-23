import torch
import numpy as np
from tqdm import tqdm
import pickle, os, json, argparse
from collections import Counter

from utils import *
from model import NeuronGraphExtractor as GraphExtractor
from model import compute_corr_matrix, node_degrees
from dataset import prepare_vlm_data, get_candidate_answers


class NeuronInterventionHook:
    """Hook to intervene (scale) specific neurons in a layer."""
    
    def __init__(self, neuron_indices, scale=0.0):
        """
        Args:
            neuron_indices: List of neuron indices to intervene on
            scale: Scaling factor (0.0 = ablate, 0.5 = halve, 2.0 = double, etc.)
        """
        self.neuron_indices = neuron_indices
        self.scale = scale
        self.handle = None
    
    def hook_fn(self, module, input, output):
        """Hook function that scales specified neurons."""
        # Some models return tuple (hidden_states, attention_weights, etc.)
        # Extract the actual hidden states tensor
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        # hidden_states shape: [batch, seq, hidden] or [batch, hidden]
        if len(hidden_states.shape) == 3:
            hidden_states[:, :, self.neuron_indices] = hidden_states[:, :, self.neuron_indices] * self.scale
        elif len(hidden_states.shape) == 2:
            hidden_states[:, self.neuron_indices] = hidden_states[:, self.neuron_indices] * self.scale
        
        # Return in the same format as input
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


def load_hub_neurons(json_path, criterion, layer, top_n, hidden_dim):
    """
    Load top-N hub neurons from saved analysis results.
    
    Args:
        json_path: Path to hub neuron analysis JSON file
        criterion: One of 'full_degree', 'text_degree', 'last_token'
        layer: Layer index
        top_n: Number of top neurons to select
        hidden_dim: Total number of neurons in the layer
    
    Returns:
        List of neuron indices
    """
    if criterion == "random":
        neuron_indices = np.random.choice(hidden_dim, size=top_n, replace=False).tolist()
        return neuron_indices
    with open(json_path, "r") as f:
        data = json.load(f)
    
    counter_data = data[criterion][str(layer)]
    sorted_neurons = sorted(counter_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
    neuron_indices = [int(n[0]) for n in sorted_neurons]
    
    return neuron_indices


def intervention_single_layer(
    dataset,
    num_samples=1000,
    model_ckpt="llava-hf/llava-1.5-7b-hf",
    verbose=False,
    device="cuda:0",
    category="color",
    log_every=200,
    hub_neuron_json=None,
    criterion="full_degree",
    num_intervene=10,
    scale=0.0,
    target_layer=None,
    neuron_indices=None,
):
    """
    Run intervention on a SINGLE layer.
    
    Args:
        target_layer: The specific layer to intervene on
    """
    candidates = get_candidate_answers(dataset, category)

    # ---- Load data ----
    print(f"Loading {dataset} data...")
    data = prepare_vlm_data(dataset=dataset, num_samples=num_samples, category=category, balance=True)
    if not data:
        print("ERROR: No samples loaded!")
        return None

    # ---- Initialize model ----
    print(f"Initializing {model_ckpt}...")
    extractor = GraphExtractor(model_ckpt=model_ckpt, device=device)

    # ---- Setup intervention ----
    hook = None
    intervened_neurons = []
    
    if hub_neuron_json and target_layer is not None:
        if neuron_indices is None:
            neuron_indices = load_hub_neurons(hub_neuron_json, criterion, target_layer, num_intervene, extractor.hidden_dim)
        intervened_neurons = neuron_indices
        
        # Register hook to the target layer
        hook = NeuronInterventionHook(neuron_indices, scale=scale)
        layer_module = extractor.model.language_model.layers[target_layer]
        hook.register(layer_module)
        
        print(f"Layer {target_layer}: Intervening on {len(neuron_indices)} neurons with scale={scale}")
        if verbose:
            print(f"  Neurons: {neuron_indices}")
    else:
        print("Running BASELINE (no intervention)")

    # ---- Run inference ----
    correct, total = 0, 0
    outputs, preds, refs = [], [], []

    for sample_idx, (image, prompt, answer) in enumerate(tqdm(data, desc=f"Layer {target_layer if target_layer else 'baseline'}", disable=True)):
        # try:
        hidden_states_all, gen, logits, [image_token_start, text_token_start] = extractor.process_single(image, prompt, max_new_tokens=1, output_logits=True)
        
        # Accuracy
        output = (gen.lower().strip() if isinstance(gen, str) else str(gen))
        probs = torch.nn.functional.log_softmax(logits[:, -1, :], dim=-1)
        tok = extractor.processor.tokenizer
        ids = [tok(c, add_special_tokens=False)["input_ids"][0] for c in candidates]
        scores = [probs[0, i].item() for i in ids]
        pred = candidates[int(np.argmax(scores))]
        ref  = (answer.lower().strip() if isinstance(answer, str) else str(answer))
        outputs.append(output)
        preds.append(pred)
        refs.append(ref)
        correct += int(pred == ref)
        total += 1

        # Logging
        if verbose and sample_idx > 0 and (sample_idx % log_every == 0):
            print(f"\n--- Sample {sample_idx} ---")
            print(f"Q: {prompt}")
            print(f"Output: {output} | Pred: {pred} | Ref: {answer}")

        torch.cuda.empty_cache()

        # except Exception as e:
        #     print(f"Error processing sample {sample_idx}: {str(e)}")
        #     continue

    # ---- Remove hook ----
    if hook is not None:
        hook.remove()

    # ---- Results ----
    acc = (correct / total) if total > 0 else 0.0
    
    results = {
        "metadata": {
            "model": model_ckpt,
            "dataset": dataset,
            "category": category,
            "num_samples": total,
            "accuracy": acc,
            "criterion": criterion if hub_neuron_json else None,
            "num_intervene": num_intervene if hub_neuron_json else 0,
            "scale": scale,
            "target_layer": target_layer,
            "hub_neuron_json": hub_neuron_json
        },
        "intervened_neurons": intervened_neurons,
        "outputs": outputs,
        "predictions": preds,
        "references": refs
    }
    
    return acc, results


def run_intervention_experiments(
    dataset,
    num_samples=1000,
    model_ckpt="llava-hf/llava-1.5-7b-hf",
    verbose=False,
    device="cuda:0",
    category="color",
    log_every=200,
    layer_slices=4,
    layer_indices=None,
    neuron_indices=None,
    hub_neuron_json=None,
    criterion="full_degree",
    num_intervene=10,
    scale=0.0,
):
    """
    Run intervention experiments across multiple layers (one layer at a time).
    """
    model_prefix = model_ckpt2name(model_ckpt)
    
    # ---- Determine layers to test ----
    # First need to get num_layers from model
    print("=" * 60)
    print(f"Initializing {model_ckpt} to determine layers...")
    temp_extractor = GraphExtractor(model_ckpt=model_ckpt, device=device)
    num_layers = temp_extractor.num_layers
    del temp_extractor
    torch.cuda.empty_cache()
    
    selected_layers = layer_indices if layer_indices is not None else evenly_spaced_layers(num_layers, layer_slices)
    
    print("=" * 60)
    print(f"INTERVENTION EXPERIMENT SETUP")
    print("=" * 60)
    print(f"Model: {model_ckpt}")
    print(f"Dataset: {dataset}, Category: {category}")
    print(f"Samples: {num_samples}")
    print(f"Criterion: {criterion}")
    print(f"Neurons per layer: {num_intervene}")
    print(f"Scale: {scale} {'(ABLATION)' if scale == 0.0 else ''}")
    print(f"Layers to test: {selected_layers}")
    print("=" * 60)
    
    all_results = {}
    
    # # ---- Run baseline (no intervention) ----
    # print(f"\n{'='*60}")
    # print(f"Running BASELINE (no intervention)")
    # print(f"{'='*60}")
    # baseline_acc, baseline_results = intervention_single_layer(
    #     dataset=dataset,
    #     num_samples=num_samples,
    #     model_ckpt=model_ckpt,
    #     verbose=verbose,
    #     device=device,
    #     category=category,
    #     log_every=log_every,
    #     hub_neuron_json=None,
    #     criterion=criterion,
    #     num_intervene=num_intervene,
    #     scale=scale,
    #     target_layer=None,
    # )
    # all_results['baseline'] = baseline_results
    # print(f"Baseline Accuracy: {baseline_acc * 100:.2f}%")
    # TODO: directly load baseline acc
    # ---- Run intervention for each layer ----
    for layer in selected_layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer} Intervention")
        print(f"{'='*60}")
        
        acc, results = intervention_single_layer(
            dataset=dataset,
            num_samples=num_samples,
            model_ckpt=model_ckpt,
            verbose=verbose,
            device=device,
            category=category,
            log_every=log_every,
            hub_neuron_json=hub_neuron_json,
            criterion=criterion,
            num_intervene=num_intervene,
            scale=scale,
            target_layer=layer,
            neuron_indices=neuron_indices
        )
        
        all_results[f'layer_{layer}'] = results
        print(f"Layer {layer} Accuracy: {acc * 100:.2f}%")
    
    # ---- Save all results ----
    os.makedirs("results/intervention", exist_ok=True)
    
    intervention_type = f"{criterion}_top{num_intervene}_scale{scale}"
    output_file = f"results/intervention/intervention_{model_prefix}_{dataset}_{category}_{intervention_type}_all_layers.json"
    if layer_indices is not None:
        layer_str = "_".join([str(l) for l in layer_indices])
        output_file = f"results/intervention/intervention_{model_prefix}_{dataset}_{category}_{intervention_type}_layers_{layer_str}.json"
    
    summary = {
        "experiment_config": {
            "model": model_ckpt,
            "dataset": dataset,
            "category": category,
            "num_samples": num_samples,
            "criterion": criterion,
            "num_intervene": num_intervene,
            "scale": scale,
            "selected_layers": selected_layers,
            "hub_neuron_json": hub_neuron_json
        },
        "results": all_results,
        "summary": {
            "layer_accuracies": {
                layer: all_results[f'layer_{layer}']['metadata']['accuracy'] 
                for layer in selected_layers
            },
            "neurons_intervened": {
                layer: all_results[f'layer_{layer}']['intervened_neurons']
                for layer in selected_layers
            }
        }
    }
    
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    # print(f"Baseline: {baseline_acc * 100:.2f}%")
    for layer in selected_layers:
        layer_acc = all_results[f'layer_{layer}']['metadata']['accuracy']
        print(f"Layer {layer:2d}: {layer_acc * 100:.2f}%")
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neuron intervention experiment (per-layer)")
    parser.add_argument("--dataset", type=str, default="clevr") 
    parser.add_argument("--model_ckpt", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--category", type=str, default="color")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--layer_slices", type=int, default=4)
    parser.add_argument("--layer_indices", nargs="+", type=int, default=None)
    parser.add_argument("--neuron_indices", nargs="+", type=int, default=None)
    
    # Intervention-specific arguments
    parser.add_argument("--hub_neuron_json", type=str, required=True,
                       help="Path to hub neuron analysis JSON file")
    parser.add_argument("--criterion", type=str, default="full_degree",
                       choices=["full_degree", "text_degree", "last_token", "random"],
                       help="Criterion for selecting neurons to intervene on")
    parser.add_argument("--num_intervene", type=int, default=10,
                       help="Number of top neurons to intervene on per layer")
    parser.add_argument("--scale", type=float, default=0.0,
                       help="Scaling factor (0.0=ablate, 0.5=suppress, 1.0=no-op, 2.0=amplify)")
    
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Neuron Intervention Experiment (Per-Layer)")
    print(f"Model: {args.model_ckpt}")
    print(f"Category: {args.category}")
    print(f"Criterion: {args.criterion}")
    print(f"Neurons to intervene: {args.num_intervene}")
    print(f"Scale: {args.scale}")
    print(f"{'='*60}\n")

    run_intervention_experiments(
        dataset=args.dataset,
        num_samples=args.num_samples,
        model_ckpt=args.model_ckpt,
        category=args.category,
        verbose=args.verbose,
        device=args.device,
        log_every=args.log_every,
        layer_slices=args.layer_slices,
        layer_indices=args.layer_indices if args.layer_indices else None,
        neuron_indices=args.neuron_indices if args.neuron_indices else None,
        hub_neuron_json=args.hub_neuron_json,
        criterion=args.criterion,
        num_intervene=args.num_intervene,
        scale=args.scale
    )