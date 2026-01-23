import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse

# Import the ablation experiment function
from probing.vqa.ablate_layer import run_clevr_ablation_experiment


def run_ablation_sweep(
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    task: str = "color",
    ablation_methods: list = None,
    ablation_percentage: float = 0.1,
    layers: list = None,
    num_samples: int = 500,
    device: str = "cuda:0",
    sparse_level: float = 0.9,
    output_dir: str = "ablation_sweep_results"
):
    """
    Run ablation experiments across multiple layers and methods, then plot results.
    
    Args:
        model_name: HuggingFace model name
        task: CLEVR task type
        ablation_methods: list of methods ['random', 'top_degree', 'top_activation']
        ablation_percentage: fixed at 0.1 (10%)
        layers: list of layer indices to test
        num_samples: number of samples per experiment
        device: cuda device
        sparse_level: sparsity level for graph analysis
        output_dir: directory to save plots and summary
    """
    
    if ablation_methods is None:
        ablation_methods = ['random', 'top_degree', 'top_activation']
    
    if layers is None:
        layers = [0, 8, 17, 26, 35]
    
    model_prefix = model_name.split('/')[-1]
    
    print("="*60)
    print("ABLATION SWEEP EXPERIMENT")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Task: {task}")
    print(f"Layers to test: {layers}")
    print(f"Methods: {ablation_methods}")
    print(f"Ablation percentage: {ablation_percentage*100:.0f}%")
    print(f"Samples per experiment: {num_samples}")
    print("="*60)
    
    # Storage for results
    all_results = {method: {'layers': [], 'accuracy': [], 'probability': []} 
                   for method in ablation_methods}
    
    # Run experiments
    for method in ablation_methods:
        for layer in layers:
            print(f"\n{'='*60}")
            print(f"Running: Layer {layer}, Method {method}")
            print(f"{'='*60}\n")
            
            try:
                accuracy, probability = run_clevr_ablation_experiment(
                    model_name=model_name,
                    task=task,
                    ablation_layer=layer,
                    ablation_method=method,
                    ablation_percentage=ablation_percentage,
                    num_samples=num_samples,
                    sparse_level=sparse_level,
                    data_dir=None,
                    device=device,
                    log_every=50
                )
                
                all_results[method]['layers'].append(layer)
                all_results[method]['accuracy'].append(accuracy)
                all_results[method]['probability'].append(probability if probability is not None else np.nan)
                
                print(f"\nCompleted: Layer {layer}, Method {method}")
                # print(f"Accuracy: {accuracy*100:.2f}%, P(correct): {probability*100 if probability else 'N/A':.2f}%\n")
                
            except Exception as e:
                print(f"ERROR in layer {layer}, method {method}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw results
    results_to_save = {}
    for method, data in all_results.items():
        results_to_save[method] = {
            'layers': data['layers'],
            'accuracy': [float(a) for a in data['accuracy']],
            'probability': [float(p) if not np.isnan(p) else None for p in data['probability']]
        }
    
    summary_path = os.path.join(output_dir, f"{model_prefix}_{task}_ablation_sweep_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Creating plots...")
    print(f"{'='*60}\n")
    
    # Create plots
    create_ablation_plots(
        all_results=all_results,
        layers=layers,
        task=task,
        model_name=model_name,
        ablation_percentage=ablation_percentage,
        output_dir=output_dir
    )
    
    print(f"\nAll results saved to: {output_dir}")
    print("="*60)


def create_ablation_plots(
    all_results: dict,
    layers: list,
    task: str,
    model_name: str,
    ablation_percentage: float,
    output_dir: str
):
    """Create two plots: accuracy and probability across layers."""
    
    model_prefix = model_name.split('/')[-1]
    
    # Define colors and markers for methods
    method_styles = {
        'random': {'color': 'gray', 'marker': 'o', 'label': 'Random'},
        'top_degree': {'color': 'blue', 'marker': 's', 'label': 'Top Degree'},
        'top_activation': {'color': 'red', 'marker': '^', 'label': 'Top Activation'}
    }
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy
    ax1 = axes[0]
    for method, data in all_results.items():
        if len(data['layers']) > 0:
            style = method_styles.get(method, {'color': 'black', 'marker': 'o', 'label': method})
            ax1.plot(
                data['layers'], 
                [acc * 100 for acc in data['accuracy']], 
                marker=style['marker'],
                color=style['color'],
                label=style['label'],
                linewidth=2,
                markersize=8
            )
    
    ax1.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Ablation Accuracy across Layers\n{model_prefix} - {task.capitalize()} Task', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(layers)
    
    # Plot 2: Probability
    ax2 = axes[1]
    for method, data in all_results.items():
        if len(data['layers']) > 0:
            style = method_styles.get(method, {'color': 'black', 'marker': 'o', 'label': method})
            # Filter out NaN values for plotting
            valid_indices = [i for i, p in enumerate(data['probability']) if not np.isnan(p)]
            valid_layers = [data['layers'][i] for i in valid_indices]
            valid_probs = [data['probability'][i] * 100 for i in valid_indices]
            
            if len(valid_layers) > 0:
                ax2.plot(
                    valid_layers, 
                    valid_probs,
                    marker=style['marker'],
                    color=style['color'],
                    label=style['label'],
                    linewidth=2,
                    markersize=8
                )
    
    ax2.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('P(Correct Answer) (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Correct Answer Probability across Layers\n{model_prefix} - {task.capitalize()} Task', 
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(layers)
    
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(output_dir, f"{model_prefix}_{task}_ablation_sweep_{int(ablation_percentage*100)}pct.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    
    # Also save as PDF
    pdf_path = plot_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved plot: {pdf_path}")
    
    plt.close()
    
    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"\nTask: {task.capitalize()}")
    print(f"Ablation: {ablation_percentage*100:.0f}% of neurons\n")
    
    for method in all_results.keys():
        data = all_results[method]
        if len(data['layers']) > 0:
            print(f"\n{method.upper().replace('_', ' ')}:")
            print(f"{'Layer':<8} {'Accuracy':<12} {'P(Correct)':<12}")
            print("-" * 35)
            for i, layer in enumerate(data['layers']):
                acc = data['accuracy'][i] * 100
                prob = data['probability'][i] * 100 if not np.isnan(data['probability'][i]) else None
                prob_str = f"{prob:.2f}%" if prob is not None else "N/A"
                print(f"{layer:<8} {acc:>6.2f}%      {prob_str:>10}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation sweep across multiple layers")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--task", type=str, default="color",
                        choices=['color', 'counting', 'existence', 'comparison', 'shape'])
    parser.add_argument("--ablation_methods", nargs='+', 
                        default=['random', 'top_degree', 'top_activation'],
                        choices=['random', 'top_degree', 'top_activation'])
    parser.add_argument("--ablation_percentage", type=float, default=0.1)
    parser.add_argument("--layers", nargs='+', type=int, default=[0, 8, 17, 26, 35])
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sparse_level", type=float, default=0.9)
    parser.add_argument("--output_dir", type=str, default="ablation_sweep_results")
    args = parser.parse_args()
    
    run_ablation_sweep(
        model_name=args.model_name,
        task=args.task,
        ablation_methods=args.ablation_methods,
        ablation_percentage=args.ablation_percentage,
        layers=args.layers,
        num_samples=args.num_samples,
        device=args.device,
        sparse_level=args.sparse_level,
        output_dir=args.output_dir
    )