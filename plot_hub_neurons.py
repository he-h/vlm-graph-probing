import json
import matplotlib.pyplot as plt
import numpy as np

def plot_hub_neurons(json_path, top_n=20, save_path=None):
    """
    Visualize hub neuron analysis results with detailed bar charts.
    
    Args:
        json_path: Path to the saved JSON results file
        top_n: Number of top neurons to display per criterion
        save_path: Path to save the plot (if None, uses auto-generated name)
    """
    # Load results
    with open(json_path, "r") as f:
        data = json.load(f)
    
    metadata = data["metadata"]
    selected_layers = metadata["selected_layers"]
    total_samples = metadata["num_samples"]
    
    # Create figure with subplots: 4 columns (criteria) x N rows (layers)  # CHANGED: 3 -> 4
    n_layers = len(selected_layers)
    fig, axes = plt.subplots(n_layers, 4, figsize=(24, 5 * n_layers))  # CHANGED: 3 -> 4, 18 -> 24
    
    # Handle single layer case
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    
    criteria = [
        ("full_degree", "Full Hidden State Degree", "#3498db"),
        ("vision_degree", "Vision-Only Hidden State Degree", "#f39c12"),  # ADDED
        ("text_degree", "Text-Only Hidden State Degree", "#e74c3c"),
        ("last_token", "Last Token Activation Magnitude", "#2ecc71")
    ]
    
    for row_idx, layer in enumerate(selected_layers):
        layer_str = str(layer)
        
        for col_idx, (criterion_key, criterion_name, color) in enumerate(criteria):
            ax = axes[row_idx, col_idx]
            
            # Get counter data for this layer and criterion
            counter_data = data[criterion_key][layer_str]
            
            # Sort by count and get top N
            sorted_neurons = sorted(counter_data.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True)[:top_n]
            
            if not sorted_neurons:
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
                ax.set_title(f"Layer {layer} - {criterion_name}")
                continue
            
            neuron_ids = [int(n[0]) for n in sorted_neurons]
            counts = [n[1] for n in sorted_neurons]
            percentages = [c / total_samples * 100 for c in counts]
            
            # Create horizontal bar plot
            y_pos = np.arange(len(neuron_ids))
            bars = ax.barh(y_pos, percentages, color=color, alpha=0.7)
            
            # Customize appearance
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"N{nid}" for nid in neuron_ids], fontsize=9)
            ax.set_xlabel("Frequency (%)", fontsize=10)
            ax.set_title(f"Layer {layer}\n{criterion_name}", fontsize=11, fontweight='bold')
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.set_xlim(0, 100)
            
            # Add count labels on bars
            for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
                ax.text(pct + 1, bar.get_y() + bar.get_height()/2, 
                       f'{count}', va='center', fontsize=8)
            
            # Invert y-axis so top neuron is at top
            ax.invert_yaxis()
    
    # Overall title
    model_name = metadata["model"].split("/")[-1]
    fig.suptitle(
        f"Hub Neuron Analysis: {model_name} | {metadata['dataset'].upper()} | "
        f"Category: {metadata['category']} | Top-{metadata['top_k']} per sample\n"
        f"Accuracy: {metadata['accuracy']*100:.1f}% | Samples: {total_samples}",
        fontsize=14, fontweight='bold', y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    if save_path is None:
        save_path = json_path.replace('.json', '_visualization.png').replace('results/', 'figures/')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.close()


def plot_hub_neurons_compact(json_path, top_n=15, save_path=None):
    """
    Original compact visualization: scatter plot showing neuron frequency across layers.
    
    Each criterion gets its own subplot, showing how frequently each neuron
    appears in top-k across different layers.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    metadata = data["metadata"]
    selected_layers = metadata["selected_layers"]
    total_samples = metadata["num_samples"]
    
    fig, axes = plt.subplots(1, 4, figsize=(26, 6))
    
    criteria = [
        ("full_degree", "Full Hidden State Degree", "#3498db"),
        ("vision_degree", "Vision-Only Hidden State Degree", "#f39c12"),
        ("text_degree", "Text-Only Hidden State Degree", "#e74c3c"),
        ("last_token", "Last Token Activation Magnitude", "#2ecc71")
    ]
    
    for col_idx, (criterion_key, criterion_name, color) in enumerate(criteria):
        ax = axes[col_idx]
        
        # Collect all unique neurons across all layers for this criterion
        all_neurons = set()
        for layer in selected_layers:
            layer_data = data[criterion_key][str(layer)]
            sorted_neurons = sorted(layer_data.items(), key=lambda x: x[1], reverse=True)[:top_n]  # CHANGED: uses top_n
            all_neurons.update([int(n[0]) for n in sorted_neurons])
        
        # For each layer, plot top neurons
        for layer_idx, layer in enumerate(selected_layers):
            layer_data = data[criterion_key][str(layer)]
            sorted_neurons = sorted(layer_data.items(), key=lambda x: x[1], reverse=True)[:top_n]  # CHANGED: uses top_n
            
            neuron_ids = [int(n[0]) for n in sorted_neurons]
            percentages = [n[1] / total_samples * 100 for n in sorted_neurons]
            
            # Plot with size proportional to frequency
            sizes = [p * 3 for p in percentages]  # Scale for visibility
            ax.scatter([layer] * len(neuron_ids), neuron_ids, s=sizes, 
                      alpha=0.6, c=color, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel("Layer", fontsize=12, fontweight='bold')
        ax.set_ylabel("Neuron ID", fontsize=12, fontweight='bold')
        ax.set_title(criterion_name, fontsize=13, fontweight='bold')
        ax.set_xticks(selected_layers)
        ax.grid(True, alpha=0.3)
    
    model_name = metadata["model"].split("/")[-1]
    fig.suptitle(
        f"Hub Neuron Consistency Across Layers: {model_name}\n"
        f"{metadata['dataset'].upper()} | Category: {metadata['category']} | "
        f"Top-{metadata['top_k']} per sample | Accuracy: {metadata['accuracy']*100:.1f}%",
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = json_path.replace('.json', '_compact_scatter.png').replace('results/', 'figures/')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Compact scatter visualization saved to: {save_path}")
    plt.close()


def plot_hub_neurons_compact_per_layer(json_path, top_n=10, save_path=None):  # CHANGED: added top_n parameter with default=10
    """
    Compact visualization: Per-layer view.
    Each layer is a panel showing top-N neuron frequencies across 4 criteria as line curves.
    
    X-axis: Rank (1-N)
    Y-axis: Frequency (%)
    4 lines per panel for 4 criteria
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    metadata = data["metadata"]
    selected_layers = metadata["selected_layers"]
    total_samples = metadata["num_samples"]
    
    n_layers = len(selected_layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 5))
    
    # Handle single layer case
    if n_layers == 1:
        axes = [axes]
    
    criteria = [
        ("full_degree", "Full HS", "#3498db", "o"),
        ("vision_degree", "Vision HS", "#f39c12", "D"),
        ("text_degree", "Text HS", "#e74c3c", "s"),
        ("last_token", "Last Token", "#2ecc71", "^")
    ]
    
    for layer_idx, layer in enumerate(selected_layers):
        ax = axes[layer_idx]
        layer_str = str(layer)
        
        for criterion_key, criterion_label, color, marker in criteria:
            counter_data = data[criterion_key][layer_str]
            
            # Sort and get top N  # CHANGED: comment updated
            sorted_neurons = sorted(counter_data.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True)[:top_n]  # CHANGED: uses top_n instead of 10
            
            if sorted_neurons:
                percentages = [n[1] / total_samples * 100 for n in sorted_neurons]
                ranks = list(range(1, len(percentages) + 1))
                
                # Plot line curve
                ax.plot(ranks, percentages, marker=marker, color=color, 
                       label=criterion_label, linewidth=2.5, markersize=8, alpha=0.8)
        
        ax.set_xlabel("Rank", fontsize=12, fontweight='bold')
        ax.set_ylabel("Frequency (%)", fontsize=12, fontweight='bold')
        ax.set_title(f"Layer {layer}", fontsize=13, fontweight='bold')
        ax.set_xticks(range(1, top_n + 1, max(1, top_n // 10)))  # CHANGED: dynamic ticks based on top_n
        ax.set_xlim(0.5, top_n + 0.5)  # CHANGED: dynamic xlim
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
    
    model_name = metadata["model"].split("/")[-1]
    fig.suptitle(
        f"Hub Neuron Frequency by Rank (Per Layer): {model_name}\n"
        f"{metadata['dataset'].upper()} | Category: {metadata['category']} | "
        f"Top-{metadata['top_k']} per sample | Accuracy: {metadata['accuracy']*100:.1f}%",
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = json_path.replace('.json', '_compact_per_layer.png').replace('results/', 'figures/')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Per-layer compact visualization saved to: {save_path}")

    plt.tight_layout()
    
    if save_path is None:
        save_path = json_path.replace('.json', '_compact_per_layer.png').replace('results/', 'figures/')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Per-layer compact visualization saved to: {save_path}")
    
    # ADDED: Save data as JSON
    save_data = {
        "metadata": {
            "model": metadata["model"],
            "dataset": metadata["dataset"],
            "category": metadata["category"],
            "num_samples": total_samples,
            "accuracy": metadata["accuracy"],
            "top_k": metadata["top_k"],
            "rank_displayed": top_n
        },
        "data_per_layer": {}
    }
    
    for layer in selected_layers:
        layer_str = str(layer)
        save_data["data_per_layer"][layer_str] = {}
        
        for criterion_key, criterion_label, color, marker in criteria:
            counter_data = data[criterion_key][layer_str]
            sorted_neurons = sorted(counter_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            neuron_data = []
            for rank, (neuron_idx, count) in enumerate(sorted_neurons, 1):
                percentage = (count / total_samples) * 100
                neuron_data.append({
                    "rank": rank,
                    "neuron_id": int(neuron_idx),
                    "count": count,
                    "percentage": round(percentage, 2)
                })
            
            save_data["data_per_layer"][layer_str][criterion_key] = neuron_data
    
    json_save_path = save_path.replace('.png', '_data.json')
    with open(json_save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Data saved to: {json_save_path}")
    
    plt.close()


def plot_hub_neurons_compact_per_criterion(json_path, top_n=10, save_path=None):  # CHANGED: added top_n parameter with default=10
    """
    Compact visualization: Per-criterion view.
    Each criterion is a panel showing top-N neuron frequencies across all layers as line curves.
    
    X-axis: Rank (1-N)
    Y-axis: Frequency (%)
    One line per layer
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    metadata = data["metadata"]
    selected_layers = metadata["selected_layers"]
    total_samples = metadata["num_samples"]
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    criteria = [
        ("full_degree", "Full Hidden State Degree", "#3498db"),
        ("vision_degree", "Vision-Only Hidden State Degree", "#f39c12"),
        ("text_degree", "Text-Only Hidden State Degree", "#e74c3c"),
        ("last_token", "Last Token Activation Magnitude", "#2ecc71")
    ]
    
    # Color map for layers
    layer_colors = plt.cm.viridis(np.linspace(0, 1, len(selected_layers)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for col_idx, (criterion_key, criterion_name, base_color) in enumerate(criteria):
        ax = axes[col_idx]
        
        for layer_idx, layer in enumerate(selected_layers):
            layer_str = str(layer)
            counter_data = data[criterion_key][layer_str]
            
            # Sort and get top N  # CHANGED: comment updated
            sorted_neurons = sorted(counter_data.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True)[:top_n]  # CHANGED: uses top_n instead of 10
            
            if sorted_neurons:
                percentages = [n[1] / total_samples * 100 for n in sorted_neurons]
                ranks = list(range(1, len(percentages) + 1))
                
                marker = markers[layer_idx % len(markers)]
                # Plot line curve
                ax.plot(ranks, percentages, marker=marker, 
                       color=layer_colors[layer_idx], 
                       label=f"Layer {layer}", linewidth=2.5, markersize=8, alpha=0.8)
        
        ax.set_xlabel("Rank", fontsize=12, fontweight='bold')
        ax.set_ylabel("Frequency (%)", fontsize=12, fontweight='bold')
        ax.set_title(criterion_name, fontsize=13, fontweight='bold')
        ax.set_xticks(range(1, top_n + 1, max(1, top_n // 10)))  # CHANGED: dynamic ticks based on top_n
        ax.set_xlim(0.5, top_n + 0.5)  # CHANGED: dynamic xlim
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9, ncol=1)
    
    model_name = metadata["model"].split("/")[-1]
    fig.suptitle(
        f"Hub Neuron Frequency by Rank (Per Criterion): {model_name}\n"
        f"{metadata['dataset'].upper()} | Category: {metadata['category']} | "
        f"Top-{metadata['top_k']} per sample | Accuracy: {metadata['accuracy']*100:.1f}%",
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = json_path.replace('.json', '_compact_per_criterion.png').replace('results/', 'figures/')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Per-criterion compact visualization saved to: {save_path}")
    plt.close()


# Usage example
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize hub neuron analysis results")
    parser.add_argument("--json_path", type=str, required=True, 
                       help="Path to JSON results file")
    parser.add_argument("--top_n", type=int, default=20,
                       help="Number of top neurons to display in detailed view")
    parser.add_argument("--top_n_compact", type=int, default=10,  # ADDED: new parameter for compact plots
                       help="Number of top neurons (rank) to display in compact views")
    parser.add_argument("--style", type=str, default="all", 
                       choices=["all", "detailed", "compact_scatter", "compact_layer", "compact_criterion"],
                       help="Visualization style to generate")
    args = parser.parse_args()
    
    if args.style in ["all", "detailed"]:
        plot_hub_neurons(args.json_path, top_n=args.top_n)
    
    if args.style in ["all", "compact_scatter"]:
        plot_hub_neurons_compact(args.json_path, top_n=args.top_n_compact)  # CHANGED: uses top_n_compact
    
    if args.style in ["all", "compact_layer"]:
        plot_hub_neurons_compact_per_layer(args.json_path, top_n=args.top_n_compact)  # CHANGED: passes top_n_compact
    
    if args.style in ["all", "compact_criterion"]:
        plot_hub_neurons_compact_per_criterion(args.json_path, top_n=args.top_n_compact)  # CHANGED: passes top_n_compact


# Usage example
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize hub neuron analysis results")
    parser.add_argument("--json_path", type=str, required=True, 
                       help="Path to JSON results file")
    parser.add_argument("--top_n", type=int, default=20,
                       help="Number of top neurons to display in detailed view")
    parser.add_argument("--style", type=str, default="all", 
                       choices=["all", "detailed", "compact_scatter", "compact_layer", "compact_criterion"],
                       help="Visualization style to generate")
    args = parser.parse_args()
    
    if args.style in ["all", "detailed"]:
        plot_hub_neurons(args.json_path, top_n=args.top_n)
    
    if args.style in ["all", "compact_scatter"]:
        plot_hub_neurons_compact(args.json_path, top_n=args.top_n)
    
    if args.style in ["all", "compact_layer"]:
        plot_hub_neurons_compact_per_layer(args.json_path, top_n=args.top_n)
    
    if args.style in ["all", "compact_criterion"]:
        plot_hub_neurons_compact_per_criterion(args.json_path, top_n=args.top_n)