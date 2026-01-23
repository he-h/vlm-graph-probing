import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import GCNPredictor as GCNRegressor


torch.manual_seed(42)


def load_data(dataset_path, layer_name='graph_layer_0'):
    """Load and prepare data"""
    with open(dataset_path, 'rb') as f:
        samples = pickle.load(f)
    print(f"key samples: {list(samples[0].keys())}")
    
    data_list = []
    num_nodes = samples[0][layer_name]['num_nodes']
    scores = [sample['meteor_score'] for sample in samples if 'meteor_score' in sample]
    print(f"score range: min {min(scores):.4f}, max {max(scores):.4f}, mean {np.mean(scores):.4f}, std {np.std(scores):.4f}")
    for sample in samples:
        graph = sample[layer_name]
        
        # TODO: from torch_geometric.utils import to_undirected

# edge_index = to_undirected(edge_index)

        data = Data(
            x=torch.arange(graph['num_nodes'], dtype=torch.long),
            edge_index=torch.tensor(graph['edge_index'], dtype=torch.long),
            edge_attr=torch.tensor(np.abs(graph['edge_weight']), dtype=torch.float32),
            y=torch.tensor(sample['meteor_score'], dtype=torch.float32)
        )
        data_list.append(data)
    
    return data_list, num_nodes

def split_data(data_list):
    """Split into train/val/test (7:1:2)"""
    n = len(data_list)
    indices = list(range(n))
    
    train_val_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42)
    
    return [data_list[i] for i in train_idx], \
           [data_list[i] for i in val_idx], \
           [data_list[i] for i in test_idx]

def train_epoch(model, loader, optimizer, criterion, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        pred = pred.squeeze(-1)
        loss = criterion(pred, batch.y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
    
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    preds, targets = [], []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = pred.squeeze(-1)
            loss = criterion(pred, batch.y)
            
            total_loss += loss.item() * batch.num_graphs
            
            # Convert to numpy and handle both single and batch predictions
            pred_np = pred.cpu().numpy()
            target_np = batch.y.cpu().numpy()
            
            if pred_np.ndim == 0:  # Single value
                preds.append(float(pred_np))
                targets.append(float(target_np))
            else:  # Batch
                preds.extend(pred_np.tolist())
                targets.extend(target_np.tolist())
    
    preds, targets = np.array(preds), np.array(targets)
    
    return total_loss / len(loader.dataset), \
           mean_absolute_error(targets, preds), \
           r2_score(targets, preds) if len(targets) > 1 else 0.0, \
           preds, targets

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Load data
    print("Loading data...")
    data_list, num_nodes = load_data('data/Qwen2.5-VL-3B_prompt_1_sparsity_90_probing_dataset/complete_dataset.pkl', layer_name='graph_layer_0')
    # data_list, num_nodes = load_data('data/LLaVA-1.5-7B_prompt_1_sparsity_90_probing_dataset/complete_dataset.pkl', layer_name='graph_layer_middle')
    # load only 1000 samples for quick testing
    data_list = data_list[:1000]
    print(f"Number of nodes: {num_nodes}")
    train_data, val_data, test_data = split_data(data_list)
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print(f"Number of nodes: {num_nodes}")
    
    batch_size = 12
    # Create loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    # Model
    model = GCNRegressor(num_nodes).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    # Training
    print("\nTraining...")
    train_losses, val_losses, val_r2s = [], [], []
    best_val_loss = float('inf')
    
    for epoch in tqdm(range(30)):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae, val_r2, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val MAE={val_mae:.4f}, Val R²={val_r2:.4f}")

        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_r2s.append(val_r2)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val MAE={val_mae:.4f}, Val R²={val_r2:.4f}")
    
    # Test
    model.load_state_dict(best_model)
    test_loss, test_mae, test_r2, test_preds, test_targets = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  R²:   {test_r2:.4f}")
    
    # Print some examples
    print("\nExample predictions:")
    for i in range(min(5, len(test_preds))):
        print(f"  Actual: {test_targets[i]:.4f}, Predicted: {test_preds[i]:.4f}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Training curves
    axes[0].plot(train_losses, label='Train')
    axes[0].plot(val_losses, label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # R² over time
    axes[1].plot(val_r2s)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('R²')
    axes[1].set_title('Validation R²')
    axes[1].grid(True, alpha=0.3)
    
    # Predictions vs actual
    axes[2].scatter(test_targets, test_preds, alpha=0.5)
    min_val = min(test_targets.min(), test_preds.min())
    max_val = max(test_targets.max(), test_preds.max())
    axes[2].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[2].set_xlabel('Actual meteor scores')
    axes[2].set_ylabel('Predicted meteor scores')
    axes[2].set_title(f'Test Predictions (R²={test_r2:.3f})')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qwen_prompt_1_2_nonlinear_last_90.png')
    plt.show()
    

if __name__ == "__main__":
    main()