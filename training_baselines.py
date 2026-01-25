# Training for baseline models
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.data import Dataset, DataLoader, PyGDataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# def nearest_neighbor_baseline(test_df, train_df, protein_cols, k=5):
#     predictions = []
#     for _, row in test_df.iterrows():
#         neighbor_ids = row['KNN']
#         # Find neighbors in training data
#         neighbor_features = train_df[train_df['CellID'].isin(neighbor_ids)][protein_cols]
#         pred = neighbor_features.mean(axis=0).values
#         predictions.append(pred)
#     return np.array(predictions)

# class LinearBaseline(nn.Module):
#     def __init__(self, num_proteins, num_neighbors=5):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(num_proteins * num_neighbors, num_proteins * 2),
#             nn.ReLU(),
#             nn.Linear(num_proteins * 2, num_proteins)
#         )
    
#     def forward(self, neighbor_features):
#         # Flatten neighbors: (batch, k, proteins) -> (batch, k*proteins)
#         x = neighbor_features.reshape(neighbor_features.size(0), -1)
#         return self.fc(x)

# ============================================================================
# Model Definitions
# ============================================================================

class GCN(nn.Module):
    def __init__(self, num_proteins, hidden_dim=256, num_layers=3, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_proteins, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, num_proteins))
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GAT(nn.Module):
    def __init__(self, num_proteins, hidden_dim=256, num_layers=3, heads=4, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(num_proteins, hidden_dim // heads, heads=heads, dropout=dropout))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
        
        # Output layer (single head)
        self.convs.append(GATConv(hidden_dim, num_proteins, heads=1, concat=False, dropout=dropout))
        
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, num_proteins, hidden_dim=256, num_layers=3, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(num_proteins, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, num_proteins))
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


# ============================================================================
# Data Preparation
# ============================================================================

def df_to_pyg_data(df, protein_cols):
    """Convert DataFrame to PyTorch Geometric Data object."""
    # Node features (protein expression)
    x = torch.FloatTensor(df[protein_cols].values)
    
    # Edge index from KNN
    edge_index = []
    cell_to_idx = {cell_id: idx for idx, cell_id in enumerate(df['CellID'])}
    
    for idx, row in df.iterrows():
        knn_str = row['KNN']
        
        # Parse KNN
        if isinstance(knn_str, (list, np.ndarray)):
            neighbors = [int(n) for n in knn_str]
        # elif isinstance(knn_str, str):
        #     try:
        #         neighbors = ast.literal_eval(knn_str)
        #         neighbors = [int(n) for n in neighbors]
        #     except:
        #         neighbors = [int(x.strip()) for x in knn_str.strip('[]').split(',')]
        else:
            neighbors = []
        
        # Add edges (bidirectional)
        source_idx = cell_to_idx.get(row['CellID'])
        if source_idx is not None:
            for neighbor_id in neighbors:
                target_idx = cell_to_idx.get(neighbor_id)
                if target_idx is not None:
                    edge_index.append([source_idx, target_idx])
                    edge_index.append([target_idx, source_idx])  # Make undirected
    
    edge_index = torch.LongTensor(edge_index).t().contiguous()
    
    # Remove duplicate edges
    edge_index = torch.unique(edge_index, dim=1)
    
    data = Data(x=x, edge_index=edge_index, y=x.clone())  # y is target (same as x for self-supervised)
    
    return data


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch_pyg(model, data, optimizer, criterion, device, train_mask):
    """Train for one epoch on PyG data."""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data.x.to(device), data.edge_index.to(device))
    
    # Compute loss only on training nodes
    loss = criterion(out[train_mask], data.y[train_mask].to(device))
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate_pyg(model, data, criterion, device, test_mask):
    """Evaluate PyG model."""
    model.eval()
    
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        
        # Get predictions and labels for test set
        preds = out[test_mask].cpu().numpy()
        labels = data.y[test_mask].cpu().numpy()
        
        # Compute loss
        loss = criterion(out[test_mask], data.y[test_mask].to(device))
        
        # Compute metrics
        mae = mean_absolute_error(labels.flatten(), preds.flatten())
        r2 = r2_score(labels.flatten(), preds.flatten())
        
        # Per-protein R²
        per_protein_r2 = []
        for i in range(labels.shape[1]):
            try:
                r2_i = r2_score(labels[:, i], preds[:, i])
                per_protein_r2.append(r2_i)
            except:
                per_protein_r2.append(0.0)
        
        mean_protein_r2 = np.mean(per_protein_r2)
        
        # Pearson correlation
        try:
            pearson_r, _ = pearsonr(labels.flatten(), preds.flatten())
        except:
            pearson_r = 0.0
    
    return {
        'loss': loss.item(),
        'mae': mae,
        'r2': r2,
        'mean_protein_r2': mean_protein_r2,
        'pearson_r': pearson_r,
        'per_protein_r2': per_protein_r2
    }


# ============================================================================
# Main Training Function
# ============================================================================

def train_pyg_model(df, protein_start_col, protein_end_col,
                    model_type='GCN',  # 'GCN', 'GAT', or 'GraphSAGE'
                    hidden_dim=256, num_layers=3, heads=4,
                    num_epochs=50, lr=1e-4,
                    test_size=0.2, rebuild_knn=True, normalize=True,
                    device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train a PyTorch Geometric model (GCN/GAT/GraphSAGE).
    
    Args:
        df: DataFrame with spatial proteomics data
        protein_start_col: Start index/name of protein columns
        protein_end_col: End index/name of protein columns
        model_type: One of 'GCN', 'GAT', 'GraphSAGE'
        hidden_dim: Hidden dimension size
        num_layers: Number of graph conv layers
        heads: Number of attention heads (GAT only)
        num_epochs: Number of training epochs
        lr: Learning rate
        test_size: Fraction of data for testing
        rebuild_knn: Whether to rebuild KNN after split
        device: Device to train on
        l2_reg: Weight decay (L2 penalty) for Adam optimization algorithm
    """
    
    # Extract protein columns
    if isinstance(protein_start_col, tuple) and isinstance(protein_end_col, tuple):
        protein_cols = df.columns[protein_start_col[0]:protein_end_col[0]].tolist()
        for i in range(1, len(protein_start_col)):
            protein_cols += df.columns[protein_start_col[i]:protein_end_col[i]].tolist()
    else:
        start_idx = df.columns.get_loc(protein_start_col)
        end_idx = df.columns.get_loc(protein_end_col) + 1
        protein_cols = df.columns[start_idx:end_idx].tolist()
    
    print(f"Training {model_type} on {len(protein_cols)} protein markers")

    # Normalize proteins BEFORE splitting
    if normalize:
        print("Normalizing protein expression (z-score per protein)...")
        scaler = StandardScaler()
        df[protein_cols] = scaler.fit_transform(df[protein_cols])
        print(f"After normalization - Mean: {df[protein_cols].mean().mean():.4f}, Std: {df[protein_cols].std().mean():.4f}")
    
    # Split data by GraphID
    unique_graphs = df['GraphID'].unique()
    train_graphs, test_graphs = train_test_split(
        unique_graphs, test_size=test_size, random_state=42
    )
    
    train_df = df[df['GraphID'].isin(train_graphs)].copy()
    test_df = df[df['GraphID'].isin(test_graphs)].copy()
    
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Rebuild KNN if requested
    if rebuild_knn:
        print("Rebuilding KNN relationships within train/test splits...")
        
        def rebuild_knn_for_split(split_df, k=5):
            split_df = split_df.reset_index(drop=True)
            new_knn = []
            
            for graph_id in split_df['GraphID'].unique():
                graph_mask = split_df['GraphID'] == graph_id
                graph_df = split_df[graph_mask]
                
                if len(graph_df) < k + 1:
                    for idx in graph_df.index:
                        others = [i for i in graph_df.index if i != idx]
                        while len(others) < k:
                            others.append(idx)
                        new_knn.append(graph_df.iloc[others[:k]]['CellID'].tolist())
                else:
                    coords = graph_df[['X:X', 'Y:Y', 'Z:Z']].values
                    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
                    distances, indices = nbrs.kneighbors(coords)
                    
                    for i, idx in enumerate(graph_df.index):
                        neighbor_indices = indices[i][1:]
                        neighbor_cell_ids = graph_df.iloc[neighbor_indices]['CellID'].tolist()
                        new_knn.append(neighbor_cell_ids)
            
            return new_knn
        
        train_df['KNN'] = rebuild_knn_for_split(train_df)
        test_df['KNN'] = rebuild_knn_for_split(test_df)
        print("KNN rebuild complete!")
    
    # Combine train/test back into single graph with masks
    full_df = pd.concat([train_df, test_df]).reset_index(drop=True)
    
    # Create PyG data object
    data = df_to_pyg_data(full_df, protein_cols)
    
    # Create train/test masks
    train_mask = torch.zeros(len(full_df), dtype=torch.bool)
    train_mask[:len(train_df)] = True
    test_mask = ~train_mask
    
    print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")
    
    # Initialize model
    if model_type == 'GCN':
        model = GCN(len(protein_cols), hidden_dim, num_layers).to(device)
    elif model_type == 'GAT':
        model = GAT(len(protein_cols), hidden_dim, num_layers, heads).to(device)
    elif model_type == 'GraphSAGE':
        model = GraphSAGE(len(protein_cols), hidden_dim, num_layers).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    # Training loop
    best_r2 = -float('inf')
    
    for epoch in range(num_epochs):
        train_loss = train_epoch_pyg(model, data, optimizer, criterion, device, train_mask)
        test_metrics = evaluate_pyg(model, data, criterion, device, test_mask)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test Loss: {test_metrics['loss']:.4f}, "
              f"MAE: {test_metrics['mae']:.2f}, "
              f"R²: {test_metrics['r2']:.4f}, "
              f"Mean Protein R²: {test_metrics['mean_protein_r2']:.4f}, "
              f"Pearson r: {test_metrics['pearson_r']:.4f}")
        
        # Save best model
        if test_metrics['r2'] > best_r2:
            best_r2 = test_metrics['r2']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'r2': best_r2,
                'protein_cols': protein_cols
            }, f'best_{model_type.lower()}_model.pt')
            print(f"  Saved best model (R²: {best_r2:.4f})")
    
    return model, data, train_mask, test_mask, test_metrics


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    df = pd.read_csv('CRC_clusters_neighborhoods_markers_cleaned.csv')
    
    # Train GCN
    gcn_model, data, train_mask, test_mask, gcn_metrics = train_pyg_model(
        df, 
        protein_start_col=(1,57), protein_end_col=(50,65),
        model_type='GCN',
        hidden_dim=256,
        num_layers=3,
        num_epochs=50
    )
    
    # Train GAT
    gat_model, data, train_mask, test_mask, gat_metrics = train_pyg_model(
        df, 
        protein_start_col=(1,57), protein_end_col=(50,65),
        model_type='GAT',
        hidden_dim=256,
        num_layers=3,
        heads=4,
        num_epochs=50
    )
    
    # Train GraphSAGE
    sage_model, data, train_mask, test_mask, sage_metrics = train_pyg_model(
        df, 
        protein_start_col=(1,57), protein_end_col=(50,65),
        model_type='GraphSAGE',
        hidden_dim=256,
        num_layers=3,
        num_epochs=50
    )
    
    # Compare results
    print("\n" + "="*50)
    print("FINAL COMPARISON")
    print("="*50)
    print(f"GCN       - R²: {gcn_metrics['r2']:.4f}, MAE: {gcn_metrics['mae']:.2f}")
    print(f"GAT       - R²: {gat_metrics['r2']:.4f}, MAE: {gat_metrics['mae']:.2f}")
    print(f"GraphSAGE - R²: {sage_metrics['r2']:.4f}, MAE: {sage_metrics['mae']:.2f}")