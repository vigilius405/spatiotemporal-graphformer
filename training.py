import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
#from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from graphformer import SpatialProteomicsDataset, GraphAttention, GraphFormer

# ============================================================================
# Training Functions
# ============================================================================

def collate_fn(batch, full_features):
    """Custom collate function to gather neighbor features."""
    cell_indices = torch.LongTensor([item['cell_idx'] for item in batch])
    features = torch.stack([item['features'] for item in batch])
    neighbor_indices = torch.stack([item['neighbor_indices'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    # Debug: Check neighbor_indices
    # print(f"Neighbor indices shape: {neighbor_indices.shape}")  # Should be (batch_size, max_neighbors)
    # print(f"Example neighbor indices: {neighbor_indices[0]}")  # Should be 5 different indices
    # print(f"Full features shape: {full_features.shape}")  # Should be (num_cells, num_proteins)
    
    # Gather neighbor features
    B, N = neighbor_indices.shape
    # Clamp indices to valid range to prevent index errors
    neighbor_indices_flat = neighbor_indices.reshape(-1).clamp(0, len(full_features) - 1)
    neighbor_features = full_features[neighbor_indices_flat]
    neighbor_features = neighbor_features.reshape(B, N, -1)
    
    # Debug: Check neighbor features
    # print(f"Neighbor features shape: {neighbor_features.shape}")  # Should be (batch_size, max_neighbors, num_proteins)
    
    return {
        'cell_indices': cell_indices,
        'features': features,
        'neighbor_features': neighbor_features,
        'labels': labels
    }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    first = False #whether or not to print debugging stuff
    for batch in dataloader:
        neighbor_features = batch['neighbor_features'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (neighbors only, no center cell features)
        outputs = model(neighbor_features)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        # Diagnosing problems
        if first:
            first = False
            print(f"\n=== DEBUG FIRST BATCH ===")
            print(f"Neighbor features shape: {neighbor_features.shape}")
            print(f"Neighbor features - min: {neighbor_features.min():.4f}, max: {neighbor_features.max():.4f}, mean: {neighbor_features.mean():.4f}")
            print(f"Neighbor features - any NaN: {torch.isnan(neighbor_features).any()}")
            print(f"Neighbor features - sample: {neighbor_features[0, 0, :5]}")
            
            print(f"\nLabels shape: {labels.shape}")
            print(f"Labels dtype: {labels.dtype}")
            print(f"Labels - min: {labels.min():.4f}, max: {labels.max():.4f}, mean: {labels.mean():.4f}")
            print(f"Labels - unique values: {torch.unique(labels)}")
            
            outputs_pre_loss = model(neighbor_features)
            print(f"\nModel outputs shape: {outputs_pre_loss.shape}")
            print(f"Outputs - min: {outputs_pre_loss.min():.4f}, max: {outputs_pre_loss.max():.4f}, mean: {outputs_pre_loss.mean():.4f}")
            print(f"Outputs - any NaN: {torch.isnan(outputs_pre_loss).any()}")
            print(f"Outputs - any Inf: {torch.isinf(outputs_pre_loss).any()}")
            
            # Check a few predictions vs labels
            probs = torch.sigmoid(outputs_pre_loss)
            print(f"\nSigmoid outputs (first cell, first 5 proteins): {probs[0, :5]}")
            print(f"True labels (first cell, first 5 proteins): {labels[0, :5]}")
            
            test_loss = criterion(outputs_pre_loss, labels.float())
            print(f"\nLoss value: {test_loss.item()}")
            print(f"Loss is finite: {torch.isfinite(test_loss)}")
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            neighbor_features = batch['neighbor_features'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(neighbor_features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Store predictions
            preds = outputs.cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.vstack(all_preds).astype(int)
    all_labels = np.vstack(all_labels).astype(int)
    #print(all_preds.shape, all_preds[0,:])
    # print(type(all_preds[0,0]))
    
    # Compute metrics
    # Overall metrics
    mae = mean_absolute_error(all_labels.flatten(), all_preds.flatten())
    r2 = r2_score(all_labels.flatten(), all_preds.flatten())
    
    # Per-protein R²
    per_protein_r2 = []
    for i in range(all_labels.shape[1]):
        try:
            r2_i = r2_score(all_labels[:, i], all_preds[:, i])
            per_protein_r2.append(r2_i)
        except:
            per_protein_r2.append(0.0)
    
    mean_protein_r2 = np.mean(per_protein_r2)
    
    # Pearson correlation
    try:
        pearson_r, _ = pearsonr(all_labels.flatten(), all_preds.flatten())
    except:
        pearson_r = 0.0
    
    return {
        'loss': total_loss / len(dataloader),
        'mae': mae,
        'r2': r2,
        'mean_protein_r2': mean_protein_r2,
        'pearson_r': pearson_r,
        'per_protein_r2': per_protein_r2
    }


    # Compute AUROC for each protein independently, then average
    # aurocs = []
    # auprcs = []

    # for i in range(all_labels.shape[1]):
    #     if len(np.unique(all_labels[:, i])) > 1:  # Check if both classes present
    #         try:
    #             auroc_i = roc_auc_score(all_labels[:, i], all_preds[:, i])
    #             aurocs.append(auroc_i)
                
    #             auprc_i = average_precision_score(all_labels[:, i], all_preds[:, i])
    #             auprcs.append(auprc_i)
    #         except:
    #             pass

    # auroc = np.mean(aurocs) if aurocs else 0.0
    # auprc = np.mean(auprcs) if auprcs else 0.0
        
    return {
        'loss': total_loss / len(dataloader),
        'auroc': auroc,
        'auprc': auprc
    }


# ============================================================================
# Main Training Pipeline
# ============================================================================

def get_protein_cols(df, protein_start_col, protein_end_col):
    """ For extracting the columns that contain protein markers"""
    # Extract protein columns
    if isinstance(protein_start_col, tuple) and isinstance(protein_end_col, tuple):
        protein_cols = df.columns[protein_start_col[0]:protein_end_col[0]].tolist()
        for i in range(1, len(protein_start_col)):
            protein_cols += df.columns[protein_start_col[i]:protein_end_col[i]].tolist()
    else:
        start_idx = df.columns.get_loc(protein_start_col)
        end_idx = df.columns.get_loc(protein_end_col) + 1
        protein_cols = df.columns[start_idx:end_idx].tolist()

    return protein_cols

def train_graphformer(df, protein_start_col, protein_end_col, normalize=True,
                      hidden_dim=256, num_layers=3, num_heads=4,
                      batch_size=64, num_epochs=50, lr=1e-4,
                      test_size=0.2, rebuild_knn=True, max_neighbors=5,
                      device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Main function to train GraphFormer model.
    
    Args:
        df: DataFrame with spatial proteomics data
        protein_start_col: Start indices (as tuple)/name of protein columns
        protein_end_col: End indices (as tuple)/name of protein columns
        hidden_dim: Hidden dimension size
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        lr: Learning rate
        test_size: Fraction of data for testing
        rebuild_knn: If True, rebuild KNN within each split to avoid cross-split neighbors
        device: Device to train on
    """
    
    # Extract protein columns
    protein_cols = get_protein_cols(df, protein_start_col, protein_end_col)
    
    print(f"Training on {len(protein_cols)} protein markers")

    # Normalize proteins BEFORE splitting
    if normalize:
        print("Normalizing protein expression (z-score per protein)...")
        scaler = StandardScaler()
        df[protein_cols] = scaler.fit_transform(df[protein_cols])
        print(f"After normalization - Mean: {df[protein_cols].mean().mean():.4f}, Std: {df[protein_cols].std().mean():.4f}")
    
    # Split data by GraphID to avoid leakage
    unique_graphs = df['GraphID'].unique()
    train_graphs, test_graphs = train_test_split(
        unique_graphs, test_size=test_size, random_state=42
    )
    
    train_df = df[df['GraphID'].isin(train_graphs)].copy()
    test_df = df[df['GraphID'].isin(test_graphs)].copy()
    
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Rebuild KNN within each split if requested
    if rebuild_knn:
        print(f"Rebuilding KNN (k={max_neighbors}) relationships within train/test splits...")
        
        def rebuild_knn_for_split(split_df, k=max_neighbors):
            """Rebuild KNN using only cells within this split."""
            split_df = split_df.reset_index(drop=True)
            new_knn = []
            
            for graph_id in split_df['GraphID'].unique():
                graph_mask = split_df['GraphID'] == graph_id
                graph_df = split_df[graph_mask]
                
                if len(graph_df) < k + 1:
                    # Not enough cells, just use all others
                    for idx in graph_df.index:
                        others = [i for i in graph_df.index if i != idx]
                        while len(others) < k:
                            others.append(idx)  # Pad with self
                        new_knn.append(graph_df.iloc[others[:k]]['CellID'].tolist())
                else:
                    # Use spatial coordinates to find neighbors
                    coords = graph_df[['X:X', 'Y:Y', 'Z:Z']].values
                    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
                    distances, indices = nbrs.kneighbors(coords)
                    
                    # Skip first neighbor (self) and get CellIDs
                    for i, idx in enumerate(graph_df.index):
                        neighbor_indices = indices[i][1:]  # Skip self
                        neighbor_cell_ids = graph_df.iloc[neighbor_indices]['CellID'].tolist()
                        new_knn.append(neighbor_cell_ids)
            
            return new_knn
        
        train_df['KNN'] = rebuild_knn_for_split(train_df)
        test_df['KNN'] = rebuild_knn_for_split(test_df)
        print("KNN rebuild complete!")

    # Baselining
    print("\nTesting neighbor average baseline...")
    baseline_r2 = simple_neighbor_average_baseline(test_df, protein_cols)
    
    # Create datasets
    train_dataset = SpatialProteomicsDataset(train_df, protein_cols, max_neighbors=max_neighbors)
    test_dataset = SpatialProteomicsDataset(test_df, protein_cols, max_neighbors=max_neighbors)
    
    # Create dataloaders with custom collate
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, train_dataset.features)
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, test_dataset.features)
    )
    
    # Initialize model
    model = GraphFormer(
        num_proteins=len(protein_cols),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    # Training loop
    best_r2 = -float('inf')  # Changed to -inf for R² (can be negative)
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test Loss: {test_metrics['loss']:.4f}, "
              f"MAE: {test_metrics['mae']:.2f}, "
              f"R²: {test_metrics['r2']:.4f}, "
              f"Mean Protein R²: {test_metrics['mean_protein_r2']:.4f}, "
              f"Pearson r: {test_metrics['pearson_r']:.4f}")
        
        # Save best model based on R²
        if test_metrics['r2'] > best_r2:
            best_r2 = test_metrics['r2']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'r2': best_r2,
                'protein_cols': protein_cols
            }, 'best_graphformer_model.pt')
            print(f"  Saved best model (R²: {best_r2:.4f})")

    ##############################
    ######### DIAGNOSIS ##########
    ##############################

    # After training, look at per-protein performance
    protein_r2 = test_metrics['per_protein_r2']
    protein_names = protein_cols  # Your list of protein column names

    # Get expression statistics for each protein
    protein_stats = []
    for i, name in enumerate(protein_names):
        mean_expr = df[name].mean()
        std_expr = df[name].std()
        max_expr = df[name].max()
        protein_stats.append({
            'name': name,
            'r2': protein_r2[i],
            'mean': mean_expr,
            'std': std_expr,
            'max': max_expr
        })

    # Sort by R²
    protein_stats_sorted = sorted(protein_stats, key=lambda x: x['r2'], reverse=True)

    print("Best predicted proteins:")
    print(f"{'Protein':<50} {'R²':>8} {'Mean':>10} {'Std':>10} {'Max':>10}")
    print("-" * 90)
    for p in protein_stats_sorted[:5]:
        print(f"{p['name']:<50} {p['r2']:>8.4f} {p['mean']:>10.2f} {p['std']:>10.2f} {p['max']:>10.2f}")

    print("\nWorst predicted proteins:")
    print(f"{'Protein':<50} {'R²':>8} {'Mean':>10} {'Std':>10} {'Max':>10}")
    print("-" * 90)
    for p in protein_stats_sorted[-5:]:
        print(f"{p['name']:<50} {p['r2']:>8.4f} {p['mean']:>10.2f} {p['std']:>10.2f} {p['max']:>10.2f}")

    # Check correlation between R² and expression magnitude
    means = [p['mean'] for p in protein_stats]
    r2s = [p['r2'] for p in protein_stats]

    plt.figure(figsize=(10, 6))
    plt.scatter(means, r2s, alpha=0.6)
    plt.xlabel('Mean Expression Level')
    plt.ylabel('R²')
    plt.title('Protein Prediction Performance vs Expression Level')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Baseline')
    plt.legend()
    plt.tight_layout()
    plt.savefig('r2_vs_expression.png')
    plt.show()

    print(f"\nCorrelation between mean expression and R²: {np.corrcoef(means, r2s)[0,1]:.4f}")
        
    return model, train_dataset, test_dataset


# ============================================================================
# Neighbor Average Baseline -- Is there spatial signal?
# ============================================================================

def simple_neighbor_average_baseline(df, protein_cols):
    """Predict each cell as average of its neighbors."""

    predictions = []
    actuals = []
    
    #cell_to_idx = {cell_id: idx for idx, cell_id in enumerate(df['CellID'])}
    cell_to_data = {cell_id: row for cell_id, row in zip(df['CellID'], df[protein_cols].values)}
    
    for _, row in df.iterrows():
        knn_str = row['KNN']
        #if isinstance(knn_str, (list, np.ndarray)):
        neighbors = [int(n) for n in knn_str]
        
        # Get neighbor features
        neighbor_features = []
        for n in neighbors:
            if n in cell_to_data:
                neighbor_features.append(cell_to_data[n])
        
        if neighbor_features:
            pred = np.mean(neighbor_features, axis=0)
        else:
            pred = df[protein_cols].mean().values
        
        predictions.append(pred)
        actuals.append(row[protein_cols].values)
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # r2 = r2_score(actuals.flatten(), predictions.flatten())
    # print(f"Neighbor Average Baseline R²: {r2:.4f}")
    # return r2

    per_protein_r2 = []
    for i, protein in enumerate(protein_cols):
        r2 = r2_score(actuals[:, i], predictions[:, i])
        per_protein_r2.append((protein, r2))
    
    # Sort by R²
    per_protein_r2.sort(key=lambda x: x[1], reverse=True)
    
    print("\nNeighbor Average Baseline - Best Proteins:")
    for protein, r2 in per_protein_r2[:10]:
        print(f"  {protein[:50]:<50} R²: {r2:7.4f}")
    
    print("\nNeighbor Average Baseline - Worst Proteins:")
    for protein, r2 in per_protein_r2[-10:]:
        print(f"  {protein[:50]:<50} R²: {r2:7.4f}")
    
    overall_r2 = r2_score(actuals.flatten(), predictions.flatten())
    mean_protein_r2 = np.mean([r2 for _, r2 in per_protein_r2])
    
    print(f"\nOverall R²: {overall_r2:.4f}")
    print(f"Mean Protein R²: {mean_protein_r2:.4f}")
    
    return per_protein_r2

# ============================================================================
# Usage Details
# ============================================================================

if __name__ == "__main__":
    df = pd.read_csv('CRC_clusters_neighborhoods_markers_cleaned.csv')
    #more layers = wider net of neighbors that influence each cell
    model, train_dataset, test_dataset = train_graphformer(
        df, protein_start_col=(1,57), protein_end_col=(50,65), normalize=True,
        hidden_dim=64, num_layers=3, num_heads=4, batch_size=64, num_epochs=3, lr=1e-4
    ) #hidden dim should be 256, layers 3
    #start end cols 'CD44 - stroma:Cyc_2_ch_2' 'CD138 - plasma cells:Cyc_21_ch_3'
