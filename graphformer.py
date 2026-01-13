import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import ast

# ============================================================================
# Dataset Class
# ============================================================================

class SpatialProteomicsDataset(Dataset):
    """Dataset for spatial proteomics with graph structure."""
    
    def __init__(self, df, protein_cols, max_neighbors=5):
        """
        Args:
            df: DataFrame with CellID, KNN, and protein columns
            protein_cols: List of column names for protein markers
            max_neighbors: Maximum number of neighbors to consider
        """
        self.df = df.reset_index(drop=True)
        self.protein_cols = protein_cols
        self.max_neighbors = max_neighbors
        
        # Create mapping from CellID to index
        self.cell_to_idx = {cell_id: idx for idx, cell_id in enumerate(df['CellID'])}
        
        # Extract protein features
        self.features = torch.FloatTensor(df[protein_cols].values)
        self.labels = self.features.clone()  # For self-supervised learning
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get neighbors
        knn_str = self.df.iloc[idx]['KNN']
        
        # Handle different KNN formats
        if isinstance(knn_str, (list, np.ndarray)):
            # Already a list (from rebuild_knn or direct data)
            neighbors = [int(n) for n in knn_str]
        elif isinstance(knn_str, str):
            try:
                # Try parsing as Python literal
                neighbors = ast.literal_eval(knn_str)
                if isinstance(neighbors, (list, tuple)):
                    neighbors = [int(n) for n in neighbors]
                else:
                    neighbors = [int(neighbors)]
            except (ValueError, SyntaxError):
                # Fallback: parse as comma-separated values
                try:
                    neighbors = [int(x.strip()) for x in knn_str.strip('[]').split(',')]
                except:
                    print(f"Warning: Could not parse KNN for cell at index {idx}: {knn_str}")
                    neighbors = []
        else:
            neighbors = []
            
        # Map neighbor CellIDs to indices, ONLY keeping valid ones
        neighbor_indices = []
        for n in neighbors:
            if n in self.cell_to_idx:
                neighbor_indices.append(self.cell_to_idx[n])
        
        # If we have no valid neighbors OR fewer than max_neighbors, use self-loops
        if len(neighbor_indices) == 0:
            # All neighbors invalid - use only self
            neighbor_indices = [idx] * self.max_neighbors
        elif len(neighbor_indices) < self.max_neighbors:
            # Some valid neighbors - repeat them to fill max_neighbors
            while len(neighbor_indices) < self.max_neighbors:
                neighbor_indices.append(neighbor_indices[len(neighbor_indices) % len(neighbor_indices)])
            
        neighbor_indices = torch.LongTensor(neighbor_indices[:self.max_neighbors])
        
        return {
            'cell_idx': idx,
            'features': self.features[idx],
            'neighbor_indices': neighbor_indices,
            'labels': self.labels[idx]
        }

# ============================================================================
# GraphFormer Model
# ============================================================================

class GraphAttention(nn.Module):
    """Graph attention layer for spatial context (neighbor-to-neighbor attention)."""
    
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0
        
        # All projections now operate on neighbor features only
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, neighbor_features):
        """
        Args:
            neighbor_features: (batch, num_neighbors, in_dim)
        Returns:
            Aggregated neighbor representation: (batch, out_dim)
        """
        B, N, _ = neighbor_features.shape
        
        # All neighbors attend to all other neighbors (including themselves)
        # Queries, keys, and values all come from neighbors
        q = self.q_proj(neighbor_features).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(neighbor_features).view(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(neighbor_features).view(B, N, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (B, num_heads, N, head_dim)
        k = k.transpose(1, 2)  # (B, num_heads, N, head_dim)
        v = v.transpose(1, 2)  # (B, num_heads, N, head_dim)
        
        # Compute attention scores (neighbor-to-neighbor)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)  # (B, N, out_dim)
        out = self.out_proj(out)
        
        # Mean pooling over neighbors to get single representation
        out = out.mean(dim=1)  # (B, out_dim)
        
        return out


class GraphFormer(nn.Module):
    """GraphFormer model for protein marker prediction."""
    
    def __init__(self, num_proteins, hidden_dim=256, num_layers=3, 
                 num_heads=4, dropout=0.1):
        super().__init__()
        
        self.num_proteins = num_proteins
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(num_proteins, hidden_dim)
        
        # Graph attention layers
        self.layers = nn.ModuleList([
            GraphAttention(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Feedforward networks
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_proteins)
        )
        
    def forward(self, neighbor_features):
        """
        Args:
            neighbor_features: (batch, num_neighbors, num_proteins)
        Returns:
            Predicted protein markers: (batch, num_proteins)
        """
        # Project neighbors to hidden dimension
        B, N, _ = neighbor_features.shape
        neighbor_h = self.input_proj(neighbor_features.reshape(-1, self.num_proteins))
        neighbor_h = neighbor_h.reshape(B, N, self.hidden_dim)
        
        # Apply graph transformer layers
        # Start with mean pooling of neighbor features as initial representation
        x = neighbor_h.mean(dim=1)  # (B, hidden_dim)
        
        for attn, norm, ffn, ffn_norm in zip(
            self.layers, self.norms, self.ffns, self.ffn_norms
        ):
            # Graph attention with residual (neighbor-to-neighbor attention)
            attn_out = attn(neighbor_h)
            x = norm(x + attn_out)
            
            # Feedforward with residual
            ffn_out = ffn(x)
            x = ffn_norm(x + ffn_out)
        
        # Output projection
        out = self.output_proj(x)
        
        return out