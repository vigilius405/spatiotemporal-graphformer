# Spatial Proteomics GraphFormer

A geometric deep learning framework for analyzing spatial dependencies in protein expression data from colorectal cancer tissue samples. This project investigates whether protein expression in individual cells can be predicted from their spatial neighborhood context with graph neural networks.

## Project Overview

### Research Questions
1. Is a GraphFormer architecture appropriate for modeling and predicting protein expression in cell clusters?
2. Which proteins are most dependent on (predictable from) surrounding cells?
3. Which cell types are most important in influencing their spatial neighborhoods?

### Dataset
- **Source**: ["Coordinated cellular neighborhoods orchestrate antitumoral immunity at the colorectal cancer invasive front"](https://data.mendeley.com/datasets/mpjzbtfgfr/1)
- **Data**: Multiplexed imaging mass cytometry (IMC) data
- **Features**: 57 protein markers per cell, spatial coordinates (X, Y, Z), cell metadata
- **Scale**: ~250,000+ cells across multiple tissue samples (GraphIDs)

### Key Findings
Our analysis found that protein expression in colorectal cancer cells is strongly predictable from immediate spatial neighbors (k=5-50) for some protein markers, while being completely unpredictable for others. The model is trained on the objective of predicting the protein expression in the current cell based on surrounding cells.

**GraphFormer Insights:** The GraphFormer architecture outperformed a veriety of models (GAT, CDN, GraphSAGE, simple average baselines). However, naive training of the GNN proved extreme overfitting of the data and required strong regularization techniques. When increasing the model input graph from 5 to 50 immediate spatial neighbors, 20 was determined to be the optimal for training models. Increasing the number of neighbors in the puts increased from 5 to 10 to 20, but did not significantly increase from 20 to 50. It is hypothesized that there is relevant graph signal from up to ~20 spatial neighbors, but increasing to 50 provides less meaningful improvements to signal over the reduction in training samples.

**Biological Insights:** Shown below are the top 5 best and bottom 5 worst predicted protein markers. Going through some of these proteins, we can see why their correlation is physiologically relevant: cytokeratin and CD34 are implicated in intercellualr adhesion, meaning cells that have high levels of these proteins are likely attached to other cells with high levels. The least correlated proteins are mostly cell surface markers of immune cells; this makes sense, as immune cells are more likely than other cell types to be traveling alone, surrounded by different cell types. Other than this relationship with immune cells, we did not find much to indicate that certain cell types were more correlated with their neighbors than others. Note that these results should also be validated on healthy tissue and with larger numbers of cells per sample. This dataset serves primarily as a demonstration that spatial GraphFormers can reveal meaningful patterns in spatial proteomics within microenvironments of roughly 20 cells.

## Model Details

### Best Predicted Proteins

| Protein                                      | R²      |
|---------------------------------------------|---------|
| Cytokeratin - epithelia:Cyc_10_ch_2        | 0.9622  |
| Na-K-ATPase - membranes:Cyc_9_ch_2         | 0.8859  |
| CD34 - vasculature:Cyc_20_ch_3             | 0.8695  |
| CD138 - plasma cells:Cyc_21_ch_3           | 0.8607  |
| CD68 - macrophages:Cyc_18_ch_4             | 0.8484  |

### Worst Predicted Proteins

| Protein                                      | R²       |
|---------------------------------------------|----------|
| CD25 - IL-2 Ra:Cyc_11_ch_4                  | 0.0875   |
| CD2 - T cells:Cyc_7_ch_4                    | 0.0499   |
| LAG-3 - checkpoint:Cyc_8_ch_4               | 0.0305   |
| CD8 - cytotoxic T cells:Cyc_3_ch_2          | -0.0011  |
| MUC-1 - epithelia:Cyc_7_ch_2                | -0.0021  |   

<img width="598" height="338" alt="Results of Final Model on Proteins" src="https://github.com/user-attachments/assets/db436df6-467f-42bc-b52f-53af62461c8f" />

### Reproducibility

[Model weights](./graphformer_weights.pt) are saved.

Model training can be reproduced with the following hyperparameters and seeds.

| Hyperparameter   | Value    |
|-----------------|----------|
| max_neighbors    | 20       |
| num_layers       | 3        |
| lr               | 0.001    |
| dropout          | 0.1      |
| l2_reg           | 1e-05    |
| hidden_dim       | 256      |
| num_heads        | 4        |
| seed             | 42       |
| epochs           | 10       |


## Repository Structure
```
.
├── data_cleaning.py              # Data preprocessing and KNN computation
├── graphformer.py                # GraphFormer model architecture and dataset class
├── training.py                   # GraphFormer training pipeline and baselines
├── training_baselines.py         # GCN, GAT, GraphSAGE baseline implementations
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd spatiotemporal-graphformer

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

First, prepare the dataset by computing k-nearest neighbors for each cell within its tissue sample:
```bash
python data_cleaning.py
```

Combines split CSV files and removes unnecessary features; performs KNN and creates unique tissue sample identifiers.

Then, the graphformer model can be trained. Hyperparameters can be updated in the main function as desired. This will also run the simple neighbor average baseline.
```bash
python training.py
```

Similarly, the more advanced baselines can be tuned in the main function of training_baselines.py and run as follows:
```bash
python training_baselines.py
```

Example usage of model training and hyperparameter tuning can be followed in [Jupyter Notebook](./exploration.ipynb).

### 2. Evaluation Metrics

Models report:
- **MSE Loss**: Mean squared error (training objective)
- **MAE**: Mean absolute error in original units
- **R²**: Coefficient of determination (0-1, higher is better; can be negative if worse than mean baseline)
- **Mean Protein R²**: Average R² across individual proteins
- **Pearson r**: Linear correlation between predictions and actuals
- **Per-protein R²**: Individual R² for each protein marker (useful for identifying spatially-dependent proteins)
