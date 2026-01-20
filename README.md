# spatiotemporal-graphformer
Learning developmental spatiotemporal proteomics

# Spatial Proteomics GraphFormer

A deep learning framework for analyzing spatial dependencies in protein expression data from colorectal cancer tissue samples. This project investigates whether protein expression in individual cells can be predicted from their spatial neighborhood context using graph neural networks.

## Project Overview

### Research Questions
1. Is a GraphFormer architecture appropriate for modeling and predicting protein expression in cell clusters?
2. Which proteins are most dependent on (predictable from) surrounding cells?
3. Which cell types are most important in influencing their spatial neighborhoods?

### Dataset
- **Source**: "Coordinated cellular neighborhoods orchestrate antitumoral immunity at the colorectal cancer invasive front" (https://data.mendeley.com/datasets/mpjzbtfgfr/1)
- **Data**: Multiplexed imaging mass cytometry (IMC) data
- **Features**: 57 protein markers per cell, spatial coordinates (X, Y, Z), cell metadata
- **Scale**: ~250,000+ cells across multiple tissue samples (GraphIDs)

### Key Finding
**Negative Result**: Our analysis found from a range of model complexities and baselines that protein expression in colorectal cancer cells is **NOT strongly predictable from immediate spatial neighbors** (k=2-20).

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

### 4. Evaluation Metrics

Models report:
- **MSE Loss**: Mean squared error (training objective)
- **MAE**: Mean absolute error in original units
- **R²**: Coefficient of determination (0-1, higher is better; can be negative if worse than mean baseline)
- **Mean Protein R²**: Average R² across individual proteins
- **Pearson r**: Linear correlation between predictions and actuals
- **Per-protein R²**: Individual R² for each protein marker (useful for identifying spatially-dependent proteins)