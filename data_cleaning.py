import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def clean_data(dataset: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Generates a dataframe with the following columns:
        Graph ID    Cell ID     X   Y   Z   Neighbors   49 protein cols     8 more protein cols     15 +cols
    
    :param dataset: Dataset from "Coordinated cellular neighborhoods orchestrate antitumoral immunity at the colorectal cancer invasive front"
    :type dataset: pd.DataFrame
    :param k: k nearest neighbors for calculating neighbors
    :type k: int
    """
    new_df = dataset.copy()
    new_df['GraphID'] = new_df['File Name'] + ' ' + new_df['patients'].astype(str) + ' ' + new_df['Region'] + ' ' + new_df['tile_nr:tile_nr'].astype(str)

    #DONT WANT to drop CellID?
    to_drop = [#'CellID', 
               'ClusterID', 'EventID', 'File Name', 'Region', 'TMA_AB', 
               'TMA_12', 'Index in File', 'groups', 'patients', 'spots', 
               #'CD44 - stroma:Cyc_2_ch_2', 'FOXP3 - regulatory T cells:Cyc_2_ch_3', 
               #'CD8 - cytotoxic T cells:Cyc_3_ch_2', 'p53 - tumor suppressor:Cyc_3_ch_3', 
               #'GATA3 - Th2 helper T cells:Cyc_3_ch_4', 'CD45 - hematopoietic cells:Cyc_4_ch_2', 
               #'T-bet - Th1 cells:Cyc_4_ch_3', 'beta-catenin - Wnt signaling:Cyc_4_ch_4', 
               #'HLA-DR - MHC-II:Cyc_5_ch_2', 'PD-L1 - checkpoint:Cyc_5_ch_3', 
               #'Ki67 - proliferation:Cyc_5_ch_4', 'CD45RA - naive T cells:Cyc_6_ch_2', 
               #'CD4 - T helper cells:Cyc_6_ch_3', 'CD21 - DCs:Cyc_6_ch_4', 'MUC-1 - epithelia:Cyc_7_ch_2', 
               #'CD30 - costimulator:Cyc_7_ch_3', 'CD2 - T cells:Cyc_7_ch_4', 'Vimentin - cytoplasm:Cyc_8_ch_2', 
               #'CD20 - B cells:Cyc_8_ch_3', 'LAG-3 - checkpoint:Cyc_8_ch_4', 'Na-K-ATPase - membranes:Cyc_9_ch_2', 
               #'CD5 - T cells:Cyc_9_ch_3', 'IDO-1 - metabolism:Cyc_9_ch_4', 'Cytokeratin - epithelia:Cyc_10_ch_2', 
               #'CD11b - macrophages:Cyc_10_ch_3', 'CD56 - NK cells:Cyc_10_ch_4', 'aSMA - smooth muscle:Cyc_11_ch_2', 
               #'BCL-2 - apoptosis:Cyc_11_ch_3', 'CD25 - IL-2 Ra:Cyc_11_ch_4', 'CD11c - DCs:Cyc_12_ch_3', 
               #'PD-1 - checkpoint:Cyc_12_ch_4', 'Granzyme B - cytotoxicity:Cyc_13_ch_2', 
               #'EGFR - signaling:Cyc_13_ch_3', 'VISTA - costimulator:Cyc_13_ch_4', 
               #'CD15 - granulocytes:Cyc_14_ch_2', 'ICOS - costimulator:Cyc_14_ch_4', 
               #'Synaptophysin - neuroendocrine:Cyc_15_ch_3', 'GFAP - nerves:Cyc_16_ch_2', 
               #'CD7 - T cells:Cyc_16_ch_3', 'CD3 - T cells:Cyc_16_ch_4', 'Chromogranin A - neuroendocrine:Cyc_17_ch_2', 
               #'CD163 - macrophages:Cyc_17_ch_3', 'CD45RO - memory cells:Cyc_18_ch_3', 'CD68 - macrophages:Cyc_18_ch_4', 
               #'CD31 - vasculature:Cyc_19_ch_3', 'Podoplanin - lymphatics:Cyc_19_ch_4', 'CD34 - vasculature:Cyc_20_ch_3', 
               #'CD38 - multifunctional:Cyc_20_ch_4', 'CD138 - plasma cells:Cyc_21_ch_3', 'cell_id:cell_id', 'tile_nr:tile_nr', 
               #'X:X', 'Y:Y', 
               'X_withinTile:X_withinTile', 'Y_withinTile:Y_withinTile', 
               #'Z:Z', 'size:size', 'HOECHST1:Cyc_1_ch_1', 'CDX2 - intestinal epithelia:Cyc_2_ch_4', 
               #'Collagen IV - bas. memb.:Cyc_12_ch_2', 'CD194 - CCR4 chemokine R:Cyc_14_ch_3', 
               #'MMP9 - matrix metalloproteinase:Cyc_15_ch_2', 'CD71 - transferrin R:Cyc_15_ch_4', 
               #'CD57 - NK cells:Cyc_17_ch_4', 'MMP12 - matrix metalloproteinase:Cyc_21_ch_4', 
               #'DRAQ5:Cyc_23_ch_4', 
               'Profile_Homogeneity:Fiter1', 'ClusterSize', 'ClusterName', 'neighborhood10', 
               #'CD4+ICOS+', 'CD4+Ki67+', 'CD4+PD-1+', 'CD68+CD163+ICOS+', 
               #'CD68+CD163+Ki67+', 'CD68+CD163+PD-1+', 'CD68+ICOS+', 'CD68+Ki67+', 'CD68+PD-1+', 
               #'CD8+ICOS+', 'CD8+Ki67+', 'CD8+PD-1+', 'Treg-ICOS+', 'Treg-Ki67+', 'Treg-PD-1+', 
               'neighborhood number final', 'neighborhood name']

    new_df.drop(to_drop, axis=1)
    coords = new_df[['X:X','Y:Y','Z:Z']].to_numpy()
    knn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    knn.fit(coords)
    _, indices = knn.kneighbors(coords)
    cell_ids = new_df["CellID"].values

    neighbor_ids = [
        cell_ids[idx[1:]].tolist()   # skip self (idx[0])
        for idx in indices
    ]

    new_df['KNN'] = neighbor_ids

    return new_df

def split_file(dataset: pd.DataFrame) -> None:
    """
    For reducing file size to 100mb for git upload
    """
    n = len(dataset) // 3
    first_part = dataset.iloc[:n,:]
    second_part = dataset.iloc[n:(2 * n),:]
    third_part = dataset.iloc[(2 * n):,:]
    first_part.to_csv('CRC_clusters_neighborhoods_markers_1.csv', index=False)
    second_part.to_csv('CRC_clusters_neighborhoods_markers_2.csv', index=False)
    third_part.to_csv('CRC_clusters_neighborhoods_markers_3.csv', index=False)

if __name__ == '__main__':
    # dataset = pd.read_csv('CRC_clusters_neighborhoods_markers.csv', index_col=0)
    # split_file(dataset)
    d1 = pd.read_csv('CRC_clusters_neighborhoods_markers_1.csv')
    d2 = pd.read_csv('CRC_clusters_neighborhoods_markers_2.csv')
    d3 = pd.read_csv('CRC_clusters_neighborhoods_markers_3.csv')
    dataset = pd.concat([d1, d2, d3])
    k = 5
    df_knn = clean_data(dataset, k)

