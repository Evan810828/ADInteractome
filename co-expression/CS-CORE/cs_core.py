import numpy as np
import scanpy as sc
import hdf5plugin
import pandas
from CSCORE import CSCORE
import pandas as pd
import argparse
import torch
import pickle

argparser = argparse.ArgumentParser()
argparser.add_argument("--data", type=str, help="Input file path")
args = argparser.parse_args()
cell_type = args.data.split('/')[-1].split('.')[0].split('_')[-1]

adata = sc.read_h5ad(args.data)

AD_cells = adata[adata.obs['Overall AD neuropathological Change'] == 'High']
NonAD_cells = adata[adata.obs['Overall AD neuropathological Change'] == 'Not AD']

mean_exp = (adata.X).sum(axis=0).A1
mean_exp_df = pd.DataFrame({'gene': adata.var.index, 'mean_expression': mean_exp})
top_genes_df = mean_exp_df.sort_values(by='mean_expression', ascending=False).head(5000)
top_genes_indices = top_genes_df.index.astype(int).to_numpy()

def cs_core(adata, top_genes_indices):
    adata.raw = adata
    res = CSCORE(adata, top_genes_indices)
    np.nan_to_num(res[0], copy=False)

    return res, top_genes_indices

AD_res_tensor, top_genes_indices = cs_core(AD_cells, top_genes_indices)
res_tensor = torch.tensor(AD_res_tensor[1])
NonAD_res_tensor, _ = cs_core(NonAD_cells, top_genes_indices)
res_tensor -= torch.tensor(NonAD_res_tensor[1])

with open(f'co-expression/score_maps/{cell_type}_cs_core.pkl', 'wb') as f:
    pickle.dump(res_tensor, f)
    
def get_gene_name(index, names):
    return names.iloc[index].name
    
def score_to_rank(res_tensor, top_genes_indices, file_name):
    print(f"Converting {file_name}...")
    i_upper, j_upper = torch.triu_indices(res_tensor.shape[0], res_tensor.shape[0], offset=1)
    # Add the lower-left part to the upper-right part using advanced indexing
    res_tensor[i_upper, j_upper] += res_tensor[j_upper, i_upper]
    res_tensor.triu_(diagonal=1)

    flat_matrix = res_tensor.flatten()

    largest_indices = np.argpartition(flat_matrix, -200000)[-200000:]
    largest_coordinates = np.unravel_index(largest_indices, res_tensor.shape)

    n = len(largest_coordinates[0])

    ranking = pd.DataFrame({
        "gene1": [get_gene_name(top_genes_indices[largest_coordinates[0][i]], adata.var) for i in range(n)],
        "gene2": [get_gene_name(top_genes_indices[largest_coordinates[1][i]], adata.var) for i in range(n)],
        "avg_score": flat_matrix[largest_indices]
    })
    
    ranking.sort_values(by="avg_score", ascending=False).to_csv(file_name)
    
score_to_rank(res_tensor, top_genes_indices, f'co-expression/score_maps/{cell_type}_cs_core.csv')