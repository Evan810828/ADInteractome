"""Convert score maps to ranked gene-pair lists.

Loads gene-gene interaction score maps (pickle files) and converts them
to ranked CSV lists of the top gene pairs by interaction strength.

Example usage:
    python feature/rank_attention.py --score_dir feature/score_maps/
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pickle
import scanpy as sc
import hdf5plugin
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse

from transformer.dataset import VariableLengthSequenceDataset


# load names
names_path = "data/split_train/cell_type_L5_ET.h5ad"
data_cell = sc.read_h5ad(names_path)
filtered_data = data_cell[data_cell.obs['Overall AD neuropathological Change'].isin(['Not AD', 'High'])]
X_filtered = filtered_data.X
y_filtered = filtered_data.obs['Overall AD neuropathological Change']
label_mapping = {'Not AD': 0, 'High': 1}
y_encoded = y_filtered.map(label_mapping)
    
dataset = VariableLengthSequenceDataset(X_filtered, y_encoded)

names = pd.DataFrame(filtered_data.var)

def score_map_to_rank(score_map, file_name):
    print(f"Converting {file_name}...")
    # normalize the score_map
    num_genes = 36601
    score_map = (score_map - torch.mean(score_map)) / torch.std(score_map)
    i_upper, j_upper = torch.triu_indices(num_genes, num_genes, offset=1)
    # Add the lower-left part to the upper-right part using advanced indexing
    score_map[i_upper, j_upper] += score_map[j_upper, i_upper]
    score_map.triu_(diagonal=1)

    flat_matrix = score_map.flatten()

    largest_indices = np.argpartition(flat_matrix, -200000)[-200000:]
    largest_coordinates = np.unravel_index(largest_indices, score_map.shape)

    n = len(largest_coordinates[0])

    ranking = pd.DataFrame({
        "gene1": [get_gene_name(largest_coordinates[0][i], names) for i in range(n)],
        "gene2": [get_gene_name(largest_coordinates[1][i], names) for i in range(n)],
        "avg_score": flat_matrix[largest_indices]
    })
    ranking.sort_values(by="avg_score", ascending=False).to_csv(file_name)


def get_gene_name(index, names):
    return names.iloc[index].name

def get_score_map(filename):
    # load data from pkl
    with open(filename, "rb") as f:
        score_map = pickle.load(f)

    score_map = torch.tensor(score_map).cpu()
    score_map.squeeze(0)
    # attention mean padding
    score_map[0, :] = 0
    score_map[:,0] = 0
        
    score_map_to_rank(score_map,  f"{filename}_ranking.csv")


_parser = argparse.ArgumentParser()
_parser.add_argument("--score_dir", type=str, required=True, help="Directory containing score map files")
_args = _parser.parse_args()

for root, dirs, files in os.walk(_args.score_dir):
    for filename in files:
        if '36601' in filename and '.csv' not in filename:
            full_path = os.path.join(root, filename)
            try:
                get_score_map(full_path)
            except Exception as e:
                print(f"Error processing {full_path}: {e}")
