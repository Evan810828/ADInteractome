import pickle
import scanpy as sc
import hdf5plugin
import torch
import pickle
import torch.nn as nn
import pandas as pd
import hdf5plugin
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import json

class VariableLengthSequenceDataset(Dataset):
    def __init__(self, X, y): # X is a sparse matrix
        self.seq = []
        self.vals = []
        self.labels = torch.tensor(y)

        print("Initializing dataset...")
        for i in tqdm(range(X.shape[0]), desc="Processing data"):
            x = X[i].toarray()
            x_tensor = torch.from_numpy(x).float()
            
            non_zero_indices = torch.nonzero(x_tensor, as_tuple=True)[1]
            self.seq.append(non_zero_indices)
            
            scaling_factor = x_tensor[0][non_zero_indices]
            self.vals.append(scaling_factor)
            
        self.seq = nn.utils.rnn.pad_sequence([seq.clone().detach() for seq in self.seq], batch_first=True, padding_value=0)+1
        self.vals = nn.utils.rnn.pad_sequence([vals.clone().detach() for vals in self.vals], batch_first=True, padding_value=0)

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, index):
        sequence = self.seq[index]
        scaling_factor = self.vals[index]
        label = self.labels[index]
        return sequence, scaling_factor, label


# load names
gene_name_dict = json.load(open("scGPT/gene_vocab.json"))

def score_map_to_rank(score_map, file_name):
    print(f"Converting {file_name}...")
    i_upper, j_upper = torch.triu_indices(score_map.shape[0], score_map.shape[0], offset=1)
    # Add the lower-left part to the upper-right part using advanced indexing
    score_map[i_upper, j_upper] += score_map[j_upper, i_upper]
    score_map.triu_(diagonal=1)

    flat_matrix = score_map.flatten()

    largest_indices = np.argpartition(flat_matrix, -200000)[-200000:]
    largest_coordinates = np.unravel_index(largest_indices, score_map.shape)

    n = len(largest_coordinates[0])

    ranking = pd.DataFrame({
        "gene1": [gene_name_dict[f'{largest_coordinates[0][i]}'] for i in range(n)],
        "gene2": [gene_name_dict[f'{largest_coordinates[1][i]}'] for i in range(n)],
        "avg_score": flat_matrix[largest_indices]
    })

    ranking.sort_values(by="avg_score", ascending=False).to_csv(file_name)

def get_score_map(filename):
    # load data from pkl
    with open(filename, "rb") as f:
        score_map = pickle.load(f)

    score_map = torch.tensor(score_map).cpu()
    # attention mean padding
    score_map[60694, :] = 0
    score_map[:,60694] = 0
            
    score_map_to_rank(score_map,  f"{filename}_ranking.csv")

for filename in os.listdir("scGPT"):
    if 'IT' in filename:
        get_score_map(f"scGPT/{filename}")