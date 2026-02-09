import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import hdf5plugin
import scanpy as sc
import os
from sklearn.metrics import f1_score
import sys
import argparse
import json
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformer.dataset import VariableLengthSequenceDataset
from transformer.main import load_model, test

class PruningDataset(VariableLengthSequenceDataset):
    def __init__(self, X, y, target_genes): # X is a sparse matrix
        self.seq = []
        self.vals = []
        self.labels = torch.tensor(y)
        self.target_genes = target_genes
    
        X = X.toarray()
        self.X = X
        
        self.apply_mask()
    
    def apply_mask(self):    
        genes_to_add = self.minimum_nonzero_intersection()
        masked_X = self.X.copy()
        filtered_gene_ids = [gid for gid in range(0, 36601) if gid not in genes_to_add]
        masked_X[:, filtered_gene_ids] = 0
        
        for i in tqdm(range(masked_X.shape[0]), desc="Processing data"):
            x = masked_X[i]
            x_tensor = torch.from_numpy(x).float()
            
            non_zero_indices = torch.nonzero(x_tensor, as_tuple=True)[0]
            self.seq.append(non_zero_indices)
            
            scaling_factor = x_tensor[non_zero_indices]
            self.vals.append(scaling_factor)
        self.seq = nn.utils.rnn.pad_sequence([seq.clone().detach() for seq in self.seq], batch_first=True, padding_value=0)+1
        self.vals = nn.utils.rnn.pad_sequence([vals.clone().detach() for vals in self.vals], batch_first=True, padding_value=0)
    
    def minimum_nonzero_intersection(self, importance_ranking="feature/pruning/self_interact.csv"):
        masked_X = self.X.copy()
        filtered_gene_ids = [gid for gid in range(0, 36601) if gid in self.target_genes]
        masked_X[:, filtered_gene_ids] = 0
        seq = []
        for i in tqdm(range(masked_X.shape[0]), desc="Processing data"):
            x = masked_X[i]
            x_tensor = torch.from_numpy(x).float()
            
            non_zero_indices = torch.nonzero(x_tensor, as_tuple=True)[0]
            seq.append(non_zero_indices)
        seq = nn.utils.rnn.pad_sequence([s.clone().detach() for s in seq], batch_first=True, padding_value=0)
        
        importance_ranking = pd.read_csv(importance_ranking)
        importance_ranking = importance_ranking.sort_values(by="avg_score", ascending=True)
        genes_to_add = []
        for gene in importance_ranking['gene_index']:
            genes_to_add.append(gene)
            
            bool_matrix = np.isin(seq, genes_to_add)
            has_intersection = bool_matrix.any(axis=1)
            if has_intersection.all():
                break
            
        print(f"Add {len(genes_to_add)} genes to the model")
        
        for gene in self.target_genes:
            if gene not in genes_to_add:
                genes_to_add.append(gene)
                
        # pdb.set_trace()        
                
        return genes_to_add

def load_data(data_path, target_genes):
    print(f"Loading data with target genes: {target_genes}")
    data_cell = sc.read_h5ad(data_path)

    filtered_data = data_cell[data_cell.obs['Overall AD neuropathological Change'].isin(['Not AD', 'High'])]
    X_filtered = filtered_data.X
    y_filtered = filtered_data.obs['Overall AD neuropathological Change']
    label_mapping = {'Not AD': 0, 'High': 1}
    y_encoded = y_filtered.map(label_mapping)
    gene_ids = pd.DataFrame(filtered_data.var)
    target_genes = [gene_ids.index.get_loc(gene_name) for gene_name in target_genes]
    # print(f"Target genes indices: {target_genes}")
    
    dataset = PruningDataset(X_filtered, y_encoded, target_genes)
    return dataset

def load_target_genes(target_genes_path, mode="list"):
    if mode == "list":
        with open(target_genes_path, 'r') as f:
            cell_type = target_genes_path.split("/")[-1].split(".")[0]
            # target_genes = [line.strip() for line in f if line.strip()]
            target_genes = [line.strip().split(",")[1] for line in f if line.strip()]
        return cell_type, target_genes
    elif mode == "pair":
        with open(target_genes_path, 'r') as f:
            cell_type = target_genes_path.split("/")[-1].split(".")[0]
            gene_pair_combinations = [line.strip().split("-") for line in f if line.strip()]
        return cell_type, gene_pair_combinations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="output_models/global_training/restful-dew-11/epoch_2_batch_3473.pth", help='Path to the model file')
    # parser.add_argument('--data_path', type=str, default="data/split_test/cell_type_L6_CT.h5ad", help='Path to the data file')
    parser.add_argument('--target_genes', type=str, default="feature/pruning", help='Path to the target genes file')
    parser.add_argument('--config_path', type=str, default="config.json", help='Path to the config file')
    parser.add_argument('--mode', type=str, default="pair", choices=["list", "pair"], help='Mode of the model')
    
    args = parser.parse_args()
    
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda")
        
    results = {}
    
    model = load_model(args.model_path, json.load(open(args.config_path)))
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
    model.to(device)
    
    if args.mode == "list":        
        for file in os.listdir(args.target_genes):
            if file.endswith(".txt"):
                cell_type, target_genes = load_target_genes(os.path.join(args.target_genes, file))
                print(f"Testing {cell_type} with {len(target_genes)} target genes")
                
                dataset = load_data(f"data/split_test/cell_type_{cell_type}.h5ad", target_genes)
                dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
                _, acc, f1, y = test(model, dataloader, criterion, device)
                
                _dataset = load_data(f"data/split_test/cell_type_{cell_type}.h5ad", [])
                _dataloader = DataLoader(_dataset, batch_size=1024, shuffle=False)
                _, _acc, _f1, _y = test(model, _dataloader, criterion, device)
                
                acc_diff = acc - _acc
                results[cell_type] = {
                    "accuracy difference": acc_diff,
                }
                
        with open("feature/pruning/list_pruning_results.json", "w") as f:
            json.dump(results, f, indent=4)
    elif args.mode == "pair":
        cell_type, gene_pair_combinations = load_target_genes(args.target_genes, mode="pair")
        print(f"Testing {cell_type} with {len(gene_pair_combinations)} target gene combinations")
        
        # baseline
        _dataset = load_data(f"data/split_test/cell_type_{cell_type}.h5ad", [])
        _dataloader = DataLoader(_dataset, batch_size=1024, shuffle=False)
        _, _acc, _f1, _y = test(model, _dataloader, criterion, device)
        
        for gene_pair in gene_pair_combinations:
            dataset = load_data(f"data/split_test/cell_type_{cell_type}.h5ad", gene_pair)
            dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
            _, acc, f1, y = test(model, dataloader, criterion, device)
            
            acc_diff = acc - _acc
            results[gene_pair[0]+"-"+gene_pair[1]] = {
                "accuracy difference": acc_diff,
            }
                
        with open(f"feature/pruning/pair_pruning_results_{cell_type}_{args.mode}.json", "w") as f:
            json.dump(results, f, indent=4)
    
    