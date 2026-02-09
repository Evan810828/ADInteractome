import pandas as pd
import numpy as np
import hdf5plugin
import scanpy as sc
import anndata
import torch
import torch.nn as nn
import argparse
import sys
import os
import pdb
from tqdm import tqdm

from mlp.mlp import MLP

def load_data(data_path: str):
    data_cell = sc.read_h5ad(data_path)
    keep = data_cell.obs["Overall AD neuropathological Change"].isin(["Not AD", "High"])
    filtered_data = data_cell[keep]
    X_filtered = filtered_data.X  # sparse matrix
    
    return X_filtered.todense()

def load_model(model_path: str):
    model = MLP(num_genes=36601)
    # load model weights
    checkpoint = torch.load(model_path, map_location='cuda')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if k.startswith("module."):
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    
    return model

def matric2dic(hessian,k):
    IS = {}
    for i in range(len(hessian[0])):
        for j in range(i+1, len(hessian[0])):
            interation = 'Interaction: '
            interation = interation + str(i + 1) + ' ' + str(j + 1) + ' '
            IS[interation] = hessian[i][j]
    Sorted_IS = [(k, IS[k]) for k in sorted(IS, key=IS.get, reverse=True)]
    return IS, Sorted_IS

def delta_main(predictor, x, baseline, main_index):
    T = baseline.clone()
    Ti = baseline.clone(); Ti[main_index] = x[main_index]
    input = torch.cat([Ti, T]).reshape(2, -1)
    with torch.no_grad(): # to prevent gradients build up in memeory
        output = predictor(input)[0] # NOTE
    return output[0].item() - output[1].item()

def deltaF_v1(predictor, x, baseline, perm):
    num_gene = x.shape[0]
    shapleyis = torch.zeros(num_gene, num_gene)
    T_base = baseline.clone()
    indices = torch.triu_indices(num_gene, num_gene, 1) # all i,j pairs in the original double for loop order
    all_inputs = torch.zeros((4, num_gene * (num_gene - 1) // 2, num_gene))
    for idx, (i, j) in enumerate(zip(indices[0], indices[1])):
        T = perm[:i] if i > 0 else []
        T_base[T] = x[T]
        interaction = perm[torch.tensor([i,j])]
        Tij = T_base.clone()
        Tij[interaction] = x[interaction]
        Ti = T_base.clone()
        Ti[interaction[0]] = x[interaction[0]]
        Tj = T_base.clone()
        Tj[interaction[1]] = x[interaction[1]]
        all_inputs[0, idx] = Tij
        all_inputs[1, idx] = Ti
        all_inputs[2, idx] = Tj
        all_inputs[3, idx] = T_base
    with torch.no_grad(): # to prevent gradients build up in memeory
        outputs = predictor(all_inputs.view(-1, num_gene))[0].view(4, -1) # NOTE
    results = outputs[0] - outputs[1] - outputs[2] + outputs[3]
    shapleyis[perm[indices[0]], perm[indices[1]]] = results
    return shapleyis

def ShapleyValue(predictor, x, baseline):
    num_gene = x.shape[0]
    shapleyvalue = torch.zeros([num_gene])
    for i in range(num_gene):
        #print(str(i) + "_ShapleyValue")
        shapleyvalue[i] = delta_main(predictor, x, baseline, [i])
    return shapleyvalue

def ShapleyIS(predictor, x, baseline, num_permutation):
    num_gene = x.shape[0]
    SHAPLEYIS = torch.zeros([num_gene, num_gene])
    for _ in range(num_permutation):
        perm = torch.randperm(num_gene)
        shapleyis = deltaF_v1(predictor, x, baseline, perm)
        SHAPLEYIS += shapleyis
    SHAPLEYIS = (SHAPLEYIS + SHAPLEYIS.T) / num_permutation
    SHAPLEYIS = SHAPLEYIS +  torch.diag(ShapleyValue(predictor, x, baseline)) # O(n)
    return SHAPLEYIS

def GlobalSIS(predictor, X, baseline, num_permutation = 2):
    X = X.to('cpu')
    baseline = baseline.to('cpu')
    num_individual, num_gene = X.shape
    Shapely = torch.zeros([num_gene, num_gene]) # maybe use tensor
    feature_importance = torch.zeros([num_individual, num_gene])
    for i in tqdm(range(num_individual)):
        print(str(i) + "_GlobalSIS")    
        x = X[i]
        current_row_shapley = abs(ShapleyIS(predictor, x, baseline, num_permutation))
        Shapely = Shapely + current_row_shapley
        feature_importance[i] = torch.diag(current_row_shapley)
    Shapely = Shapely / num_individual
    GlobalSIS, topGlobalSIS = matric2dic(Shapely, 10)
    return GlobalSIS, topGlobalSIS, Shapely, feature_importance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_type', required=True)
    args = parser.parse_args()
    
    model_path = f"output_models/MLP/{args.cell_type}/mlp_epoch_9.pth"
    model = load_model(model_path)
    model = model.to('cuda')
    model.eval()
    
    train_data_path = f"data/split_train/cell_type_{args.cell_type}.h5ad"
    test_data_path = f"data/split_test/cell_type_{args.cell_type}.h5ad"
    train_x = load_data(train_data_path)
    test_x= load_data(test_data_path)
    train_x = torch.tensor(train_x, dtype=torch.float32).to('cuda')
    test_x = torch.tensor(test_x, dtype=torch.float32).to('cuda')
    
    
    baseline = model(test_x)
    baseline = baseline.mean(dim=0)
    
    model_func = lambda x: (model(x), None)
    
    GlobalSIS, topGlobalSIS, Interaction_matrix,feature_importance = GlobalSIS(model_func, train_x, baseline)

    pd.DataFrame(Interaction_matrix).to_csv(f"output/{args.cell_type}_shap_ranking.csv")
