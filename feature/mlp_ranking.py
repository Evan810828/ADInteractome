import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import pandas as pd
import hdf5plugin
import scanpy as sc

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


names_path = "data/split_train/cell_type_L5_ET.h5ad"
data_cell = sc.read_h5ad(names_path)
filtered_data = data_cell[data_cell.obs['Overall AD neuropathological Change'].isin(['Not AD', 'High'])]

names = pd.DataFrame(filtered_data.var)


def preprocess_weights(weights):
    w_later = torch.abs(weights[-1])
    w_input = torch.abs(weights[0])
    for i in range(len(weights) - 2, 0, -1):
        w_later = torch.matmul(w_later, torch.abs(weights[i]))

    return w_input, w_later


class CalID(torch.nn.Module):
    def __init__(self, w_later) -> None:
        super().__init__()
        self.w_later = torch.nn.Parameter(w_later)

    def forward(self, batch_w_input):
        batch_w_input_min = torch.min(batch_w_input, dim=1).values
        before_agg_strength = torch.matmul(batch_w_input_min, self.w_later)
        # NOTE for Yuntao:
        # here we explicitly select the column 1 and only use it for calculate the strength.
        # In this way, all other classes are disabled.
        col_to_be_selected = [1]
        before_agg_strength = before_agg_strength[:, col_to_be_selected]
        strength = before_agg_strength.sum(dim=-1)
        return strength


class GeneDataset(Dataset):
    def __init__(self, w_input) -> None:
        super(GeneDataset, self).__init__()
        self.w_input = w_input
        # self.pair_indices = torch.combinations(range(w_input.shape[0]), 2)
        candi_nums = w_input.shape[0]
        self.pair_indices = torch.combinations(torch.arange(start=0, end=candi_nums, dtype=torch.long), 2)
        

    def __getitem__(self, index):
        pair_indices = self.pair_indices[index]
        return pair_indices, self.w_input[pair_indices]

    def __len__(self):
        return len(self.pair_indices)


def interpret_pairwise_interactions(w_input, w_later, per_device_batch_size):
    interaction_ranking = pd.DataFrame(columns=['gene1', 'gene2', 'avg_score'])
    device = torch.device("cuda:0")
    # w_input.shape : #input_feats x #embed_dim
    # w_later.shape : #embed_dim x #classes
    w_input, w_later = w_input.T.half(), w_later.T.half()
    dataset = GeneDataset(w_input)
    batch_size = per_device_batch_size * torch.cuda.device_count()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    id_module = CalID(w_later)
    id_module = id_module.to(device)
    if torch.cuda.device_count() > 1:
        id_module = torch.nn.DataParallel(id_module)
        # disable cuda:2
        
        print("Using DataParallel:", torch.cuda.device_count())
    
    temp_data = []
    for indices, batch_w_input in tqdm(loader):
        # batch_w_input.shape: batch_size x 2 x D, 2 represents the ``pair'' information.
        batch_w_input = batch_w_input.to(device)
        w_later = w_later.to(device)
        strength = id_module(batch_w_input)
        strength = strength.cpu().tolist()
        # convert the indices to gene names
        gene1 = names.iloc[indices[:, 0].tolist()]['gene_ids'].tolist()
        gene2 = names.iloc[indices[:, 1].tolist()]['gene_ids'].tolist()     
        temp_data.append(pd.DataFrame({'gene1': gene1, 'gene2': gene2, 'avg_score': strength}))

    interaction_ranking = pd.concat(temp_data, ignore_index=True)
    interaction_ranking.sort_values('avg_score', ascending=False, inplace=True)
    
    # take the first 200000 pairs
    interaction_ranking = interaction_ranking.head(200000)
    return interaction_ranking


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_type', required=True)
    
    args = parser.parse_args()
    print(f"Running task for cell type: {args.cell_type}")
    model_path = f"output_models/MLP/{args.cell_type}/mlp_epoch_9.pth"
    model = torch.load(model_path, map_location='cpu')
    weights = []
    for k, v in model.items():
        if k.endswith('weight'):
            # we do not need bias
            # print(k)
            # print(v.shape)
            weights.append(v)
    w_input, w_later = preprocess_weights(weights)
    interaction_ranking = interpret_pairwise_interactions(w_input, w_later, 1024)
    interaction_ranking.to_csv(f'feature/score_maps/mlp/{args.cell_type}_interaction_ranking.csv', index=False)


if __name__ == "__main__":
    main()