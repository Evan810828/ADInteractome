import torch
import pickle
import torch.nn as nn
import pandas as pd
import sys
import argparse
import json
import scanpy as sc
import math
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import hdf5plugin
from torch.utils.data import Dataset
from collections import OrderedDict
import os
import pdb

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pretrainmodels.mae_autobin_finetuned import MaeAutobin_finetuned
    
class scFoundationDataset(Dataset):
    def __init__(self, X, y, main_gene_indices): # X is a sparse matrix
        self.seq = []
        self.labels = torch.tensor(y)
        self.main_genes_indices = main_gene_indices

        print("Initializing dataset...")
        for i in tqdm(range(X.shape[0]), desc="Processing data"):
            x = X[i].toarray()
            x_tensor = torch.from_numpy(x).float()[0]
            
            self.seq.append(x_tensor[main_gene_indices])

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, index):
        sequence = self.seq[index]
        label = self.labels[index]
        return sequence, label
    
def load_data(data_path, batch_size=1):
    data_cell = sc.read_h5ad(data_path)

    filtered_data = data_cell[data_cell.obs['Overall AD neuropathological Change'].isin(['Not AD', 'High'])]
    X_filtered = filtered_data.X
    y_filtered = filtered_data.obs['Overall AD neuropathological Change']
    label_mapping = {'Not AD': 0, 'High': 1}
    y_encoded = y_filtered.map(label_mapping)
    
    gene_list_df = pd.read_csv('foundation_models/scFoundation/OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
    gene_list = list(gene_list_df['gene_name'])
    global main_gene_indices
    main_gene_indices = [i for i in filtered_data.var.index.get_indexer(gene_list) if i != -1]
    
    
    dataset = scFoundationDataset(X_filtered, y_encoded, main_gene_indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def compute_selfattention(transformer_encoder, x, i_layer, d_model, num_heads):
    h = F.linear(x, transformer_encoder.layers[i_layer].self_attn.in_proj_weight, bias=transformer_encoder.layers[i_layer].self_attn.in_proj_bias)
    qkv = h.reshape(x.shape[0], x.shape[1], num_heads, 3 * d_model//num_heads)
    qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
    q, k, v = qkv.chunk(3, dim=-1) # [Batch, Head, SeqLen, d_head=d_model//num_heads]
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) # [Batch, Head, SeqLen, SeqLen]
    d_k = q.size()[-1]
    attn_probs = attn_logits / math.sqrt(d_k)
    # setting masked logits to -inf before softmax
    attn_probs = F.softmax(attn_probs, dim=-1)
    return attn_probs


def compute_rank(model, dataloader, config, args):
    model = model.to(args.device)
    transformer_encoder = model.encoder
    score_map = torch.zeros((config['num_genes'], config['num_genes'])).to(args.device)
    frequency_map = torch.zeros((config['num_genes'], config['num_genes'])).to(args.device)
    
    global main_gene_indices
    main_gene_indices = torch.tensor(main_gene_indices).to(args.device)
    
    iterator = iter(dataloader)
    for i in tqdm(range(len(dataloader)), desc="Computing rank"):
        seq, _ = next(iterator)
        seq = seq.to(args.device)
        idx = torch.nonzero(seq[0]).squeeze().to(args.device)
        
        E_attn = torch.zeros((idx.shape[0], idx.shape[0]))
        
        embedded_X, _ = model.emb(seq)
        
        # extract self-attention maps
        num_layers = transformer_encoder.num_layers
        d_model = transformer_encoder.layers[0].self_attn.embed_dim
        num_heads = transformer_encoder.layers[0].self_attn.num_heads
        norm_first = transformer_encoder.layers[0].norm_first
        with torch.no_grad():
            for i in range(num_layers):
                # compute attention of layer i
                h = embedded_X.clone()
                if norm_first:
                    h = transformer_encoder.layers[i].norm1(h)
                # attention_maps.append(attn) # of shape [batch_size,seq_len,seq_len]
                attn_probs = compute_selfattention(transformer_encoder, h, i, d_model, num_heads)
                
                # compute mean
                # attn_probs = attn_probs.cpu()
                E_attn = attn_probs.sum(dim=1)
        
        E_attn = E_attn / (num_layers * attn_probs.shape[1])
        
        # compute score
        index_map = torch.zeros((36601, idx.shape[0])).to(args.device)
        index_map[main_gene_indices[idx]] = main_gene_indices[idx].float()
        
        res = E_attn.sum(0)
        value_map = torch.zeros((36601, idx.shape[0])).to(args.device)
        value_map[main_gene_indices[idx], :] = res
            
        score_map.scatter_add_(1, index_map.to(torch.int64), value_map)
        
        # compute frequency
        frequency_block = torch.ones(idx.shape[0], idx.shape[0]).to(args.device)
        value_map = torch.zeros((36601, idx.shape[0])).to(args.device)
        value_map[main_gene_indices[idx], :] = frequency_block
            
        frequency_map.scatter_add_(1, index_map.long(), value_map.to(torch.float32))
        frequency_map += 1e-5
                
    result = (score_map / frequency_map).cpu().numpy()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="gene-transformer")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the JSON file containing the hyperparameters")
    parser.add_argument("--model_path", type=str, default="finetuned_ckpts/finetuned_model_epoch_3.pth", help="Path to the model file")
    parser.add_argument("--data", type=str, default="data/split_train/cell_type_L5_ET.h5ad", help="Path to the data file")
    parser.add_argument("--device", type=str, default="cuda:2", help="Device to run the model on")
    parser.add_argument("--output", type=str, default="scFoundation/score_maps/score_map_Chandelier_scFoundation.pkl", help="Path to the output file")
    args = parser.parse_args()
    
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    config = json.load(open(args.config))
    
    # Loading the model
    print("Loading model...")
    
    model = MaeAutobin_finetuned()
    state_dict = torch.load(args.model_path)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    # Loading the data
    print("Loading data...")
    dataloader = load_data(args.data)
    
    # Computing the rank
    score_map = compute_rank(model, dataloader, config, args)
    
    # save the score map
    with open(args.output, 'wb') as f:
        pickle.dump(score_map, f)
    