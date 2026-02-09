"""Extract gene-gene interaction scores using sampling-based attention averaging.

Supports both diversified (zero-ICWS) and uniform sampling strategies to
compute attention-based gene interaction score maps from a trained CelluFormer.

Example usage:
    python feature/sampling_attention_mean.py --config config.json \\
        --model_path <checkpoint>.pth --data <data>.h5ad \\
        --sample_mode diversified --sample_size 200
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pickle
import argparse
import json
import scanpy as sc
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Sampler
from tqdm import tqdm
import hdf5plugin

from transformer.celluformer import CelluFormer
from transformer.dataset import VariableLengthSequenceDataset
from transformer.weighted_sampling_dataset import zero_ICWS
from utils import setup_seed, load_checkpoint, compute_selfattention


class UniformSampler(Sampler):
    def __init__(self, data_source, sample_size):
        self.data_source = data_source
        self.sample_size = sample_size

    def __iter__(self):
        return iter(torch.randperm(len(self.data_source)).tolist()[:self.sample_size])
    
    def __len__(self):
        return self.sample_size


def load_data(data_path, args):
    data_cell = sc.read_h5ad(data_path)

    filtered_data = data_cell[data_cell.obs['Overall AD neuropathological Change'].isin(['Not AD', 'High'])]
    X_filtered = filtered_data.X
    y_filtered = filtered_data.obs['Overall AD neuropathological Change']
    label_mapping = {'Not AD': 0, 'High': 1}
    y_encoded = y_filtered.map(label_mapping)
    
    if args.sample_mode == 'diversified':
        # weighted sampling
        seeds = np.arange(500)
        dataset = zero_ICWS(X_filtered, y_encoded, 36601, 500, seeds)
    elif args.sample_mode == 'uniform':
        # uniform sampling
        dataset = VariableLengthSequenceDataset(X_filtered, y_encoded)
    
    return dataset


def compute_rank(model, dataset, config, args):
    if args.sample_mode == 'diversified':
        # weighted sampling
        dataset_weight_tensor = torch.tensor(dataset.weighted_array).type(torch.DoubleTensor).to(args.device)
        sampler = WeightedRandomSampler(dataset_weight_tensor, int(args.sample_size * 0.01 * len(dataset)))
        dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)
    elif args.sample_mode == 'uniform':
        # uniform sampling
        sampler = UniformSampler(dataset, int(args.sample_size * 0.01 * len(dataset)))
        dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)
    
    model = model.to(args.device)
    transformer_encoder = model.encoder
    score_map = torch.zeros((config['num_genes'], config['num_genes']))
    frequency_map = torch.zeros(config['num_genes'], config['num_genes'])
    
    iterator = iter(dataloader)
    for i in tqdm(range(len(dataloader)), desc="Computing rank"):
        if args.sample_mode == 'diversified':
            seq, vals, _, sampling_prob = next(iterator)
        elif args.sample_mode == 'uniform':
            seq, vals, _ = next(iterator)
            sampling_prob = torch.ones(1)
        
        seq = seq.to(args.device) - 1
        vals = vals.to(args.device)
        
        E_attn = torch.zeros((seq.shape[1], seq.shape[1]))
        
        embedded_X = model.emb(seq)
        embedded_X = embedded_X * vals.unsqueeze(2)
        
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
                attn_probs = compute_selfattention(transformer_encoder, h, i, d_model, num_heads)
                
                # compute mean
                attn_probs = attn_probs.cpu()
                E_attn = E_attn + attn_probs.sum(dim=1)
        
        E_attn = E_attn / (num_layers * attn_probs.shape[1])
        
        # compute score
        index_map = torch.zeros((36601, seq.shape[1]))
        index_map[seq[0]] = seq[0].float().cpu()
        
        res = E_attn.sum(0)
        value_map = torch.zeros((36601, seq.shape[1]))
        value_map[seq[0], :] = res.cpu()
        value_map = value_map * sampling_prob
            
        score_map.scatter_add_(1, index_map.long(), value_map.to(torch.float32))
        
        # compute frequency
        frequency_block = torch.ones(seq[0].shape[0], seq[0].shape[0])
        value_map = torch.zeros((36601, seq.shape[1]))
        value_map[seq[0], :] = frequency_block
        value_map = value_map * sampling_prob
            
        frequency_map.scatter_add_(1, index_map.long(), value_map.to(torch.float32))
        frequency_map += 1e-5
                
    return score_map / frequency_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="gene-transformer")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the JSON file containing the hyperparameters")
    parser.add_argument("--model_path", type=str, default="output_models/global_training/restful-dew-11/epoch_2_batch_3473.pth", help="Path to the model file")
    parser.add_argument("--data", type=str, default="data/split_train/cell_type_L6_CT.h5ad", help="Path to the data file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    parser.add_argument("--sample_size", type=int, default=200, help="Number of samples to draw")
    parser.add_argument("--sample_mode", type=str, default="diversified", help="Choice of the sampling method")
    args = parser.parse_args()
    
    config = json.load(open(args.config))
    
    # Loading the model
    print("Loading model...")
    model = load_checkpoint(config, args.model_path, map_location=args.device)
    # Loading the data
    print("Loading data...")
    dataset = load_data(args.data, args)
    
    for i in range(1, 6):
        # Computing the rank
        score_map = compute_rank(model, dataset, config, args)
        
        # save the score map
        cell_type = args.data.split('/')[-1].split('.')[0].split('_')[-1]
        with open(f"feature/score_maps/n_sample=500/score_map_{cell_type}_neuron_{args.sample_mode}_{args.sample_size}_{i}.pkl", 'wb') as f:
            pickle.dump(score_map, f)
