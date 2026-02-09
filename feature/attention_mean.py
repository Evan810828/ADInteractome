"""Extract gene-gene interaction scores from transformer attention maps.

Loads a trained CelluFormer checkpoint and computes averaged self-attention
maps across the dataset to produce gene-gene interaction score matrices.

Example usage:
    python feature/attention_mean.py --config config.json \\
        --model_path <checkpoint>.pth --data <data>.h5ad \\
        --output feature/score_maps/score_map.pkl
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pickle
import argparse
import json
import scanpy as sc
import gc
from torch.utils.data import DataLoader
from tqdm import tqdm
import hdf5plugin

from transformer.celluformer import CelluFormer
from transformer.dataset import VariableLengthSequenceDataset
from transformer.SeqLength_sampling_dataset import SeqLengthScore
from utils import setup_seed, load_checkpoint, compute_selfattention


def load_data(data_path):
    data_cell = sc.read_h5ad(data_path)

    filtered_data = data_cell[data_cell.obs['Overall AD neuropathological Change'].isin(['Not AD', 'High'])]
    X_filtered = filtered_data.X
    y_filtered = filtered_data.obs['Overall AD neuropathological Change']
    label_mapping = {'Not AD': 0, 'High': 1}
    y_encoded = y_filtered.map(label_mapping)
    
    # # weighted sampling
    # dataset = SeqLengthScore(X_filtered, y_encoded, 1e3)
    
    # normal
    dataset = VariableLengthSequenceDataset(X_filtered, y_encoded)
    
    return dataset


def compute_rank(model, dataset, config, args):
    """Compute gene-gene interaction score maps by averaging attention across samples."""
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    model = model.to(args.device)
    transformer_encoder = model.encoder
    score_map = torch.zeros((config['num_genes'], config['num_genes'])).to(args.device)
    frequency_map = torch.zeros((config['num_genes'], config['num_genes'])).to(args.device)
    
    iterator = iter(dataloader)
    for i in tqdm(range(len(dataloader)), desc="Computing rank"):
        seq, vals, _ = next(iterator)
        
        seq = seq.to(args.device) - 1
        vals = vals.to(args.device)
        
        E_attn = torch.zeros((seq.shape[1], seq.shape[1])).to(args.device)
        
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
                
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="gene-transformer")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the JSON file containing the hyperparameters")
    parser.add_argument("--model_path", type=str, default="output_models/global_training/restful-dew-11/epoch_2_batch_3473.pth", help="Path to the model file")
    parser.add_argument("--data", type=str, default="data/split_train/cell_type_Lamp5_Lhx6.h5ad", help="Path to the data file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    parser.add_argument("--output", type=str, default="feature/score_maps/adjust/score_map_cell_type_Lamp5_Lhx6_adjusted.pkl", help="Path to the output file")
    args = parser.parse_args()
    
    setup_seed(42)
    
    config = json.load(open(args.config))
    
    # Loading the model
    print("Loading model...")
    model = load_checkpoint(config, args.model_path)
    # Loading the data
    print("Loading data...")
    dataset = load_data(args.data)
    
    # Computing the rank
    score_map = compute_rank(model, dataset, config, args)
    
    # save the score map
    with open(args.output, 'wb') as f:
        pickle.dump(score_map, f)
