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
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import hdf5plugin

from transformer.dataset import VariableLengthSequenceDataset
from utils import setup_seed, load_checkpoint


def load_data(data_path):
    data_cell = sc.read_h5ad(data_path)

    filtered_data = data_cell[data_cell.obs['Overall AD neuropathological Change'].isin(['Not AD', 'High'])]
    X_filtered = filtered_data.X
    y_filtered = filtered_data.obs['Overall AD neuropathological Change']
    label_mapping = {'Not AD': 0, 'High': 1}
    y_encoded = y_filtered.map(label_mapping)
    
    dataset = VariableLengthSequenceDataset(X_filtered, y_encoded)
    
    return dataset


def compute_selfattention(encoder, x, layer, d_model, num_heads, logit_scale):
    """Use the QKV layout and optional logit multiplier of the reference script."""
    attention = encoder.layers[layer].self_attn
    h = F.linear(x, attention.in_proj_weight, bias=attention.in_proj_bias)
    qkv = h.reshape(x.shape[0], x.shape[1], num_heads, 3 * d_model // num_heads)
    q, k, _ = qkv.permute(0, 2, 1, 3).chunk(3, dim=-1)
    return F.softmax((q @ k.transpose(-2, -1)) * logit_scale / math.sqrt(q.size(-1)), dim=-1)


def compute_rank(model, dataset, config, args):
    """Restore the aggregation in temp.py, including its padding and smoothing rules."""
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    model = model.to(args.device)
    transformer_encoder = model.encoder
    score_map = torch.zeros((config['num_genes'], config['num_genes'])).to(args.device)
    frequency_map = torch.zeros((config['num_genes'], config['num_genes'])).to(args.device)
    
    for seq, vals, _ in tqdm(dataloader, desc="Computing rank"):
        
        seq = seq.to(args.device) - 1
        vals = vals.to(args.device)
        
        E_attn = torch.zeros((seq.shape[1], seq.shape[1])).to(args.device)
        
        with torch.no_grad():
            embedded_X = model.emb(seq) * vals.unsqueeze(2)
            num_layers = transformer_encoder.num_layers
            d_model = transformer_encoder.layers[0].self_attn.embed_dim
            num_heads = transformer_encoder.layers[0].self_attn.num_heads
            norm_first = transformer_encoder.layers[0].norm_first
            for i in range(num_layers):
                h = embedded_X.clone()
                if norm_first:
                    h = transformer_encoder.layers[i].norm1(h)
                attn_probs = compute_selfattention(
                    transformer_encoder, h, i, d_model, num_heads, args.logit_scale
                )
                E_attn += attn_probs.sum(dim=1)[0]

            E_attn /= num_layers * num_heads

            # Match the reference scatter operation: each distinct source row
            # receives the same column-summed attention; repeated indices in
            # the padded sequence contribute repeatedly to destination columns.
            index_map = torch.zeros((config['num_genes'], seq.shape[1]), device=args.device)
            index_map[seq[0]] = seq[0].float()
            value_map = torch.zeros_like(index_map)
            value_map[seq[0], :] = E_attn.sum(0).float()
            score_map.scatter_add_(1, index_map.long(), value_map)

            value_map.zero_()
            value_map[seq[0], :] = 1.0
            frequency_map.scatter_add_(1, index_map.long(), value_map)
            frequency_map += 1e-5

    return (score_map / frequency_map).cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="gene-transformer")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the JSON file containing the hyperparameters")
    parser.add_argument("--model_path", type=str, default="output_models/global_training/restful-dew-11/epoch_2_batch_3473.pth", help="Path to the model file")
    parser.add_argument("--data", type=str, default="data/split_train/cell_type_Lamp5_Lhx6.h5ad", help="Path to the data file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    parser.add_argument("--output", type=str, default="feature/score_maps/adjust/score_map_cell_type_Lamp5_Lhx6_adjusted.pkl", help="Path to the output file")
    parser.add_argument("--logit_scale", type=float, choices=[1.0, 1000.0], default=1000.0,
                        help="Attention-logit multiplier: 1000 follows temp.py; 1 follows utils.py")
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
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(score_map, f)
