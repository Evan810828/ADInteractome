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


def compute_selfattention(encoder, x, layer, d_model, num_heads, logit_scale,
                          query_block_size):
    """Yield query blocks using the reference QKV layout and logit multiplier.

    Each query is normalized over the complete key sequence, exactly as in the
    full attention matrix. Only the query dimension is partitioned.
    """
    if query_block_size <= 0:
        raise ValueError("query_block_size must be positive")
    attention = encoder.layers[layer].self_attn
    h = F.linear(x, attention.in_proj_weight, bias=attention.in_proj_bias)
    qkv = h.reshape(x.shape[0], x.shape[1], num_heads, 3 * d_model // num_heads)
    q, k, _ = qkv.permute(0, 2, 1, 3).chunk(3, dim=-1)
    kt = k.transpose(-2, -1)
    for start in range(0, x.shape[1], query_block_size):
        stop = min(start + query_block_size, x.shape[1])
        logits = (q[:, :, start:stop] @ kt) * logit_scale / math.sqrt(q.size(-1))
        yield start, stop, F.softmax(logits, dim=-1)


def compute_rank(model, dataset, config, args):
    """Restore the aggregation in temp.py, including its padding and smoothing rules."""
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    model = model.to(args.device)
    transformer_encoder = model.encoder
    score_map = torch.zeros((config['num_genes'], config['num_genes'])).to(args.device)
    frequency_map = torch.zeros((config['num_genes'], config['num_genes'])).to(args.device)
    
    for seq, vals, _ in tqdm(dataloader, desc="Computing rank"):
        
        # Keep the dataset's 1-based tokens for the trained embedding table.
        seq = seq.to(args.device)
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
                for start, stop, attn_probs in compute_selfattention(
                    transformer_encoder, h, i, d_model, num_heads,
                    args.logit_scale, args.query_block_size
                ):
                    E_attn[start:stop] += attn_probs.sum(dim=1)[0]

            E_attn /= num_layers * num_heads

            # Only score-map indices are 0-based; embedding tokens stay 1-based.
            gene_indices = seq[0] - 1
            # Every distinct source gene receives the same column sums. Fold
            # repeated padding destinations before updating those rows, avoiding
            # three dense (num_genes, sequence_length) scatter buffers.
            genes, inverse, counts = torch.unique(
                gene_indices, return_inverse=True, return_counts=True
            )
            column_scores = torch.zeros(genes.numel(), device=args.device)
            column_scores.scatter_add_(0, inverse, E_attn.sum(0).float())
            rows, columns = genes[:, None], genes[None, :]
            score_map[rows, columns] += column_scores[None, :]
            frequency_map[rows, columns] += counts[None, :].float()
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
    parser.add_argument("--query_block_size", type=int, default=256,
                        help="Queries per attention block; softmax still covers every key")
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
