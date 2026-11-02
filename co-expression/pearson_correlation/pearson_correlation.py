"""Rank signed differences in gene-gene Pearson correlation between AD groups.

For each cell type, compute corr(High) - corr(Not AD), as in the original
script, and retain the highest-scoring unordered gene pairs. Matrix blocks
avoid materializing the full 36,601 by 36,601 correlation matrix.
"""
import argparse
from pathlib import Path

import anndata as ad
import hdf5plugin  # noqa: F401: register the compressed h5ad filter
import numpy as np
import pandas as pd
import torch


def normalize_group(matrix, device):
    """Return centered, column-normalized values; zero-variance genes are zero."""
    if matrix.shape[0] < 2:
        raise ValueError("Each diagnosis group needs at least two cells")
    dense = matrix.toarray() if hasattr(matrix, 'toarray') else np.asarray(matrix)
    tensor = torch.as_tensor(dense, dtype=torch.float64, device=device)
    del dense
    tensor -= tensor.mean(dim=0, keepdim=True)
    norms = torch.linalg.vector_norm(tensor, dim=0)
    norms[norms == 0] = 1.0
    tensor /= norms
    return tensor


def pearson_top_pairs(high, not_ad, names, top_k=200000, block_size=128, device='cuda:0'):
    """Return highest signed High-minus-Not-AD Pearson differences (i < j)."""
    names = np.asarray(names, dtype=str)
    n = len(names)
    if high.shape[1] != n or not_ad.shape[1] != n or n < 2:
        raise ValueError('Input gene dimensions do not match gene names')
    if len(set(names)) != n or pd.isna(names).any():
        raise ValueError('Gene names must be unique and nonmissing')
    if top_k <= 0 or block_size <= 0:
        raise ValueError('top_k and block_size must be positive')
    top_k = min(top_k, n * (n - 1) // 2)
    high_norm = normalize_group(high, device)
    not_ad_norm = normalize_group(not_ad, device)
    best_scores = np.empty(0, dtype=np.float64)
    best_pairs = np.empty(0, dtype=np.int64)
    with torch.no_grad():
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            # Match np.corrcoef's per-group clipping and zero-variance behavior.
            a = torch.matmul(high_norm[:, start:end].T, high_norm).clamp_(-1, 1)
            b = torch.matmul(not_ad_norm[:, start:end].T, not_ad_norm).clamp_(-1, 1)
            diff = (a - b).cpu().numpy()
            i, j = np.nonzero(np.arange(start, end)[:, None] < np.arange(n)[None, :])
            values = np.concatenate((best_scores, diff[i, j]))
            pairs = np.concatenate((best_pairs, (i + start) * n + j))
            if len(values) > top_k:
                cutoff = np.partition(values, len(values) - top_k)[len(values) - top_k]
                above = np.flatnonzero(values > cutoff)
                tied = np.flatnonzero(values == cutoff)
                remaining = top_k - len(above)
                if remaining < len(tied):
                    tied = tied[np.argpartition(pairs[tied], remaining - 1)[:remaining]] if remaining else tied[:0]
                selected = np.concatenate((above, tied))
                values, pairs = values[selected], pairs[selected]
            best_scores, best_pairs = values, pairs
            print(f'Processed genes {end}/{n}', flush=True)
    order = np.lexsort((best_pairs, -best_scores))
    pairs = best_pairs[order]
    return pd.DataFrame({'gene1': names[pairs // n], 'gene2': names[pairs % n],
                         'avg_score': best_scores[order]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Cell-type training h5ad')
    parser.add_argument('--output', type=Path, required=True, help='GSEA ranking CSV')
    parser.add_argument('--top_k', type=int, default=200000)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    data = ad.read_h5ad(args.data)
    group = data.obs['Overall AD neuropathological Change']
    high = data.X[group.eq('High').to_numpy()]
    not_ad = data.X[group.eq('Not AD').to_numpy()]
    print(f'High: {high.shape[0]}; Not AD: {not_ad.shape[0]}; genes: {data.n_vars}', flush=True)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    ranking = pearson_top_pairs(high, not_ad, data.var_names, args.top_k, args.block_size, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.output, index=False)
    print(f'Wrote {len(ranking)} pairs to {args.output}', flush=True)


if __name__ == '__main__':
    main()
