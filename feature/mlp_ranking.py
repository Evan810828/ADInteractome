"""Rank gene pairs by the trained MLP's weight-based interaction score."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def preprocess_weights(weights):
    w_later = torch.abs(weights[-1])
    w_input = torch.abs(weights[0])
    for i in range(len(weights) - 2, 0, -1):
        w_later = torch.matmul(w_later, torch.abs(weights[i]))
    return w_input, w_later


def top_indices(scores, pairs, k):
    """Keep highest scores, breaking ties by ascending gene-index pair."""
    if len(scores) <= k:
        return np.arange(len(scores))
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    above = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)
    needed = k - len(above)
    if needed < len(tied):
        tied = tied[np.argpartition(pairs[tied], needed - 1)[:needed]]
    return np.concatenate((above, tied))


def pair_batches(n_genes, batch_size):
    """Enumerate i < j in input gene order without materializing all pairs."""
    first, second, count = [], [], 0
    for i in range(n_genes - 1):
        start = i + 1
        while start < n_genes:
            take = min(batch_size - count, n_genes - start)
            first.append(np.full(take, i, dtype=np.int64))
            second.append(np.arange(start, start + take, dtype=np.int64))
            count += take
            start += take
            if count == batch_size:
                yield np.concatenate(first), np.concatenate(second)
                first, second, count = [], [], 0
    if count:
        yield np.concatenate(first), np.concatenate(second)


def interpret_pairwise_interactions(w_input, w_later, names, batch_size=8192,
                                    top_k=200000, device='cuda:0', merge_size=1000000):
    # Preserve the original half-precision min(first-layer weights) and
    # matrix multiplication with class-1 downstream absolute weights.
    w_input = w_input.T.half().to(device)
    w_later = w_later.T.half().to(device)
    n_genes = w_input.shape[0]
    if len(names) != n_genes or w_later.shape[1] < 2:
        raise ValueError('Gene count or output class count does not match checkpoint')
    if batch_size < 1 or top_k < 1 or merge_size < 1:
        raise ValueError('batch_size, top_k, and merge_size must be positive')
    kept_scores = np.empty(0, dtype=np.float16)
    kept_pairs = np.empty(0, dtype=np.int64)
    pending_scores, pending_pairs, pending_count = [], [], 0

    def merge():
        nonlocal kept_scores, kept_pairs, pending_scores, pending_pairs, pending_count
        scores = np.concatenate([kept_scores, *pending_scores])
        pairs = np.concatenate([kept_pairs, *pending_pairs])
        if not np.isfinite(scores).all():
            raise ValueError('Interaction scores contain non-finite values')
        chosen = top_indices(scores, pairs, top_k)
        kept_scores, kept_pairs = scores[chosen], pairs[chosen]
        pending_scores, pending_pairs, pending_count = [], [], 0

    total = n_genes * (n_genes - 1) // 2
    with torch.no_grad():
        for i, j in tqdm(pair_batches(n_genes, batch_size), total=(total + batch_size - 1) // batch_size):
            ti = torch.from_numpy(i).to(device)
            tj = torch.from_numpy(j).to(device)
            pair_min = torch.minimum(w_input[ti], w_input[tj])
            scores = torch.matmul(pair_min, w_later)[:, 1].cpu().numpy()
            pending_scores.append(scores)
            pending_pairs.append(i * n_genes + j)
            pending_count += len(i)
            if pending_count >= merge_size:
                merge()
    if pending_count:
        merge()
    order = np.lexsort((kept_pairs, -kept_scores.astype(np.float32)))
    pairs = kept_pairs[order]
    symbols = np.asarray(names, dtype=object)
    return pd.DataFrame({'gene1': symbols[pairs // n_genes],
                         'gene2': symbols[pairs % n_genes],
                         'avg_score': kept_scores[order].astype(np.float32)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True, help='h5ad used for MLP training; var index is gene symbols')
    parser.add_argument('--model_path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--top_k', type=int, default=200000)
    args = parser.parse_args()

    import hdf5plugin  # Register compression filters before opening the h5ad.
    import scanpy as sc
    data = sc.read_h5ad(args.data, backed='r')
    try:
        names = data.var_names.to_numpy().copy()
    finally:
        data.file.close()
    if pd.Index(names).has_duplicates or pd.isna(names).any():
        raise ValueError('Gene symbols must be unique and nonmissing')
    model = torch.load(args.model_path, map_location='cpu', weights_only=True)
    weights = [v for k, v in model.items() if k.endswith('weight')]
    if len(weights) < 2:
        raise ValueError('Expected at least two weight matrices in checkpoint')
    w_input, w_later = preprocess_weights(weights)
    ranking = interpret_pairwise_interactions(w_input, w_later, names,
                                               args.batch_size, args.top_k, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
