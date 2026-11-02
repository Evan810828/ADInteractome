"""Rank gene pairs by the original ADInteractome CS-CORE baseline.

This reproduces the repository's historical implementation: the split h5ad X
matrix is passed to CS-CORE, the top expressed genes are selected across both
labels, and pairs are scored by High minus Not AD *p-values* (CSCORE result 1),
with both directed entries added. CS-CORE documents raw UMI counts as its input;
the 2026 X is normalized expression, so these scores must not be interpreted as
CS-CORE statistics computed from raw counts.
"""
import argparse
from pathlib import Path

import anndata as ad
import hdf5plugin  # Registers zstd used by split h5ad files.
import numpy as np
import pandas as pd
from scipy import sparse

# CSCORE 1.0.0 uses np.Inf, which NumPy 2 removed. This is its old value.
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
from CSCORE import CSCORE_IRLS


def rank_pairs(score, gene_names, top_k=200000, row_block=64):
    """Rank finite unordered pairs by original directed-score sum.

    Excludes diagonal, lower-triangle placeholders and non-finite p-value
    differences, which cannot form a valid gene-pair ranking or GSEA input.
    Ties are ordered deterministically by input indices.
    """
    n = len(gene_names)
    if score.shape != (n, n) or top_k <= 0 or row_block <= 0:
        raise ValueError('Invalid score shape, top_k or row_block')
    best_s = np.empty(0, dtype=np.float64)
    best_i = np.empty(0, dtype=np.int64)
    best_j = np.empty(0, dtype=np.int64)
    invalid = 0
    for start in range(0, n, row_block):
        end = min(n, start + row_block)
        ii, jj = np.nonzero(np.arange(start, end)[:, None] < np.arange(n)[None, :])
        ii += start
        vals = score[ii, jj] + score[jj, ii]
        finite = np.isfinite(vals)
        invalid += int((~finite).sum())
        vals, ii, jj = vals[finite], ii[finite], jj[finite]
        vals = np.concatenate((best_s, vals))
        ii = np.concatenate((best_i, ii))
        jj = np.concatenate((best_j, jj))
        if len(vals) > top_k:
            threshold = np.partition(vals, len(vals) - top_k)[len(vals) - top_k]
            above = np.flatnonzero(vals > threshold)
            tied = np.flatnonzero(vals == threshold)
            order = np.lexsort((jj[tied], ii[tied]))
            chosen = np.concatenate((above, tied[order[:top_k - len(above)]]))
            vals, ii, jj = vals[chosen], ii[chosen], jj[chosen]
        best_s, best_i, best_j = vals, ii, jj
    if len(best_s) < top_k:
        raise ValueError(f'Only {len(best_s)} finite pairs available; requested {top_k}; excluded {invalid}')
    order = np.lexsort((best_j, best_i, -best_s))
    names = np.asarray(gene_names, dtype=str)
    print(f'Excluded {invalid} non-finite unordered pairs', flush=True)
    return pd.DataFrame({'gene1': names[best_i[order]], 'gene2': names[best_j[order]],
                         'avg_score': best_s[order]})


def run(data_path, output, top_genes=5000, top_k=200000):
    data = ad.read_h5ad(data_path)
    if not data.var_names.is_unique or data.var_names.isna().any():
        raise ValueError('Gene symbols must be unique and nonmissing')
    labels = data.obs['Overall AD neuropathological Change']
    keep = labels.isin(['High', 'Not AD']).to_numpy()
    if not keep.all():
        raise ValueError('Expected a split containing only High/Not AD cells')
    x = data.X.tocsr() if sparse.issparse(data.X) else np.asarray(data.X)
    sums = np.asarray(x.sum(axis=0)).ravel()
    # pandas reproduces the original script's top-gene ordering.
    columns = pd.DataFrame({'gene': data.var_names, 'mean_expression': sums}).sort_values(
        by='mean_expression', ascending=False).head(top_genes).index.to_numpy(dtype=int)
    if len(columns) * (len(columns) - 1) // 2 < top_k:
        raise ValueError('Not enough genes for the requested number of pairs')
    print(f'Using normalized h5ad X, {len(columns)} top genes, p-value High minus Not AD', flush=True)
    results = {}
    for name in ('High', 'Not AD'):
        mask = labels.eq(name).to_numpy()
        if mask.sum() < 2:
            raise ValueError(f'{name}: fewer than two cells')
        subset = x[mask]
        depth = np.asarray(subset.sum(axis=1)).ravel()
        if not np.isfinite(depth).all() or (depth <= 0).any():
            raise ValueError(f'{name}: invalid sequencing depth from X')
        print(f'{name}: {mask.sum()} cells', flush=True)
        estimate, p_value, statistic = CSCORE_IRLS(subset[:, columns], depth, post_process=True)
        # Preserve the original selection of result[1]; estimates are not ranked.
        results[name] = p_value
        del estimate, statistic, subset, p_value
    score = results['High'] - results['Not AD']
    del results
    ranking = rank_pairs(score, data.var_names.to_numpy()[columns], top_k)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output, index=False)
    print(f'Wrote {len(ranking)} ranked pairs to {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Cell type training split h5ad')
    parser.add_argument('--output', required=True, help='Output gene-pair ranking CSV')
    parser.add_argument('--top_genes', type=int, default=5000)
    parser.add_argument('--top_k', type=int, default=200000)
    args = parser.parse_args()
    run(args.data, args.output, args.top_genes, args.top_k)


if __name__ == '__main__':
    main()
