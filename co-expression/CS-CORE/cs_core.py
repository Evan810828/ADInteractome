"""Rank gene pairs by the original ADInteractome CS-CORE baseline.

Select the top expressed genes using the split's normalized X, as before, and
run CS-CORE on raw UMI counts from the original h5ad. Cell identities, labels,
and gene order are checked before extracting counts. Rank High minus Not AD
*p-values* (CSCORE result 1), with both directed entries added, as in the
historical implementation.
"""
import argparse
from pathlib import Path

import anndata as ad
import h5py
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


def _source_positions(source, split):
    """Map split observations to source rows without loading source matrices."""
    obs, var = source['obs'], source['var']
    src_names = pd.Index(ad.io.read_elem(obs[obs.attrs['_index']]))
    gene_names = pd.Index(ad.io.read_elem(var[var.attrs['_index']]))
    if not src_names.is_unique or src_names.hasnans or not split.obs_names.is_unique or split.obs_names.hasnans:
        raise ValueError('Source and split cell IDs must be unique and nonmissing')
    if not gene_names.is_unique or not gene_names.equals(split.var_names):
        raise ValueError('Source and split gene names or order do not match')
    positions = src_names.get_indexer(split.obs_names)
    if (positions < 0).any():
        raise ValueError('Split contains cell IDs missing in source')
    for name in ('Overall AD neuropathological Change', 'Subclass'):
        if name not in split.obs or name not in obs:
            raise ValueError(f'Missing cell metadata: {name}')
        source_values = pd.Series(ad.io.read_elem(obs[name])).iloc[positions].reset_index(drop=True)
        split_values = split.obs[name].reset_index(drop=True)
        if source_values.isna().any() or split_values.isna().any() or not np.array_equal(
            source_values.astype(str).to_numpy(), split_values.astype(str).to_numpy()
        ):
            raise ValueError(f'Source and split {name} values do not match')
    return positions


def _extract_umis(source, positions, columns, max_span=1_000_000):
    """Read bounded contiguous CSR spans, then restore the requested row order.

    Compute sequencing depth before subsetting genes. The selected counts remain
    sparse; only one bounded source span is held in memory at a time.
    """
    group = source['layers/UMIs']
    if group.attrs.get('encoding-type') != 'csr_matrix':
        raise ValueError('UMIs layer must use CSR encoding')
    n_rows, n_cols = map(int, group.attrs['shape'])
    positions = np.asarray(positions, dtype=np.int64)
    columns = np.asarray(columns, dtype=np.int64)
    if max_span <= 0 or positions.ndim != 1 or not len(positions) or len(np.unique(positions)) != len(positions):
        raise ValueError('Positions must be a nonempty list of unique rows; max_span must be positive')
    if (positions < 0).any() or (positions >= n_rows).any() or (columns < 0).any() or (columns >= n_cols).any():
        raise ValueError('Source row or gene index out of bounds')
    ptr = np.asarray(group['indptr'][:], dtype=np.int64)
    if len(ptr) != n_rows + 1 or ptr[0] != 0 or (np.diff(ptr) < 0).any():
        raise ValueError('Invalid CSR indptr in UMI layer')
    order = np.argsort(positions)
    sorted_pos = positions[order]
    pieces, depths = [], []
    start = 0
    while start < len(sorted_pos):
        end = start + 1
        # A single unusually large row is allowed even if it exceeds max_span.
        while end < len(sorted_pos) and ptr[sorted_pos[end] + 1] - ptr[sorted_pos[start]] <= max_span:
            end += 1
        first, last = int(sorted_pos[start]), int(sorted_pos[end - 1])
        first_ptr, last_ptr = int(ptr[first]), int(ptr[last + 1])
        values = np.asarray(group['data'][first_ptr:last_ptr])
        indices = np.asarray(group['indices'][first_ptr:last_ptr])
        block = sparse.csr_matrix((values, indices, ptr[first:last + 2] - first_ptr), shape=(last-first+1, n_cols))
        selected = block[sorted_pos[start:end] - first]
        if not np.isfinite(selected.data).all() or (selected.data < 0).any() or not np.equal(selected.data, np.floor(selected.data)).all():
            raise ValueError('Selected UMI counts must be finite nonnegative integers')
        depth = np.asarray(selected.sum(axis=1)).ravel()
        if not np.isfinite(depth).all() or (depth <= 0).any():
            raise ValueError('Selected cells must have finite positive UMI depths')
        depths.append(depth)
        pieces.append(selected[:, columns].tocsr())
        start = end
    reverse = np.argsort(order)
    return sparse.vstack(pieces, format='csr')[reverse], np.concatenate(depths)[reverse]


def run(data_path, source_path, output, top_genes=5000, top_k=200000):
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
    print(f'Using raw UMI counts, {len(columns)} top genes selected on normalized X, p-value High minus Not AD', flush=True)
    results = {}
    with h5py.File(source_path, 'r') as source:
        positions = _source_positions(source, data)
        if tuple(source['layers/UMIs'].attrs['shape']) != (len(source['obs'][source['obs'].attrs['_index']]), data.n_vars):
            raise ValueError('UMI shape does not match source cell and gene dimensions')
        for name in ('High', 'Not AD'):
            mask = labels.eq(name).to_numpy()
            if mask.sum() < 2:
                raise ValueError(f'{name}: fewer than two cells')
            print(f'{name}: extracting UMI counts for {mask.sum()} cells', flush=True)
            selected, depth = _extract_umis(source, positions[mask], columns)
            estimate, p_value, statistic = CSCORE_IRLS(selected, depth, post_process=True)
            results[name] = p_value
            del estimate, statistic, selected, depth, p_value
    score = results['High'] - results['Not AD']
    del results
    ranking = rank_pairs(score, data.var_names.to_numpy()[columns], top_k)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output, index=False)
    print(f'Wrote {len(ranking)} ranked pairs to {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Cell type training split h5ad')
    parser.add_argument('--source', required=True, help='Original h5ad with raw counts in layers/UMIs')
    parser.add_argument('--output', required=True, help='Output gene-pair ranking CSV')
    parser.add_argument('--top_genes', type=int, default=5000)
    parser.add_argument('--top_k', type=int, default=200000)
    args = parser.parse_args()
    run(args.data, args.source, args.output, args.top_genes, args.top_k)


if __name__ == '__main__':
    main()
