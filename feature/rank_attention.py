"""Convert CelluFormer score maps into ranked, unordered gene pairs.

The reference uses a global z-score, adds the two directed scores, and
keeps the highest 200,000 pairs. Processing rows in blocks avoids allocating
an additional 36,601-by-36,601 index tensor.
"""
import argparse
import os
import pickle
from pathlib import Path

import anndata as ad
import hdf5plugin  # noqa: F401 -- register h5ad compression filters
import numpy as np
import pandas as pd
import torch


def score_map_to_rank(score_map, names, top_k=200000, row_block=64):
    score = np.asarray(score_map).squeeze()
    n = len(names)
    if score.shape != (n, n):
        raise ValueError(f"Score map {score.shape} does not match {n} gene names")
    if n < 2 or top_k < 1 or row_block < 1:
        raise ValueError("At least two genes, positive top_k and row_block are required")
    if not np.isfinite(score).all():
        raise ValueError("Score map contains non-finite values")
    # Preserve the reference's treatment of gene zero as padding.
    score = np.array(score, dtype=np.float32, copy=True)
    score[0, :] = 0
    score[:, 0] = 0
    tensor = torch.from_numpy(score)
    mean = torch.mean(tensor).item()
    std = torch.std(tensor).item()
    if not np.isfinite(std) or std == 0:
        raise ValueError("Score map has zero or non-finite standard deviation")
    top_k = min(top_k, n * (n - 1) // 2)
    best_scores = np.empty(0, dtype=np.float32)
    best_i = np.empty(0, dtype=np.int64)
    best_j = np.empty(0, dtype=np.int64)
    for start in range(0, n, row_block):
        end = min(start + row_block, n)
        # Use separate normalization for each direction as in the original.
        block = (score[start:end, :] - mean) / std
        block += (score[:, start:end].T - mean) / std
        ii, jj = np.nonzero(np.arange(start, end)[:, None] < np.arange(n)[None, :])
        values = block[ii, jj]
        ii = ii + start
        values = np.concatenate((best_scores, values))
        ii = np.concatenate((best_i, ii))
        jj = np.concatenate((best_j, jj))
        if len(values) > top_k:
            # Lexicographic tie breaking is deterministic even at the cutoff.
            cutoff = np.partition(values, len(values) - top_k)[len(values) - top_k]
            above = np.flatnonzero(values > cutoff)
            tied = np.flatnonzero(values == cutoff)
            tie_order = np.lexsort((jj[tied], ii[tied]))
            selected = np.concatenate((above, tied[tie_order[:top_k - len(above)]]))
            values, ii, jj = values[selected], ii[selected], jj[selected]
        best_scores, best_i, best_j = values, ii, jj
    order = np.lexsort((best_j, best_i, -best_scores))
    names = np.asarray(names, dtype=str)
    return pd.DataFrame({"gene1": names[best_i[order]], "gene2": names[best_j[order]],
                         "avg_score": best_scores[order]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--score_map", help="Pickled CelluFormer score map")
    sources.add_argument("--score_dir", help="Recursively process pickle files")
    parser.add_argument("--data", required=True, help="h5ad with matching gene order")
    parser.add_argument("--output", help="Output CSV for --score_map")
    parser.add_argument("--top_k", type=int, default=200000)
    parser.add_argument("--row_block", type=int, default=64)
    args = parser.parse_args()
    if args.score_map and not args.output:
        parser.error("--output is required with --score_map")
    if args.score_dir and args.output:
        parser.error("--output is only valid with --score_map")
    adata = ad.read_h5ad(args.data, backed="r")
    names = adata.var_names.to_numpy()
    adata.file.close()
    paths = [Path(args.score_map)] if args.score_map else sorted(
        p for p in Path(args.score_dir).rglob("*") if p.is_file() and p.suffix in (".pkl", ".pickle"))
    if not paths:
        raise SystemExit("No score map files found")
    for path in paths:
        with path.open("rb") as fh:
            score = pickle.load(fh)
        ranking = score_map_to_rank(score, names, args.top_k, args.row_block)
        output = Path(args.output) if args.score_map else Path(str(path) + "_ranking.csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        ranking.to_csv(output, index=False)
        print(f"Wrote {len(ranking)} pairs to {output}")


if __name__ == "__main__":
    main()
