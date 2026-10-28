"""Reproduce the global cell split and write one h5ad per subclass.

Only X is loaded into memory; layers and other large matrices remain backed.
"""
import argparse
from pathlib import Path

import anndata as ad
import hdf5plugin
import numpy as np
from sklearn.model_selection import train_test_split


def split_cell_types(data_path, train_output_path, test_output_path):
    data = ad.read_h5ad(data_path, backed="r")
    try:
        if not data.obs_names.is_unique:
            raise ValueError("Observation names must be unique to reproduce the original split")
        columns = ["Overall AD neuropathological Change", "Subclass"]
        obs = data.obs[columns]
        selected = np.flatnonzero(obs[columns[0]].isin(["Not AD", "High"]).to_numpy())
        # Splitting positional indices uses the same sklearn permutation as splitting
        # the filtered X and labels, without creating two copies of that matrix.
        train, test = train_test_split(selected, test_size=0.2, random_state=42)
        print("Loading X into memory; all other matrices remain on disk", flush=True)
        matrix = data.X.to_memory() if hasattr(data.X, "to_memory") else data.X[:]
        for split_name, indices, output in (("train", train, train_output_path),
                                             ("test", test, test_output_path)):
            output = Path(output)
            output.mkdir(parents=True, exist_ok=True)
            split_obs = obs.iloc[indices]
            seen = set()
            for name, group in split_obs.groupby("Subclass", observed=True):
                filename = "cell_type_" + name.replace(" ", "_").replace("/", "_") + ".h5ad"
                if filename in seen:
                    raise ValueError(f"Subclass names collide at {filename}")
                seen.add(filename)
                # groupby retains row order, which is the original shuffled split order.
                positions = data.obs_names.get_indexer(group.index)
                print(f"Packing {split_name} {name}: {len(group)} cells", flush=True)
                subset = ad.AnnData(X=matrix[positions, :], var=data.var.copy(), obs=group.copy())
                subset.write_h5ad(output / filename, compression=hdf5plugin.FILTERS["zstd"],
                                  compression_opts=hdf5plugin.Zstd(clevel=5).filter_options)
    finally:
        data.file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="/scratch0/zx22/yifan/SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad")
    parser.add_argument("--train_output_path", default="data/split_train")
    parser.add_argument("--test_output_path", default="data/split_test")
    args = parser.parse_args()
    split_cell_types(args.data, args.train_output_path, args.test_output_path)
