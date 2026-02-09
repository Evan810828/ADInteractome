import scanpy as sc
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import pandas as pd
import hdf5plugin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gene-transformer")
    parser.add_argument("--data", type=str, default="/scratch0/zx22/yifan/SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad", help="Path to the h5ad file containing the data")
    parser.add_argument("--train_output_path", type=str, default="data/split_train", help="Path to save the training data")
    parser.add_argument("--test_output_path", type=str, default="data/split_test", help="Path to save the testing data")
    args = parser.parse_args()
    
    os.makedirs(args.train_output_path, exist_ok=True)
    os.makedirs(args.test_output_path, exist_ok=True)
    
    # Loading the data
    print("Loading data...")
    data_cell = sc.read_h5ad(args.data)
    # Filter the data
    filtered_data = data_cell[data_cell.obs['Overall AD neuropathological Change'].isin(['Not AD', 'High'])]
    X_filtered = filtered_data.X
    y_filtered = filtered_data.obs['Overall AD neuropathological Change']
    label_mapping = {'Not AD': 0, 'High': 1}
    y_encoded = y_filtered.map(label_mapping)
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_encoded, test_size=0.2, random_state=42)

    # Save the train and test data
    train_idx_list = [y_encoded.index.get_loc(idx) for idx in y_train.index]

    for name, group in filtered_data.obs[['Overall AD neuropathological Change', 'Subclass']].iloc()[train_idx_list].groupby('Subclass'):
        matrix = filtered_data[group.index.to_list()].X
        print(f"Packing train {name}...")
        adata = sc.AnnData(X = matrix, var = data_cell.var, obs = group)
        # remove the blank space in name
        name = name.replace(" ", "_")
        name = name.replace("/", "_")
        adata.write_h5ad(os.path.join(args.train_output_path, 'cell_type_'+name+'.h5ad'), compression=hdf5plugin.FILTERS["zstd"], compression_opts=hdf5plugin.Zstd(clevel=5).filter_options)
        
    test_idx_list = [y_encoded.index.get_loc(idx) for idx in y_test.index]
    
    for name, group in filtered_data.obs[['Overall AD neuropathological Change', 'Subclass']].iloc()[test_idx_list].groupby('Subclass'):
        matrix = filtered_data[group.index.to_list()].X
        print(f"Packing test {name}...")
        adata = sc.AnnData(X = matrix, var = data_cell.var, obs = group)
        # remove the blank space in name
        name = name.replace(" ", "_")
        name = name.replace("/", "_")
        adata.write_h5ad(os.path.join(args.test_output_path, 'cell_type_'+name+'.h5ad'), compression=hdf5plugin.FILTERS["zstd"], compression_opts=hdf5plugin.Zstd(clevel=5).filter_options)