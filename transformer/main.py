"""Training and evaluation script for the CelluFormer model.

Trains a transformer-based classifier on single-cell RNA-seq data to distinguish
between 'Not AD' and 'High' AD neuropathological change categories.

Example usage:
    python transformer/main.py --config config.json \\
        --data data/SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad \\
        --output output_models/global_training --epochs 4
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import anndata as ad
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
import wandb
import os
import json
import argparse
import hdf5plugin
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformer.celluformer import CelluFormer
from transformer.dataset import LazyGlobalPaddedDataset
from utils import setup_seed, load_checkpoint

# Subclass names are the exact values in SEA-AD's obs["Subclass"].
NEURONAL_SUBCLASSES = (
    "Chandelier", "L2/3 IT", "L4 IT", "L5 ET", "L5 IT", "L5/6 NP",
    "L6 IT", "L6 CT", "L6b", "L6 IT Car3", "Lamp5", "Pax6", "Vip",
    "Sst Chodl", "Sst", "Pvalb", "Lamp5 Lhx6", "Sncg",
)


def select_neuronal_split(obs):
    """Split all eligible cells first, then retain the 18 neuronal subclasses.

    This preserves the per-cell train/test assignment used by split_cell_type.py
    and hence by the per-subclass MLP models.
    """
    classes = obs['Overall AD neuropathological Change']
    rows = np.flatnonzero(classes.isin(['Not AD', 'High']).to_numpy())
    labels = classes.iloc[rows].map({'Not AD': 0, 'High': 1}).to_numpy(dtype=np.int64)
    train_pos, test_pos = train_test_split(np.arange(len(rows)), test_size=0.2, random_state=42)
    is_neuronal = obs['Subclass'].iloc[rows].isin(NEURONAL_SUBCLASSES).to_numpy()
    train_pos = train_pos[is_neuronal[train_pos]]
    test_pos = test_pos[is_neuronal[test_pos]]
    if not len(train_pos) or not len(test_pos):
        raise ValueError('No neuronal cells in one or both train/test splits')
    return rows[train_pos], rows[test_pos], labels[train_pos], labels[test_pos]
    
def init(JSON_PATH):
    """Initialize wandb and load hyperparameters from a JSON config file."""
    wandb.login()
    wandb.init(project="gene-interact")
    print(f"Wandb run initialized as {wandb.run.name}")

    # hyperparameters
    config = json.load(open(JSON_PATH))
    wandb.config.update({
        "num_genes": config["num_genes"],
        "embedding_dim": config["embedding_dim"],
        "dim_feedforward": config["dim_feedforward"],
        "head": config["head"],
        "depth": config["depth"],
        "dropout": config["dropout"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"]
    })
    return config

def load_data(data_path, is_test):
    """Load and split scRNA-seq data into train/test datasets.

    Args:
        data_path: Path to the .h5ad file.
        is_test: If True, only creates the test dataset (train_dataset will be None).

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    print("Loading data...")
    # Avoid loading every h5ad layer and avoid extra copies of X per split.
    # The index-based split produces exactly the same permutation as splitting
    # the filtered matrix and labels directly.
    data_cell = ad.read_h5ad(data_path, backed="r")
    try:
        train_rows, test_rows, train_labels, test_labels = select_neuronal_split(data_cell.obs)
        print(f"Neuronal cells: {len(train_rows)} train, {len(test_rows)} test", flush=True)
        print("Subclass counts (train / test):", flush=True)
        for subclass in NEURONAL_SUBCLASSES:
            counts = (int((data_cell.obs['Subclass'].iloc[train_rows] == subclass).sum()),
                      int((data_cell.obs['Subclass'].iloc[test_rows] == subclass).sum()))
            print(f"  {subclass}: {counts[0]} / {counts[1]}", flush=True)
        X = data_cell.X.to_memory()
    finally:
        data_cell.file.close()
    if getattr(X, "format", None) != "csr":
        X = X.tocsr()
    train_dataset = None if is_test else LazyGlobalPaddedDataset(X, train_rows, train_labels)
    test_dataset = LazyGlobalPaddedDataset(X, test_rows, test_labels)
    return train_dataset, test_dataset

def train(train_dataset, test_dataset, config, args):
    """Train the CelluFormer model with periodic evaluation and checkpointing."""
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle = False)
    
    model = CelluFormer(config["num_genes"], config["embedding_dim"], config["dim_feedforward"], config["head"], config["depth"], config["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    model = nn.DataParallel(model, device_ids=args.device_ids)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    best_f1 = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        test_loss, acc, f1_value = 0, 0, 0

        for batch_idx, (batch_sequences, batch_values, batch_labels) in enumerate(train_dataloader): #enumerate(dataloader)
            # Move tensors to GPU
            batch_sequences = batch_sequences.to(device)
            batch_values = batch_values.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            output = model(batch_sequences, batch_values)
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % args.train_loss_log == 0:
                print(f"    Batch {batch_idx}/{len(train_dataloader)} Total Loss: {total_loss / (batch_idx+1):.4f}")
                wandb.log({"epoch": epoch, "step": batch_idx, "train_loss": total_loss / (batch_idx+1)})
                
            if (not args.epoch_wise_test) and batch_idx % args.eval_every == 0 and batch_idx > 0 and epoch > 0:
                test_loss, acc, f1_value, _ = test(model, test_dataloader, criterion, device)
                model.train()
                wandb.log({"epoch": epoch, "step": batch_idx, "test_loss": test_loss, "accuracy": acc, "f1_score": f1_value})
                if f1_value > best_f1:
                    best_f1 = f1_value
                    print(f"    Model saved to {args.output}/{wandb.run.name}/epoch_{epoch}_batch_{batch_idx}.pth")
                    torch.save(model.state_dict(), f"{args.output}/{wandb.run.name}/epoch_{epoch}_batch_{batch_idx}.pth")
        
        if args.epoch_wise_test:
            test_loss, acc, f1_value, _ = test(model, test_dataloader, criterion, device)
            print({"epoch": epoch, "test_loss": test_loss, "accuracy": acc, "f1_score": f1_value})
            if f1_value > best_f1:
                best_f1 = f1_value
                print(f"    Model saved to {args.output}/{wandb.run.name}/epoch_{epoch}_batch_{batch_idx}.pth")
                torch.save(model.state_dict(), f"{args.output}/{wandb.run.name}/epoch_{epoch}_batch_{batch_idx}.pth")
                    
    return


def test(model, test_dataloader, criterion, device):
    """Evaluate model on test data. Returns (loss, accuracy, f1_score, predictions)."""
    print("Evaluating...")
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_idx, (batch_sequences, batch_values, batch_labels) in enumerate(test_dataloader):
            # Move tensors to GPU
            batch_sequences = batch_sequences.to(device)
            batch_values = batch_values.to(device)
            batch_labels = batch_labels.to(device)

            output = model(batch_sequences, batch_values)
            loss = criterion(output, batch_labels)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
            y_true.extend(batch_labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
    accuracy = (100 * (correct / total))
    f1_value = f1_score(y_true, y_pred, average="weighted") * 100
    
    print(f"Test loss: {total_loss / len(test_dataloader)}, Accuracy: {accuracy}, f1_score: {f1_value}")
    
    return total_loss / len(test_dataloader), accuracy, f1_value, y_pred
    
def parse_list(arg_value):
    return [int(item) for item in arg_value.split(',')]

def load_model(model_path, config):
    """Load a CelluFormer model from a DataParallel checkpoint."""
    return load_checkpoint(config, model_path)
    
    
if __name__ == "__main__":
    ## example usage:
    ## python transformer/main.py --config config.json --data SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad --output output_models --device_ids 0,1,2,3 --eval_every 1000 --epochs 4 --device cuda --model_path model.pth
    parser = argparse.ArgumentParser(description="gene-transformer")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the JSON file containing the hyperparameters")
    parser.add_argument("--data", type=str, default="data/SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad", help="Path to the h5ad file containing the data")
    parser.add_argument("--output", type=str, default="output_models/global_training", help="Path to the output directory")
    parser.add_argument("--device_ids", type=parse_list, default=[0, 1, 2, 3], help="List of device ids to use")
    parser.add_argument("--eval_every", type=int, default=1000, help="Evaluate the model every n steps")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs to train the model")
    parser.add_argument("--is_test", action="store_true", help="Test the model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training and evaluation")
    parser.add_argument("--model_path", type=str, default="model.pth", help="Path to the model")
    parser.add_argument("--epoch_wise_test", type=bool, default=False, help="Test the model after each epoch")
    parser.add_argument("--train_loss_log", type=int, default=50, help="Log the train loss every n steps")
    args = parser.parse_args()
    
    setup_seed(42)
    
    if (not args.is_test):
        config = init(args.config)
        os.makedirs(f"{args.output}/{wandb.run.name}", exist_ok=True)
    train_dataset, test_dataset = load_data(args.data, args.is_test)
    
    ## hyperparameters format:
    ## {
    ##     "embedding_dim": 128,
    ##     "dim_feedforward": 512,
    ##     "head": 8,
    ##     "depth": 4,
    ##     "dropout": 0.1,
    ##     "batch_size": 64,
    ##     "learning_rate": 0.001,
    ## }
    
    if (not args.is_test):
        train(train_dataset, test_dataset, config, args)
    else:
        config = json.load(open(args.config))
        test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle = False)
        model = load_model(args.model_path, config)
        model = nn.DataParallel(model, device_ids=args.device_ids)
        model = model.to(args.device)
        test(model, test_dataloader, nn.CrossEntropyLoss(), args.device)
