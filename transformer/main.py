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
import hashlib

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

def _data_signature(dataset):
    """Identify the selected rows and labels used by the real training dataset."""
    digest = hashlib.sha256()
    digest.update(str(len(dataset)).encode())
    for name in ("rows", "labels"):
        value = getattr(dataset, name, None)
        if value is not None:
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            digest.update(np.asarray(value).tobytes())
    digest.update(str(getattr(dataset, "max_length", None)).encode())
    return digest.hexdigest()


def _file_identity(path):
    path = os.path.realpath(path)
    st = os.stat(path)
    return {"path": path, "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _save_recovery(path, state):
    """Publish a complete checkpoint atomically; leave the previous one on failure."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


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
    best_checkpoint_path = None
    start_epoch, next_batch, total_loss = 0, 0, 0.0
    resume = None
    run_dir = os.path.join(args.output, wandb.run.name)
    os.makedirs(run_dir, exist_ok=True)
    recovery_path = args.recovery_path or os.path.join(run_dir, "recovery_latest.pt")
    identity = {"config": config, "data": _file_identity(args.data),
                "train": _data_signature(train_dataset), "test": _data_signature(test_dataset),
                "eval_every": args.eval_every, "epoch_wise_test": args.epoch_wise_test}
    if args.resume_checkpoint:
        resume = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        if resume["identity"] != identity:
            raise ValueError("Recovery checkpoint does not match data, split, configuration or evaluation settings")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        start_epoch, next_batch = resume["epoch"], resume["next_batch"]
        total_loss, best_f1 = resume["total_loss"], resume["best_f1"]
        best_checkpoint_path = resume["best_checkpoint_path"]
        if best_checkpoint_path is not None and not os.path.isfile(best_checkpoint_path):
            raise FileNotFoundError(f"Best inference checkpoint is missing: {best_checkpoint_path}")
        torch.set_rng_state(resume["cpu_rng"])
        if torch.cuda.is_available():
            if resume["cuda_rng"] is None:
                raise ValueError("Cannot exactly resume a CPU checkpoint on CUDA")
            torch.cuda.set_rng_state_all(resume["cuda_rng"])
        elif resume["cuda_rng"] is not None:
            raise ValueError("Cannot exactly resume a CUDA checkpoint on CPU")
        print(f"Resuming epoch {start_epoch}, batch {next_batch} from {args.resume_checkpoint}")

    def save_recovery(epoch, next_batch, loss, epoch_rng):
        _save_recovery(recovery_path, {
            "version": 1, "identity": identity, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "epoch": epoch, "next_batch": next_batch,
            "total_loss": loss, "best_f1": best_f1, "best_checkpoint_path": best_checkpoint_path,
            "epoch_rng": epoch_rng,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        })

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if epoch != start_epoch or resume is None:
            total_loss = 0.0
            next_batch = 0
        test_loss, acc, f1_value = 0, 0, 0

        # Recreate the original shuffled iterator, then restore checkpoint RNG
        # after skipping consumed batches. The current dataset is deterministic
        # and DataLoader uses num_workers=0; dropout must not be replayed.
        if resume is not None and epoch == start_epoch and next_batch:
            epoch_rng = resume["epoch_rng"]
            checkpoint_cpu_rng = torch.get_rng_state()
            torch.set_rng_state(epoch_rng)
            iterator = iter(train_dataloader)
            for _ in range(next_batch):
                next(iterator)
            torch.set_rng_state(checkpoint_cpu_rng)
        else:
            epoch_rng = torch.get_rng_state()
            iterator = iter(train_dataloader)

        for batch_idx, (batch_sequences, batch_values, batch_labels) in enumerate(iterator, start=next_batch):
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
                    best_checkpoint_path = os.path.abspath(os.path.join(run_dir, f"epoch_{epoch}_batch_{batch_idx}.pth"))
                    print(f"    Model saved to {best_checkpoint_path}")
                    torch.save(model.state_dict(), best_checkpoint_path)

            if args.checkpoint_every > 0 and (batch_idx + 1) % args.checkpoint_every == 0 and batch_idx + 1 < len(train_dataloader):
                save_recovery(epoch, batch_idx + 1, total_loss, epoch_rng)
        
        if args.epoch_wise_test:
            test_loss, acc, f1_value, _ = test(model, test_dataloader, criterion, device)
            print({"epoch": epoch, "test_loss": test_loss, "accuracy": acc, "f1_score": f1_value})
            if f1_value > best_f1:
                best_f1 = f1_value
                best_checkpoint_path = os.path.abspath(os.path.join(run_dir, f"epoch_{epoch}_batch_{batch_idx}.pth"))
                print(f"    Model saved to {best_checkpoint_path}")
                torch.save(model.state_dict(), best_checkpoint_path)
        save_recovery(epoch + 1, 0, 0.0, None)
        resume = None
    print(f"Best inference checkpoint: {best_checkpoint_path} (weighted F1: {best_f1})")
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
    parser.add_argument("--checkpoint_every", type=int, default=250, help="Save resumable state every N batches; 0 disables intra-epoch saves")
    parser.add_argument("--recovery_path", type=str, default=None, help="Path for the latest resumable training checkpoint")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Resume training from a recovery checkpoint")
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
