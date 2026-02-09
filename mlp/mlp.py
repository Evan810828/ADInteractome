"""MLP baseline model for cell-type-specific AD classification.

Trains a simple multi-layer perceptron on dense gene expression vectors
as a baseline comparison to the transformer-based approach.

Example usage:
    python mlp/mlp.py --cell_type L5_ET --epochs 10
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import scanpy as sc
import wandb


def load_data(data_path: str):
    """Load scRNA-seq data from an h5ad file and return (X, y) pairs for binary AD classification."""
    print(f"Loading data from {data_path} …")
    data_cell = sc.read_h5ad(data_path)

    keep = data_cell.obs["Overall AD neuropathological Change"].isin(["Not AD", "High"])
    filtered_data = data_cell[keep]
    X_filtered = filtered_data.X  # sparse matrix
    y_filtered = filtered_data.obs["Overall AD neuropathological Change"]

    label_mapping = {"Not AD": 0, "High": 1}
    y_encoded = y_filtered.map(label_mapping)

    # return list of dense tensors & int labels
    return list(zip(X_filtered.todense(), y_encoded))

class MLP(nn.Module):
    """Simple 3-layer MLP for binary cell classification from gene expression.

    Args:
        num_genes: Input dimension (number of genes).
        hidden1: First hidden layer dimension.
        hidden2: Second hidden layer dimension.
    """

    def __init__(self, num_genes: int, hidden1: int = 128, hidden2: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_genes, hidden1, bias=False),
            nn.Dropout(p=0.5),
            nn.Softplus(beta=1, threshold=20),
            nn.Linear(hidden1, hidden2),
            nn.Dropout(p=0.5),
            nn.Softplus(beta=1, threshold=20),
            nn.Linear(hidden2, 2)  # binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.squeeze(x, dim=1)
        return self.layers(x)

def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                    criterion: nn.Module, device: torch.device, epoch: int) -> float:
    """Train the model for one epoch and return the average loss."""
    model.train()
    running_loss = 0.0

    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 10 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            print(f"  Batch {batch_idx}/{len(dataloader)} — loss: {avg_loss:.4f}")
            wandb.log({"epoch": epoch, "step": batch_idx, "train_loss": avg_loss})

    return running_loss / len(dataloader)


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
             device: torch.device):
    """Evaluate model and return (avg_loss, accuracy, f1_score)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            total += y.size(0)
            correct += (preds == y).sum().item()

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    f1 = f1_score(y_true, y_pred, average="weighted") * 100
    print(f"Test loss: {avg_loss:.4f}, accuracy: {accuracy:.2f}%, f1: {f1:.2f}%")

    return avg_loss, accuracy, f1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_type", required=True, type=str, help="Target cell‑type tag")
    parser.add_argument("--epochs", default=10, type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", default=128, type=int, help="Mini‑batch size")
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize Weights & Biases (store key in env var for safety)
    wandb.login()
    wandb.init(project="gene-interact-MLP")
    print(f"Wandb run initialized as {wandb.run.name}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Dataloaders
    train_ds = load_data(f"data/split_train/cell_type_{args.cell_type}.h5ad")
    test_ds = load_data(f"data/split_test/cell_type_{args.cell_type}.h5ad")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Model & optimizer
    NUM_GENES = 36601
    model = MLP(NUM_GENES)
    model = nn.DataParallel(model.to(device), device_ids=[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    out_dir = Path(f"output_models/MLP/{args.cell_type}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_dl, optimizer, criterion, device, epoch)
        test_loss, acc, f1 = evaluate(model, test_dl, criterion, device)

        wandb.log({"epoch": epoch, "test_loss": test_loss, "accuracy": acc, "f1_score": f1})

        # Save a checkpoint every 5 epochs (starting from epoch 5)
        if (epoch + 1) % 5 == 0:
            ckpt_path = out_dir / f"mlp_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
