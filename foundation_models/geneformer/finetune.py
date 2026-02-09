import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import scanpy as sc
from tqdm import tqdm
from sklearn.metrics import f1_score
import wandb
import os
import argparse
import hdf5plugin
import pickle
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertForSequenceClassification

global gene_list
    
class VariableLengthSequenceDataset_revised(Dataset):
    def __init__(self, X, y): # X is a sparse matrix
        self.seq = []
        self.vals = []
        self.labels = torch.tensor(y)
        global gene_list

        print("Initializing dataset...")
        for i in tqdm(range(X.shape[0]), desc="Processing data"):
            x = X[i].toarray()[0][gene_list]
            x_tensor = torch.from_numpy(x).float().unsqueeze(0)
            
            non_zero_indices = torch.nonzero(x_tensor, as_tuple=True)[1]
            self.seq.append(non_zero_indices)
            
            scaling_factor = x_tensor[0][non_zero_indices]
            self.vals.append(scaling_factor)
            
        self.seq = nn.utils.rnn.pad_sequence([seq.clone().detach() for seq in self.seq], batch_first=True, padding_value=0)+1
        self.vals = nn.utils.rnn.pad_sequence([vals.clone().detach() for vals in self.vals], batch_first=True, padding_value=0)

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, index):
        sequence = self.seq[index]
        scaling_factor = self.vals[index]
        label = self.labels[index]
        return sequence, scaling_factor, label

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def init(args):
    wandb.login()
    wandb.init(project="gene-interact-Geneformer")
    print(f"Wandb run initialized as {wandb.run.name}")

    # hyperparameters
    wandb.config.update({
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    })

def load_data(data_path):
    # Loading the data
    global gene_list
    Geneformer_embedding_dict = pd.read_csv("geneformer/Geneformer_embedding_dict.csv")
    gene_list = Geneformer_embedding_dict['SEEAD_id'].values
    
    data_cell = sc.read_h5ad(data_path)

    filtered_data = data_cell[data_cell.obs['Overall AD neuropathological Change'].isin(['Not AD', 'High'])]
    X_filtered = filtered_data.X
    y_filtered = filtered_data.obs['Overall AD neuropathological Change']
    label_mapping = {'Not AD': 0, 'High': 1}
    y_encoded = y_filtered.map(label_mapping)

    
    dataset = VariableLengthSequenceDataset_revised(X_filtered, y_encoded)
    
    return dataset

def train(
        train_dataset, test_dataset, 
        args
    ):
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle = False)
    
    model = BertForSequenceClassification.from_pretrained("geneformer")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
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
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            output = model(batch_sequences, batch_values)
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"    Batch {batch_idx}/{len(train_dataloader)} Total Loss: {total_loss / (batch_idx+1):.4f}")
                wandb.log({"epoch": epoch, "step": batch_idx, "train_loss": total_loss / (batch_idx+1)})
                
        test_loss, acc, f1_value = test(model, test_dataloader, criterion, args.device)
        wandb.log({"epoch": epoch, "step": batch_idx, "test_loss": test_loss, "accuracy": acc, "f1_score": f1_value})
        if f1_value > best_f1:
            best_f1 = f1_value
            print(f"    Model saved to {args.output}/{wandb.run.name}/epoch_{epoch}_batch_{batch_idx}.pth")
            torch.save(model.state_dict(), f"{args.output}/{wandb.run.name}/epoch_{epoch}_batch_{batch_idx}.pth")
                    
    return


def test(model, test_dataloader, criterion, device):
    print("     Evaluating...")
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
    
    print(f"Test loss: {total_loss / len(y_pred)}, Accuracy: {accuracy}, f1_score: {f1_value}")
    
    return total_loss / len(y_pred), accuracy, f1_value
    
def parse_list(arg_value):
    return [int(item) for item in arg_value.split(',')]

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gene-transformer")
    parser.add_argument("--output", type=str, default="output_models/augment_feature", help="Path to the output directory")
    parser.add_argument("--device_ids", type=parse_list, default=[0, 1, 2, 3, 4, 5, 6, 7], help="List of device ids to use")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs to train the model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training and evaluation")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training")
    args = parser.parse_args()
    
    setup_seed(42)
    
    init(args)
    os.makedirs(f"{args.output}/{wandb.run.name}", exist_ok=True)
    # train_dataset = load_data('data/split_train/neuron_data.h5ad')
    # test_dataset = load_data('data/split_test/neuron_data.h5ad')
    train_dataset = load_data('data/split_train/cell_type_L5_ET.h5ad')
    test_dataset = load_data('data/split_test/cell_type_L5_ET.h5ad')
    
    train(train_dataset, test_dataset, args)