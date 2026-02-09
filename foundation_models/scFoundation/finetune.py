from get_embedding import *
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import scanpy as sc
from tqdm import tqdm
from sklearn.metrics import f1_score
import wandb
import os
import json
import argparse
import hdf5plugin
from torch.utils.data import Dataset

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def init():
    wandb.login()
    wandb.init(project="scFoundation-finetune")
    print(f"Wandb run initialized as {wandb.run.name}")

class LinearProbingClassifier(nn.Module):

    def __init__(self, ckpt_path,frozenmore=False):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.frozenmore = frozenmore

    def build(self):
        model,model_config = load_model_frommmf(self.ckpt_path)
        self.token_emb = model.token_emb
        self.pos_emb = model.pos_emb
        self.encoder = model.encoder
        
        if self.frozenmore:
            for _,p in self.token_emb.named_parameters():
                p.requires_grad = False
            for _,p in self.pos_emb.named_parameters():
                p.requires_grad = False
            print('self.pos_emb and self.token_emb also frozen')
        
        for na, param in self.encoder.named_parameters():
            param.requires_grad = False
        for na, param in self.encoder.transformer_encoder[-2].named_parameters():
            print('self.encoder.transformer_encoder ',na,' have grad')
            param.requires_grad = True


        self.fc1 = nn.Sequential(
        nn.Linear(model_config['encoder']['hidden_dim'], 256),
        nn.ReLU(),
        nn.Linear(256, 2)  # ['n_class']
        ) 
        self.norm = torch.nn.BatchNorm1d(model_config['encoder']['hidden_dim'], affine=False, eps=1e-6)
        self.model_config = model_config
        
    def forward(self, X, *args, **kwargs):
        x = X
        value_labels = x > 0
        x, x_padding = gatherData(x, value_labels, self.model_config['pad_token_id'])
        data_gene_ids = torch.arange(19264, device=x.device).repeat(x.shape[0], 1)
        position_gene_ids, _ = gatherData(data_gene_ids, value_labels,
                                        self.model_config['pad_token_id'])
        
        x = self.token_emb(torch.unsqueeze(x, 2).float(), output_weight = 0)
        position_emb = self.pos_emb(position_gene_ids)
        x += position_emb

        logits = self.encoder(x,x_padding)

        # mlp
        logits_pooled = logits.mean(dim=1)
        output = self.fc1(logits_pooled)

        return output
    
class scFoundationDataset(Dataset):
    def __init__(self, X, y, main_gene_indices): # X is a sparse matrix
        self.seq = []
        self.labels = torch.tensor(y)
        self.main_genes_indices = main_gene_indices

        print("Initializing dataset...")
        for i in tqdm(range(X.shape[0]), desc="Processing data"):
            x = X[i].toarray()
            x_tensor = torch.from_numpy(x).float()[0]
            
            self.seq.append(x_tensor[main_gene_indices])

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, index):
        sequence = self.seq[index]
        label = self.labels[index]
        return sequence, label
    
def load_data(data_path, batch_size=4):
    data_cell = sc.read_h5ad(data_path)

    filtered_data = data_cell[data_cell.obs['Overall AD neuropathological Change'].isin(['Not AD', 'High'])]
    X_filtered = filtered_data.X
    y_filtered = filtered_data.obs['Overall AD neuropathological Change']
    label_mapping = {'Not AD': 0, 'High': 1}
    y_encoded = y_filtered.map(label_mapping)
    
    gene_list_df = pd.read_csv('OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
    gene_list = list(gene_list_df['gene_name'])
    main_gene_indices = [i for i in filtered_data.var.index.get_indexer(gene_list) if i != -1]
    
    dataset = scFoundationDataset(X_filtered, y_encoded, main_gene_indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def finetune_model(learning_rate=1e-6, epochs=10, ckpt_path='models/model.ckpt', batch_size=128):
    train_dataloader = load_data('data/split_train/neuron_data.h5ad', 32)
    test_dataloader = load_data('data/split_test/neuron_data.h5ad', 8)
    
    finetune_model = LinearProbingClassifier(ckpt_path=ckpt_path)
    finetune_model.build()
    
    optimizer = torch.optim.Adam(finetune_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    finetune_model = nn.DataParallel(finetune_model, device_ids=[0,1,2,3,4,5,6,7])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    finetune_model = finetune_model.to(device)

    for epoch in range(epochs):
        finetune_model.train()
        total_loss = 0.0

        for batch_idx, (batch_sequences, batch_labels) in enumerate(train_dataloader):
            try:
                # Move tensors to GPU
                batch_sequences = batch_sequences.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()
                output = finetune_model(batch_sequences)
                loss = criterion(output, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if batch_idx % 1 == 0:
                    print(f"    Batch {batch_idx}/{len(train_dataloader)} Total Loss: {total_loss / (batch_idx+1):.4f}")
                    wandb.log({"epoch": epoch, "step": batch_idx, "train_loss": total_loss / (batch_idx+1)})
            except Exception as e:
                print(batch_sequences.shape)
                  
        finetune_model.eval()
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        test_loss, accuracy, f1_value = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (batch_sequences, batch_labels) in enumerate(test_dataloader):
                try:
                    batch_sequences = batch_sequences.to(device)
                    batch_labels = batch_labels.to(device)
                    output = finetune_model(batch_sequences)
                    loss = criterion(output, batch_labels)
                    test_loss += loss.item()
                    
                    _, predicted = torch.max(output.data, 1)
                    total += batch_labels.shape[0]
                    correct += (predicted == batch_labels).sum().item()
                    y_true.extend(batch_labels.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
                        
                except Exception as e:
                    print(batch_sequences.shape)
                
            print(correct, total)
            test_loss /= len(test_dataloader)
            accuracy = (100 * float(correct / total))
            f1_value = f1_score(y_true, y_pred, average="weighted") * 100
            print(f"Epoch {epoch} Test Loss: {test_loss:.4f} Accuracy: {accuracy:.4f} F1: {f1_value:.4f}")
            wandb.log({"epoch": epoch, "test_loss": test_loss, "accuracy": accuracy, "f1_score": f1_value})
    
        torch.save(finetune_model.state_dict(), f"./finetuned_ckpts/finetuned_model_epoch_{epoch}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--ckpt_path", type=str, default='models/model.ckpt')
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    setup_seed(42)
    init()
    wandb.config.update(args)
    finetune_model(args.learning_rate, args.epochs, args.ckpt_path, args.batch_size)