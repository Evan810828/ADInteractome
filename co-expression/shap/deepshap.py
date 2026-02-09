#! /usr/bin/env /ua/choi267/anaconda3/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from torch.utils.data.dataset import ConcatDataset

import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import numpy as np

from sklearn.model_selection import KFold

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import matplotlib
import torch.nn as nn

from tqdm import tqdm

from sklearn.model_selection import train_test_split

matplotlib.style.use('ggplot')


import copy

import sys
import pandas as pd

import sys
sys.argv=['']
del sys

# Load your data
data1 = pd.read_csv('GRN_TFs_GE_sub.csv', header=0)
data2 = pd.read_csv('GRN_TGs_GE.csv', header=0)

TG = "MBP"

model_path = "/result/" + TG + ".pt"
train_path = "/result/" + TG + "_train.png"
test_path = "/result/" + TG + "_test.csv"
history_path = "/result/" + TG + "_history.csv"
MSE_path = "/result/MSE/" + TG + "_MSE.csv"
MSE_box_path = "/result/MSE/" + TG + "_MSE_box.png"
interaction_path = "/result/Interaction_matrix/" + TG + "_interaction.csv"


# Define a function for library size normalization and log transformation
def normalize_and_scale(data):
    total_counts = data.sum(axis=1)
    total_counts += 1e-8
    data_normalized = data.div(total_counts, axis=0) * 1e6  # Scale counts to 1 million per cell
    data_normalized_log = np.log1p(data_normalized + 1)
    return data_normalized_log

# Normalize and scale the data1
data1_normalized_scaled = normalize_and_scale(data1)

# Normalize and scale the data2
data2_normalized_scaled = normalize_and_scale(data2)

# Concatenate the normalized and scaled data
df3 = pd.concat([data1_normalized_scaled, data2_normalized_scaled.iloc[:, 0]], axis=1)

# Separate data and labels
data = df3.loc[:, df3.columns != df3.columns[df3.shape[1] - 1]].values
labels = df3.loc[:, TG].values

features = data.shape[1]

# Define your network
class Net(nn.Module):
    def __init__(self, features):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(features, features * 2),
            nn.BatchNorm1d(features * 2),
            nn.LeakyReLU(),

            nn.Linear(features * 2, features * 3),
            nn.BatchNorm1d(features * 3),
            nn.LeakyReLU(),

            nn.Linear(features * 3, features * 4),
            nn.BatchNorm1d(features * 4),
            nn.LeakyReLU(),

            nn.Linear(features * 4, features * 3),
            nn.BatchNorm1d(features * 3),
            nn.LeakyReLU(),

            nn.Linear(features * 3, features * 2),
            nn.BatchNorm1d(features * 2),
            nn.LeakyReLU(),

            nn.Linear(features * 2, features),
            nn.BatchNorm1d(features),
            nn.LeakyReLU(),

            nn.Linear(features, features//2),
            nn.BatchNorm1d(features//2),
            nn.LeakyReLU(),

            nn.Linear(features//2, 1),
            nn.BatchNorm1d(1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = x.float()
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        return x

# Define your custom dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Define your fit and validate functions
def fit(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch_idx, (data, labels) in enumerate(tqdm(dataloader, total=len(dataloader))):
        optimizer.zero_grad()
        data, labels = data.to(device), labels.to(device)
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(dataloader)
    return train_loss

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for data, labels in tqdm(dataloader, total=len(dataloader)):
            data = data.to(device)
            labels = labels.to(device)
            output = model(data)
            loss = criterion(output, labels)
            running_loss += loss.item()

    val_loss = running_loss / len(dataloader)
    return val_loss

# Set your device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define your parameters
batch_size = 32
epochs = 100
lr = 0.0001
k = 5  # Number of folds for cross-validation

# Initialize K-Fold cross-validation
splits = KFold(n_splits=k, shuffle=True, random_state=42+j)

# Define your loss function
criterion = nn.MSELoss()


# Split the data into train and validation (hold-out) sets
train_data, val_data, _, _ = train_test_split(data, labels, test_size=0.25, random_state=42, shuffle=True)

# Get the total number of data points
total_samples = len(data)

# Generate integer indices
all_indices = np.arange(total_samples)

# Split the indices into train and validation indices
train_indices, val_indices = train_test_split(all_indices, test_size=0.25, random_state=42, shuffle=True)

# Save train and validation data along with their indices
np.savetxt(indice_path + str(j) + '_train.csv', train_data, delimiter=',')
np.savetxt(indice_path + str(j) + '_val.csv', val_data, delimiter=',')
np.savetxt(indice_path + str(j) + '_train_indices.csv', train_indices, delimiter=',')
np.savetxt(indice_path + str(j) + '_val_indices.csv', val_indices, delimiter=',')

# Initialize best validation loss
best_val_loss = float('inf')
best_model_weights = None

# 2. Train using train data only and 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Iterate over each fold
# Initialize history dictionary
history = {
    'fold_num': [],
    'epoch_num': [],
    'train_loss': [],
    'val_train_loss': [],
    'val_loss': []
}

# Iterate over each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
    print('Fold {}'.format(fold + 1))
    
    # Separate train and validation data for this fold
    train_data_fold, val_data_fold = data[train_idx], data[val_idx]
    train_labels_fold, val_labels_fold = labels[train_idx], labels[val_idx]

    # DataLoader for training data
    train_dataset_fold = CustomDataset(train_data_fold, train_labels_fold)
    train_loader_fold = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True)

    # DataLoader for validation data
    val_dataset_fold = CustomDataset(val_data_fold, val_labels_fold)
    val_loader_fold = DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False)

    # Initialize the model, optimizer, and scheduler for each fold
    model = Net(features).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    lambda1 = lambda epoch: epoch / 1000
    scheduler = lr_scheduler.LambdaLR(optimizer, lambda1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")

        # Train the model and get the training loss
        train_epoch_loss = fit(model, train_loader_fold, optimizer, criterion, device)
        print(f"train Loss: {train_epoch_loss:.4f}")
        
        # Validate the model on training data (within fold) and get the loss
        val_train_epoch_loss = validate(model, train_loader_fold, criterion, device)
        print(f"val Train Loss: {val_train_epoch_loss:.4f}")

        # Validate the model on validation data (within fold) and get the loss
        val_epoch_loss = validate(model, val_loader_fold, criterion, device)
        print(f"val Loss: {val_epoch_loss:.4f}")

        # Check if current validation loss is better than the best validation loss
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            # Save the current model as the best model
            best_model_path = model_path
            torch.save({
            'fold': fold + 1,
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'val_loss': best_val_loss,
            }, best_model_path)

        # Append fold and epoch numbers to history
        history['fold_num'].append(fold + 1)
        history['epoch_num'].append(epoch + 1)

        # Append training and validation losses to history
        history['train_loss'].append(train_epoch_loss)
        history['val_train_loss'].append(val_train_epoch_loss)
        history['val_loss'].append(val_epoch_loss)

        # Step the scheduler
        scheduler.step()

# Save history to a CSV file
history_df = pd.DataFrame.from_dict(history)
history_df.to_csv(history_path, sep=',', header=True, index=False)


# Report train losses
plt.rcParams["figure.figsize"] = [10, 8]
plt.rcParams["figure.autolayout"] = True

fig, ((ax1, ax2)) = plt.subplots(2,1)

# history_df = pd.DataFrame(history)
history_df.set_index('epoch_num', inplace=True)
history_df.groupby('fold_num')['train_loss'].plot(ax=ax1, title ='MSE (train)')
history_df.groupby('fold_num')['val_train_loss'].plot(ax=ax2, title ='MSE (val-train)')

plt.legend()
plt.savefig(train_path)


# Report evaluation metric
data_1 = history_df[history_df.fold_num == 1]
data_1_min = np.min(data_1['val_loss'])
data_2 = history_df[history_df.fold_num == 2]
data_2_min = np.min(data_2['val_loss'])
data_3 = history_df[history_df.fold_num == 3]
data_3_min = np.min(data_3['val_loss'])
data_4 = history_df[history_df.fold_num == 4]
data_4_min = np.min(data_4['val_loss'])
data_5 = history_df[history_df.fold_num == 5]
data_5_min = np.min(data_5['val_loss'])

MSE_mean = np.mean((data_1_min,data_2_min,data_3_min,data_4_min,data_5_min))
MSE_sd = np.std((data_1_min,data_2_min,data_3_min,data_4_min,data_5_min)) * 1.96

print(MSE_mean)
print(MSE_sd)

MSE_df = pd.DataFrame([MSE_mean,MSE_sd])
MSE_df.to_csv(MSE_path + '_df',
                  sep = ',',header=True,index=True)
np.savetxt(MSE_path, MSE_df, delimiter=',')

MSE_history = pd.DataFrame([(data_1_min,data_2_min,data_3_min,data_4_min,data_5_min)])
MSE_history.to_csv(MSE_path + '_history_df',
                  sep = ',',header=True,index=True)

np.savetxt(MSE_path, MSE_history, delimiter=',')


def matric2dic(hessian,k):
    IS = {}
    for i in range(len(hessian[0])):
        for j in range(i+1, len(hessian[0])):
            interation = 'Interaction: '
            interation = interation + str(i + 1) + ' ' + str(j + 1) + ' '
            IS[interation] = hessian[i][j]
    Sorted_IS = [(k, IS[k]) for k in sorted(IS, key=IS.get, reverse=True)]
    return IS, Sorted_IS

def delta_main(predictor, x, baseline, main_index):
    T = baseline.clone()
    Ti = baseline.clone(); Ti[main_index] = x[main_index]
    input = torch.cat([Ti, T]).reshape(2, -1)
    with torch.no_grad(): # to prevent gradients build up in memeory
        output = predictor(input)[0] # NOTE
    return output[0].item() - output[1].item()

def deltaF_v1(predictor, x, baseline, perm):
    num_gene = x.shape[0]
    shapleyis = torch.zeros(num_gene, num_gene)
    T_base = baseline.clone()
    indices = torch.triu_indices(num_gene, num_gene, 1) # all i,j pairs in the original double for loop order
    all_inputs = torch.zeros((4, num_gene * (num_gene - 1) // 2, num_gene))
    for idx, (i, j) in enumerate(zip(indices[0], indices[1])):
        T = perm[:i] if i > 0 else []
        T_base[T] = x[T]
        interaction = perm[torch.tensor([i,j])]
        Tij = T_base.clone()
        Tij[interaction] = x[interaction]
        Ti = T_base.clone()
        Ti[interaction[0]] = x[interaction[0]]
        Tj = T_base.clone()
        Tj[interaction[1]] = x[interaction[1]]
        all_inputs[0, idx] = Tij
        all_inputs[1, idx] = Ti
        all_inputs[2, idx] = Tj
        all_inputs[3, idx] = T_base
    with torch.no_grad(): # to prevent gradients build up in memeory
        outputs = predictor(all_inputs.view(-1, num_gene))[0].view(4, -1) # NOTE
    results = outputs[0] - outputs[1] - outputs[2] + outputs[3]
    shapleyis[perm[indices[0]], perm[indices[1]]] = results
    return shapleyis

def ShapleyValue(predictor, x, baseline):
    num_gene = x.shape[0]
    shapleyvalue = torch.zeros([num_gene])
    for i in range(num_gene):
        #print(str(i) + "_ShapleyValue")
        shapleyvalue[i] = delta_main(predictor, x, baseline, [i])
    return shapleyvalue

def ShapleyIS(predictor, x, baseline, num_permutation):
    num_gene = x.shape[0]
    SHAPLEYIS = torch.zeros([num_gene, num_gene])
    for _ in range(num_permutation):
        perm = torch.randperm(num_gene)
        shapleyis = deltaF_v1(predictor, x, baseline, perm)
        SHAPLEYIS += shapleyis
    SHAPLEYIS = (SHAPLEYIS + SHAPLEYIS.T) / num_permutation
    SHAPLEYIS = SHAPLEYIS +  torch.diag(ShapleyValue(predictor, x, baseline)) # O(n)
    return SHAPLEYIS

def GlobalSIS(predictor, X, baseline, num_permutation = 2):
    X = X.to('cpu')
    baseline = baseline.to('cpu')
    num_individual, num_gene = X.shape
    Shapely = torch.zeros([num_gene, num_gene]) # maybe use tensor
    feature_importance = torch.zeros([num_individual, num_gene])
    for i in tqdm(range(num_individual)):
        print(str(i) + "_GlobalSIS")    
        x = X[i]
        current_row_shapley = abs(ShapleyIS(predictor, x, baseline, num_permutation))
        Shapely = Shapely + current_row_shapley
        feature_importance[i] = torch.diag(current_row_shapley)
    Shapely = Shapely / num_individual
    GlobalSIS, topGlobalSIS = matric2dic(Shapely, 10)
    return GlobalSIS, topGlobalSIS, Shapely, feature_importance

#%%
model.eval()
model.to('cpu')


SHAP_test = np.array(val_data)
test = torch.tensor(SHAP_test)
test = test.to(device)
test = test.float()
test.shape[1]

gene_test = model(test.to(device)); 
# gene_test.detach_()
baseline = torch.mean(gene_test, dim = 0).view(1,-1).to(device)

model.eval()
model(test).shape
baseline = test.mean(dim=0)
model_func = lambda x: (model(x), None)

GlobalSIS, topGlobalSIS, Interaction_matrix,feature_importance = GlobalSIS(model_func, test, baseline)

np.savetxt(interaction_path, Interaction_matrix, delimiter=",")

