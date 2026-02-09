import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import hdf5plugin
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import scanpy as sc
import os
from torch.utils.data.sampler import WeightedRandomSampler

class zero_ICWS(Dataset):
    def __init__(self, X, y, dim, n_sample, seeds):
        self.parse_csr(X, y)

        self.count = len(self.idxs)
        self.seeds = seeds

        self.dim = dim # number of features in total
        self.n_sample = n_sample # number of samples to hash

        self.generate_random_number()
        self.generate_samples()
        self.scoring()
        
        self.idxs = nn.utils.rnn.pad_sequence([seq.clone().detach() for seq in self.idxs], batch_first=True, padding_value=0)+1
        self.weights = nn.utils.rnn.pad_sequence([vals.clone().detach() for vals in self.weights], batch_first=True, padding_value=0)

    def __getitem__(self, index):
        sequence = self.idxs[index]
        weight = self.weights[index]
        label = self.labels[index]

        return sequence, weight, label

    def __len__(self):
        return self.count

    def parse_csr(self, X, y):
        self.idxs = []
        self.weights = []
        self.labels = torch.tensor(y.to_numpy())

        print("Parsing CSR matrix...")
        for i in tqdm(range(X.shape[0]), desc="Processing data"):
            x = X[i].toarray()
            x_tensor = torch.from_numpy(x).float()

            non_zero_indices = torch.nonzero(x_tensor, as_tuple=True)[1]
            self.idxs.append(non_zero_indices)

            scaling_factor = x_tensor[0][non_zero_indices]
            self.weights.append(scaling_factor)

    def generate_random_number(self):
        print("Generating random numbers...")
        self.ran_r = [[] for _ in range(self.n_sample)]
        self.ran_c = [[] for _ in range(self.n_sample)]
        self.ran_b = [[] for _ in range(self.n_sample)]

        for i in range(len(self.seeds)):
            np.random.seed(self.seeds[i])
            self.ran_r[i] = np.random.gamma(2, 1, self.dim)
            self.ran_c[i] = np.random.gamma(2, 1, self.dim)
            self.ran_b[i] = np.random.uniform(0, 1, self.dim)

    def generate_samples(self):
        print("Generating samples...")
        self.race = np.zeros((self.dim, self.n_sample))
        for seq, weight in zip(self.idxs, self.weights):
            seq = seq.numpy()
            weight = weight.numpy()
            hashValues = self.hashing(seq, weight)

            for i in range(self.n_sample):
                self.race[int(hashValues[i]), i] += 1

    def hashing(self, seq, weight):
        hashValues = np.zeros(self.n_sample)
        for i in range(self.n_sample):
            t = np.floor((np.log(weight) / self.ran_r[i][seq]) + self.ran_b[i][seq])
            y = np.exp(self.ran_r[i][seq] * (t - self.ran_b[i][seq]))
            a = self.ran_b[i][seq] / (y * np.exp(self.ran_r[i][seq]))

            k = seq[np.nanargmin(a)] # hash value
            hashValues[i] = k
        return hashValues


    def scoring(self):
        print("Scoring...")
        self.weighted_array = np.zeros(self.count)
        for i in range(self.count):
            seq = self.idxs[i].numpy()
            weight = self.weights[i].numpy()
            hashValues = self.hashing(seq, weight)
            race_cnt = 0
            for j in range(self.n_sample):
                race_cnt += self.race[int(hashValues[j]), j]
            assert race_cnt > 0
            self.weighted_array[i] = self.n_sample / race_cnt
    
             
if __name__ == "__main__":
    np.random.seed(42)
    seeds = np.random.randint(0, 10, 100)
    
    # Loading the data
    data_cell_type_23 = sc.read_h5ad("data/split_train/cell_type_L5_ET.h5ad")

    X = data_cell_type_23.X  # Sparse Matrix
    filtered_data = data_cell_type_23[data_cell_type_23.obs['Overall AD neuropathological Change'].isin(['Not AD', 'High'])]
    X_filtered = filtered_data.X
    y_filtered = filtered_data.obs['Overall AD neuropathological Change']
    label_mapping = {'Not AD': 0, 'High': 1}
    y_encoded = y_filtered.map(label_mapping)
    
    dataset = zero_ICWS(X_filtered, y_encoded, 36601, 100, seeds)
    dataset_weight_tensor = torch.tensor(dataset.weighted_array).type('torch.DoubleTensor')
    sampler = WeightedRandomSampler(dataset_weight_tensor, len(dataset_weight_tensor))
    val_loader = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=4)
                
                