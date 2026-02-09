"""Dataset for variable-length sparse gene expression sequences."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm


class VariableLengthSequenceDataset(Dataset):
    """Dataset that converts sparse scRNA-seq data into padded gene sequences.

    For each cell, extracts non-zero gene indices and their expression values,
    then pads all sequences to the same length for batched processing.

    Args:
        X: Sparse matrix of shape (n_cells, n_genes) containing expression values.
        y: Array-like of integer labels for each cell.
    """

    def __init__(self, X, y):
        self.seq = []
        self.vals = []
        self.labels = torch.tensor(y)

        print("Initializing dataset...")
        for i in tqdm(range(X.shape[0]), desc="Processing data"):
            x = X[i].toarray()
            x_tensor = torch.from_numpy(x).float()
            
            non_zero_indices = torch.nonzero(x_tensor, as_tuple=True)[1]
            self.seq.append(non_zero_indices)
            
            scaling_factor = x_tensor[0][non_zero_indices]
            self.vals.append(scaling_factor)
            
        self.seq = nn.utils.rnn.pad_sequence([seq.clone().detach() for seq in self.seq], batch_first=True, padding_value=0)+1
        self.vals = nn.utils.rnn.pad_sequence([vals.clone().detach() for vals in self.vals], batch_first=True, padding_value=0)

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, index):
        """Returns (gene_indices, expression_values, label) for a single cell."""
        sequence = self.seq[index]
        scaling_factor = self.vals[index]
        label = self.labels[index]
        return sequence, scaling_factor, label
