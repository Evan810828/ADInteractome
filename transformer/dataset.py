"""Dataset for variable-length sparse gene expression sequences."""

import torch
import torch.nn as nn
import numpy as np
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


class LazyGlobalPaddedDataset(Dataset):
    """Generate the original globally padded sequences on demand from CSR data.

    Padding deliberately uses gene index 1 and expression value 0, just as
    ``VariableLengthSequenceDataset`` does after adding one to padded indices.
    The fixed length is calculated separately for each train/test split.
    """

    def __init__(self, X, rows, labels):
        if getattr(X, "format", None) != "csr":
            raise ValueError("LazyGlobalPaddedDataset requires a CSR matrix")
        self.X = X
        self.rows = np.asarray(rows, dtype=np.int64)
        self.labels = torch.as_tensor(np.asarray(labels, dtype=np.int64))
        if len(self.rows) != len(self.labels) or not len(self.rows):
            raise ValueError("rows and labels must be nonempty and have equal lengths")

        # CSR stored counts are upper bounds on nonzero values after dense
        # conversion, including explicit zeros and repeated column indices.
        # Examine largest candidates until no remaining row can exceed the
        # observed maximum; this also respects the original float32 cast.
        bounds = np.diff(X.indptr)[self.rows]
        order = np.argsort(bounds)[::-1]
        self.max_length = 0
        for position in order:
            if bounds[position] <= self.max_length:
                break
            row = torch.from_numpy(X[int(self.rows[position])].toarray()).float()
            self.max_length = max(self.max_length, int(torch.count_nonzero(row)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        dense = torch.from_numpy(self.X[int(self.rows[index])].toarray()).float()[0]
        indices = torch.nonzero(dense, as_tuple=True)[0]
        seq = torch.ones(self.max_length, dtype=torch.long)
        vals = torch.zeros(self.max_length, dtype=torch.float32)
        seq[:len(indices)] = indices + 1
        vals[:len(indices)] = dense[indices]
        return seq, vals, self.labels[index]
