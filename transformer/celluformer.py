"""CelluFormer model for single-cell gene expression classification."""

import torch.nn as nn


class CelluFormer(nn.Module):
    """Transformer-based model for classifying cells from sparse gene expression data.

    The model embeds variable-length gene sequences (non-zero expressed genes),
    scales embeddings by expression values, processes them through a transformer
    encoder, and classifies via mean-pooled representations.

    Args:
        num_genes: Total number of genes in the vocabulary.
        embedding_dim: Dimension of gene embeddings.
        dim_feedforward: Hidden dimension of the transformer feedforward network.
        head: Number of attention heads.
        depth: Number of transformer encoder layers.
        dropout: Dropout probability.
    """

    def __init__(self, num_genes, embedding_dim=128, dim_feedforward=512, head=8, depth=4, dropout=.1):
        super(CelluFormer, self).__init__()
        self.emb = nn.Embedding(num_genes+1, embedding_dim, padding_idx=0)
        nn.init.uniform_(self.emb.weight, a=-1.0/num_genes, b=1.0/num_genes)
        self.emb.weight.data[0].fill_(0)
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=head, 
            batch_first=True, 
            dropout=dropout, 
            dim_feedforward=dim_feedforward
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=depth)
        
        self.output = nn.Linear(embedding_dim, 2)
        
    def forward(self, seq, vals):
        """Forward pass.

        Args:
            seq: Gene index sequences of shape (batch_size, seq_len).
                 Indices are 1-based (0 is padding).
            vals: Expression values of shape (batch_size, seq_len).

        Returns:
            Logits of shape (batch_size, 2) for binary classification.
        """
        x = self.emb(seq)
        x = x * vals.unsqueeze(2)
        x = self.encoder(x)
        x_pooled = x.mean(dim=1)

        return self.output(x_pooled)