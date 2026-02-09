"""Shared utilities for the gene-interact project.

Centralizes commonly used functions to avoid code duplication across modules.
"""

import math
import torch
import torch.nn.functional as F
from collections import OrderedDict


def setup_seed(seed=42):
    """Set random seeds for reproducibility across CPU and CUDA devices."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint(config, model_path, map_location=None):
    """Load a CelluFormer model from a DataParallel checkpoint.

    Handles the 'module.' prefix that DataParallel adds to state dict keys.

    Args:
        config: Dict with model hyperparameters (num_genes, embedding_dim,
                dim_feedforward, head, depth, dropout).
        model_path: Path to the saved .pth checkpoint file.
        map_location: Optional device mapping for torch.load (e.g., 'cpu').

    Returns:
        A CelluFormer model loaded with the checkpoint weights.
    """
    from transformer.celluformer import CelluFormer

    state_dict = torch.load(model_path, map_location=map_location)

    # Strip the 'module.' prefix added by DataParallel
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model = CelluFormer(
        config['num_genes'], config['embedding_dim'],
        config['dim_feedforward'], config['head'],
        config['depth'], config['dropout']
    )
    model.load_state_dict(new_state_dict)
    return model


def compute_selfattention(transformer_encoder, x, i_layer, d_model, num_heads):
    """Compute self-attention probabilities for a given encoder layer.

    Manually extracts Q, K, V from the multi-head attention in_proj weights
    and computes scaled dot-product attention probabilities.

    Args:
        transformer_encoder: The nn.TransformerEncoder module.
        x: Input tensor of shape (batch, seq_len, d_model).
        i_layer: Index of the encoder layer to inspect.
        d_model: Model embedding dimension.
        num_heads: Number of attention heads.

    Returns:
        Attention probability tensor of shape (batch, heads, seq_len, seq_len).
    """
    h = F.linear(
        x,
        transformer_encoder.layers[i_layer].self_attn.in_proj_weight,
        bias=transformer_encoder.layers[i_layer].self_attn.in_proj_bias,
    )
    qkv = h.reshape(x.shape[0], x.shape[1], num_heads, 3 * d_model // num_heads)
    qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
    q, k, v = qkv.chunk(3, dim=-1)  # [Batch, Head, SeqLen, d_head]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))  # [Batch, Head, SeqLen, SeqLen]
    d_k = q.size()[-1]
    attn_probs = attn_logits / math.sqrt(d_k)
    attn_probs = F.softmax(attn_probs, dim=-1)
    return attn_probs
