import math

import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings.

    The fixed sin/cos position table from the original Transformer (§3.5).
    Padding positions are zeroed out (fairseq convention) — redundant with a
    proper key-padding mask in attention, but kept for parity with the
    learned variant.

    References
    ----------
    Vaswani et al. (2017). *Attention Is All You Need.*
    [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ori_padding_idx = padding_idx

        half_dim = embedding_dim // 2
        scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, dtype=torch.float) * -scale)
        angles = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1).view(num_embeddings, -1)
        # Registered as buffer so it moves with .to(device) and ends up in state_dict
        # only when explicitly asked (persistent=False keeps checkpoints lean since
        # the table is fully determined by the constructor args).
        self.register_buffer("emb", emb, persistent=False)

    def forward(self, x):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = x.shape
        mask = x.ne(self.ori_padding_idx).int().unsqueeze(2)  # 1 1 1 1 0 0 0
        pos = self.emb[:seq_len, :].unsqueeze(0).expand(bsz, -1, -1)
        return pos * mask
