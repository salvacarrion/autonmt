import torch
import torch.nn as nn
import math
from PIL import Image
import matplotlib.pyplot as plt


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        super().__init__()
        self.embedding_dim = embedding_dim

        half_dim = embedding_dim // 2
        self.emb = math.log(10000) / (half_dim - 1)
        self.emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -self.emb)
        self.emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * self.emb.unsqueeze(0)
        self.emb = torch.cat([torch.sin(self.emb), torch.cos(self.emb)], dim=1).view(num_embeddings, -1)

        # self.emb[padding_idx, :] = 0
        self.ori_padding_idx = padding_idx

        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    def forward(self, x):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = x.shape
        self.emb = self.emb.to(self._float_tensor)

        mask = x.ne(self.ori_padding_idx).int().unsqueeze(2)  # 1 1 1 1 0 0 0
        pos = torch.tile(self.emb[:x.size(1), :].unsqueeze(0), (bsz, 1, 1))
        return pos*mask
