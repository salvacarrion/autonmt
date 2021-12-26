import torch
import torch.nn as nn
import math
from PIL import Image
import matplotlib.pyplot as plt


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """
    def __init__(self, max_length, embedding_dim, padding_idx):
        super().__init__()
        self.embedding_dim = embedding_dim

        half_dim = embedding_dim // 2
        self.emb = math.log(10000) / (half_dim - 1)
        self.emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -self.emb)
        self.emb = torch.arange(max_length, dtype=torch.float).unsqueeze(1) * self.emb.unsqueeze(0)
        self.emb = torch.cat([torch.sin(self.emb), torch.cos(self.emb)], dim=1).view(max_length, -1)

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

#
# class SinusoidalPositionalEmbedding(nn.Module):
#     """This module produces sinusoidal positional embeddings of any length.
#     Padding symbols are ignored.
#     """
#
#     def __init__(self, max_len, embedding_dim, padding_idx):
#         super().__init__()
#
#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, embedding_dim)
#         pos = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / (embedding_dim/2)))
#         pe[:, 0::2] = torch.sin(pos * div_term)
#         pe[:, 1::2] = torch.cos(pos * div_term)
#
#         self.register_buffer("pe", pe.unsqueeze(0))
#         self.padding_idx = padding_idx
#
#     def forward(self, x):
#         batch_size, length = x.shape
#         mask = x.ne(self.padding_idx).int()  # 1 1 1 1 0 0 0
#         pos = self.pe[:, :x.size(1), :]
#         pos = torch.tile(pos, (batch_size, 1, 1)) * mask.unsqueeze(2)
#         return pos


def make_positions(tensor, padding_idx: int, onnx_trace: bool = False):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx