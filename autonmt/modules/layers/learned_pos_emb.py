import torch
import torch.nn as nn


class LearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        # Force the padding index to be zero to avoid problems such as: 1, 2, 3*, 4, 5, 3*, 3*, 3*
        self.ori_padding_idx = padding_idx
        super().__init__(num_embeddings, embedding_dim, padding_idx=0)

    def forward(self, x):
        # 1, 2, 3,... but 0 where padding is.
        mask = x.ne(self.ori_padding_idx).int()  # 1 1 1 1 0 0 0
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long()  # 1 2 3 4 0 0 0
        return super().forward(positions)
