import torch
import torch.nn as nn


class LearnedPositionalEmbedding(nn.Embedding):
    """Learned absolute positional embeddings.

    A trainable lookup table indexed by position (1-based, with 0 reserved for
    padding) and added to the token embeddings. The learned counterpart to
    :class:`SinusoidalPositionalEmbedding`; capped at ``num_embeddings``
    positions, so unlike the sinusoidal table it cannot extrapolate beyond the
    sequence lengths seen during training.

    References
    ----------
    Gehring et al. (2017). *Convolutional Sequence to Sequence Learning.*
    [arXiv:1705.03122](https://arxiv.org/abs/1705.03122)
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        # Force the padding index to be zero to avoid problems such as: 1, 2, 3*, 4, 5, 3*, 3*, 3*
        self.ori_padding_idx = padding_idx
        super().__init__(num_embeddings, embedding_dim, padding_idx=0)

    def forward(self, x):
        # 1, 2, 3,... but 0 where padding is.
        mask = x.ne(self.ori_padding_idx).int()  # 1 1 1 1 0 0 0
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long()  # 1 2 3 4 0 0 0
        return super().forward(positions)
