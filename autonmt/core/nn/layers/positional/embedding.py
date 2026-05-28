import torch.nn as nn

from autonmt.core.nn.layers.positional.learned import LearnedPositionalEmbedding
from autonmt.core.nn.layers.positional.sinusoidal import SinusoidalPositionalEmbedding


class PositionalEmbedding(nn.Module):
    """Thin dispatcher that picks between learned and sinusoidal absolute PE."""

    def __init__(self, num_embeddings, embedding_dim, padding_idx, learned):
        super().__init__()
        cls = LearnedPositionalEmbedding if learned else SinusoidalPositionalEmbedding
        self.pos_emb = cls(num_embeddings, embedding_dim, padding_idx)

    def forward(self, x):
        return self.pos_emb(x)
