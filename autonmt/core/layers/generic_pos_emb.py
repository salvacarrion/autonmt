import torch.nn as nn

from autonmt.core.layers.learned_pos_emb import LearnedPositionalEmbedding
from autonmt.core.layers.sinusoidal_pos_emb import SinusoidalPositionalEmbedding


class PositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, learned):
        super().__init__()
        cls = LearnedPositionalEmbedding if learned else SinusoidalPositionalEmbedding
        self.pos_emb = cls(num_embeddings, embedding_dim, padding_idx)

    def forward(self, x):
        return self.pos_emb(x)

