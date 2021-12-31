import torch
import torch.nn as nn

from autonmt.modules.layers.learned_pos_emb import LearnedPositionalEmbedding
from autonmt.modules.layers.sinusoidal_pos_emb import SinusoidalPositionalEmbedding


class PositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, learned):
        super().__init__()
        self.learned = learned
        if learned:
            self.pos_emb = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
        else:
            self.pos_emb = SinusoidalPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)

    def forward(self, x):
        output = self.pos_emb(x)
        return output

