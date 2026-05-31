import torch.nn as nn

from autonmt.core.nn.layers.positional.learned import LearnedPositionalEmbedding
from autonmt.core.nn.layers.positional.sinusoidal import SinusoidalPositionalEmbedding


class PositionalEmbedding(nn.Module):
    """Absolute positional embedding dispatcher.

    Picks :class:`LearnedPositionalEmbedding` when ``learned=True`` and
    :class:`SinusoidalPositionalEmbedding` otherwise, adding the chosen
    positional signal to the token embeddings. This is what the built-in
    ``Transformer`` uses; for the rotary alternative see
    :class:`RotaryPositionalEmbedding`.
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx, learned):
        super().__init__()
        cls = LearnedPositionalEmbedding if learned else SinusoidalPositionalEmbedding
        self.pos_emb = cls(num_embeddings, embedding_dim, padding_idx)

    def forward(self, x):
        return self.pos_emb(x)
