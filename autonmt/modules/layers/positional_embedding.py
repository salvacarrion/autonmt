from autonmt.modules.layers.learned_positional_embedding import LearnedPositionalEmbedding
from autonmt.modules.layers.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, learned):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
    else:
        m = SinusoidalPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
    return m
