import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Reference: Su et al., *RoFormer: Enhanced Transformer with Rotary Position
    Embedding*, 2021 (arXiv:2104.09864).

    Unlike additive positional embeddings, RoPE rotates query/key vectors
    inside attention by a position-dependent angle, so it has to be applied
    on Q and K *before* the dot product, not on the token embedding. Typical
    usage inside a custom attention block::

        rope = RotaryPositionalEmbedding(head_dim, max_seq_len=2048)
        q, k = rope(q), rope(k)        # both: (B, H, L, head_dim)
        attn = scaled_dot_product_attention(q, k, v, ...)

    Because nn.MultiheadAttention doesn't expose Q/K, RoPE can't be dropped
    into the default AutoNMT ``Transformer`` as-is — subclass it and replace
    the attention with one that calls this module on Q and K.

    The "rotate-half" convention is used (LLaMA / HuggingFace), where pairs
    are formed by splitting the last dim in halves rather than interleaving.
    """

    def __init__(self, head_dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires an even head_dim"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)                       # (max_seq_len, head_dim/2)
        # Duplicate so cos/sin line up with the rotate-half layout below.
        cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1)    # (max_seq_len, head_dim)
        sin = torch.cat((freqs.sin(), freqs.sin()), dim=-1)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim

    @staticmethod
    def _rotate_half(x):
        # (x1, x2) -> (-x2, x1) on the last dim halves.
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, positions=None):
        """Apply rotary embedding to ``x``.

        Args:
            x: tensor whose last two dims are ``(seq_len, head_dim)``.
            positions: optional 1-D tensor of absolute positions. Defaults to
                ``arange(seq_len)``. Useful for incremental decoding, where
                each call sees a single new token at its absolute position.
        """
        seq_len = x.shape[-2]
        if positions is None:
            cos = self.cos[:seq_len]
            sin = self.sin[:seq_len]
        else:
            cos = self.cos[positions]
            sin = self.sin[positions]
        return x * cos + self._rotate_half(x) * sin
