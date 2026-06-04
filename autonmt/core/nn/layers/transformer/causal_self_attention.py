"""Causal (decoder-only) multi-head self-attention with optional KV cache.

This is the one genuinely new modelling primitive the LM path needs: the v1.0
:class:`IncrementalTransformerDecoderLayer` couples self-attention to encoder
cross-attention, which a decoder-only GPT doesn't have. The KV-cache approach
mirrors that layer's ``_sa_block_incremental`` / ``_scaled_dp`` so behaviour
stays consistent across the codebase, but here it's self-contained and uses the
``(batch, length, dim)`` convention (cleaner for a standalone model; no
``nn.Transformer`` state-dict parity to preserve).

Two paths, selected by ``incremental_state``:
  * ``None`` — parallel training: one causal :func:`scaled_dot_product_attention`
    over the whole sequence.
  * a dict — incremental decoding: each call appends the new K/V to a per-layer
    cache and attends to all cached positions (prefill handles ``L > 1`` causally,
    subsequent steps feed one token).

Rotary position embeddings (RoPE) are applied to Q/K *inside* attention when
``use_rope=True``; this is why RoPE can't be dropped into the default
``Transformer`` (which uses ``nn.MultiheadAttention`` and never exposes Q/K).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from autonmt.core.nn.layers.positional.rotary import RotaryPositionalEmbedding


class CausalSelfAttention(nn.Module):
    """Decoder-only multi-head self-attention.

    Parameters
    ----------
    d_model : int
        Model dimension. Must be divisible by ``nhead``.
    nhead : int
        Number of attention heads.
    dropout : float, default 0.0
        Attention dropout probability (applied only during training).
    bias : bool, default False
        Whether the QKV / output projections have a bias term.
    use_rope : bool, default True
        Apply rotary position embeddings to Q/K.
    max_seq_len : int, default 2048
        Maximum sequence length supported by the RoPE tables.
    rope_base : float, default 10000.0
        RoPE frequency base.

    References
    ----------
    Vaswani et al. (2017). *Attention Is All You Need.* (scaled dot-product
    multi-head attention) [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

    Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position
    Embedding.* [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
    """

    def __init__(self, d_model, nhead, dropout=0.0, bias=False,
                 use_rope=True, max_seq_len=2048, rope_base=10000.0):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.rope = (RotaryPositionalEmbedding(self.head_dim, max_seq_len=max_seq_len, base=rope_base)
                     if use_rope else None)

    def _split_heads(self, t, B, L):
        # (B, L, D) -> (B, H, L, head_dim)
        return t.view(B, L, self.nhead, self.head_dim).transpose(1, 2)

    def forward(self, x, incremental_state=None):
        """``x`` is ``(B, L, D)``. Returns ``(B, L, D)``."""
        B, L, _ = x.shape
        q, k, v = self.qkv(x).split(self.d_model, dim=-1)
        q = self._split_heads(q, B, L)
        k = self._split_heads(k, B, L)
        v = self._split_heads(v, B, L)

        if incremental_state is None:
            if self.rope is not None:
                q, k = self.rope(q), self.rope(k)         # positions = arange(L)
            attn = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            attn = self._forward_incremental(q, k, v, incremental_state, B, L, x.device, x.dtype)

        # (B, H, L, head_dim) -> (B, L, D)
        attn = attn.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(attn)

    def _forward_incremental(self, q, k, v, incremental_state, B, L, device, dtype):
        # Each layer owns its slice of the cache, keyed by id(self) (matches the
        # convention in IncrementalTransformerDecoderLayer).
        cache = incremental_state.setdefault(id(self), {})
        past_len = cache["k"].shape[2] if "k" in cache else 0

        if self.rope is not None:
            positions = torch.arange(past_len, past_len + L, device=device)
            q = self.rope(q, positions=positions)
            k = self.rope(k, positions=positions)

        if "k" in cache:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
        cache["k"], cache["v"] = k, v

        # Additive mask: query at absolute position (past_len + i) may attend to
        # key j iff j <= past_len + i. This is lower-triangular for the prefill
        # (past_len=0, L>1) and "attend to everything" for a single-token step.
        L_kv = k.shape[2]
        q_pos = torch.arange(past_len, past_len + L, device=device).unsqueeze(1)  # (L, 1)
        k_pos = torch.arange(L_kv, device=device).unsqueeze(0)                    # (1, L_kv)
        attn_mask = torch.zeros(L, L_kv, dtype=dtype, device=device)
        attn_mask.masked_fill_(k_pos > q_pos, float("-inf"))
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
