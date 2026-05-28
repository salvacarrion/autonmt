"""Drop-in replacement for ``nn.TransformerDecoder`` that supports incremental
decoding via an external cache dict.

Module structure mirrors ``nn.TransformerDecoder`` so state_dicts transfer
cleanly (param names like ``layers.0.self_attn.in_proj_weight`` are identical).
Parallel mode (``incremental_state=None``) delegates to ``nn.MultiheadAttention``
unchanged → bit-exact compatible with PyTorch's stock decoder for training.
Incremental mode reuses the same projection weights but manages K/V manually
so each decode step costs O(L) instead of O(L^2).
"""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from autonmt.core.nn.layers.positional.learned import LearnedPositionalEmbedding
from autonmt.core.nn.layers.positional.sinusoidal import SinusoidalPositionalEmbedding


def _resolve_activation(name):
    if not isinstance(name, str):
        return name
    return {"relu": F.relu, "gelu": F.gelu}[name.lower()]


def pos_embedding_at(pos_emb_module, step, device):
    """Look up the positional embedding for a single absolute position.

    The standard ``PositionalEmbedding.forward`` consumes a full token tensor
    and applies padding masking, which isn't useful in incremental decoding
    where we feed one known-non-pad token per step. This bypasses that path
    and indexes the underlying table directly.
    """
    backend = pos_emb_module.pos_emb
    if isinstance(backend, SinusoidalPositionalEmbedding):
        return backend.emb[step]
    if isinstance(backend, LearnedPositionalEmbedding):
        # Learned embeddings reserve index 0 for <pad>; first real position is 1.
        return backend.weight[step + 1]
    raise TypeError(f"Unsupported positional embedding backend: {type(backend)}")


class IncrementalTransformerDecoderLayer(nn.Module):
    """One decoder layer with optional KV-cache support."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", norm_first=False):
        super().__init__()
        # Submodule names match nn.TransformerDecoderLayer for state_dict parity.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _resolve_activation(activation)
        self.norm_first = norm_first

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_is_causal=None, memory_is_causal=False,
                incremental_state=None):
        # Pick block fns (parallel vs incremental) and run with the chosen
        # norm ordering. Post-LN is identical to nn.TransformerDecoderLayer.
        if incremental_state is None:
            sa = lambda h: self._sa_block(h, tgt_mask, tgt_key_padding_mask)  # noqa: E731
            mha = lambda h: self._mha_block(h, memory, memory_mask, memory_key_padding_mask)  # noqa: E731
        else:
            # Each layer owns its slice of the cache, keyed by id(self) so
            # multiple layers don't collide.
            layer_cache = incremental_state.setdefault(id(self), {})
            sa = lambda h: self._sa_block_incremental(h, layer_cache)  # noqa: E731
            mha = lambda h: self._mha_block_cached(h, memory, layer_cache, memory_key_padding_mask)  # noqa: E731

        x = tgt
        if self.norm_first:
            x = x + sa(self.norm1(x))
            x = x + mha(self.norm2(x))
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + sa(x))
            x = self.norm2(x + mha(x))
            x = self.norm3(x + self._ff_block(x))
        return x

    # ---- parallel-mode blocks (delegated to nn.MultiheadAttention) ----------

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _mha_block(self, x, memory, attn_mask, key_padding_mask):
        x = self.multihead_attn(x, memory, memory, attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout2(x)

    def _ff_block(self, x):
        return self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(x)))))

    # ---- incremental-mode blocks (manual KV cache) --------------------------

    def _sa_block_incremental(self, x, layer_cache):
        """Self-attention with growing K/V cache. ``x`` is (1, B, D)."""
        # nn.MultiheadAttention packs QKV into a single in_proj_weight of shape
        # (3D, D); we project the new token and chunk into Q, K, V.
        qkv = F.linear(x, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        q, k_new, v_new = qkv.chunk(3, dim=-1)              # each (1, B, D)

        if "self_k" in layer_cache:
            k = torch.cat([layer_cache["self_k"], k_new], dim=0)  # (T, B, D)
            v = torch.cat([layer_cache["self_v"], v_new], dim=0)
        else:
            k, v = k_new, v_new
        layer_cache["self_k"] = k
        layer_cache["self_v"] = v

        attn = self._scaled_dp(q, k, v, attn_mask=None)
        out = self.self_attn.out_proj(attn)
        return self.dropout1(out)

    def _mha_block_cached(self, x, memory, layer_cache, memory_key_padding_mask):
        """Cross-attention with encoder K/V cached on first call."""
        D = self.d_model
        in_proj_weight = self.multihead_attn.in_proj_weight
        in_proj_bias = self.multihead_attn.in_proj_bias
        # nn.MultiheadAttention encoder-decoder shape: W[:D] projects Q;
        # W[D:] projects the KV stream packed together.
        q = F.linear(x, in_proj_weight[:D], in_proj_bias[:D])           # (1, B, D)

        if "cross_k" not in layer_cache:
            kv = F.linear(memory, in_proj_weight[D:], in_proj_bias[D:])  # (L_src, B, 2D)
            k, v = kv.chunk(2, dim=-1)
            layer_cache["cross_k"] = k
            layer_cache["cross_v"] = v
        k = layer_cache["cross_k"]
        v = layer_cache["cross_v"]

        attn_mask = None
        if memory_key_padding_mask is not None:
            # (B, L_src) bool → additive (B, 1, 1, L_src) mask with -inf where pad.
            B, L_src = memory_key_padding_mask.shape
            attn_mask = torch.zeros(B, 1, 1, L_src, dtype=q.dtype, device=q.device)
            attn_mask.masked_fill_(memory_key_padding_mask.view(B, 1, 1, L_src), float("-inf"))

        attn = self._scaled_dp(q, k, v, attn_mask=attn_mask)
        out = self.multihead_attn.out_proj(attn)
        return self.dropout2(out)

    def _scaled_dp(self, q, k, v, attn_mask):
        """Multi-head SDPA on (L, B, D) inputs. Returns (L_q, B, D).

        Q has length 1 in incremental mode (only the new token). K/V have the
        cached length (T for self-attn, L_src for cross-attn). Causality holds
        by construction — Q at position T attends to all of K at positions 0..T.
        """
        L_q, B = q.shape[0], q.shape[1]
        L_kv = k.shape[0]

        def to_mh(t, L):
            return t.view(L, B, self.nhead, self.head_dim).permute(1, 2, 0, 3).contiguous()

        q_mh = to_mh(q, L_q)        # (B, H, L_q, hd)
        k_mh = to_mh(k, L_kv)
        v_mh = to_mh(v, L_kv)

        attn = F.scaled_dot_product_attention(q_mh, k_mh, v_mh, attn_mask=attn_mask)
        # (B, H, L_q, hd) -> (L_q, B, D)
        return attn.permute(2, 0, 1, 3).reshape(L_q, B, self.d_model)


class IncrementalTransformerDecoder(nn.Module):
    """Stack of :class:`IncrementalTransformerDecoderLayer`. Mirrors the layout
    of ``nn.TransformerDecoder`` (``layers`` + optional ``norm``)."""

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_is_causal=None, memory_is_causal=False,
                incremental_state=None):
        output = tgt
        for layer in self.layers:
            output = layer(
                output, memory,
                tgt_mask=tgt_mask, memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                incremental_state=incremental_state,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output
