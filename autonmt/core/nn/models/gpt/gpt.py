"""nanoGPT-style decoder-only Transformer.

A pre-norm stack of (causal self-attention + SwiGLU) blocks with RMSNorm and
rotary position embeddings — the modern default — reusing the AutoNMT layer
primitives (:class:`RMSNorm`, :class:`SwiGLU`, :class:`RotaryPositionalEmbedding`)
and the new :class:`CausalSelfAttention`. Built for research iteration, not
production scale.

Implements :class:`~autonmt.core.nn.lm.LitLM`'s single abstract method,
:meth:`forward`, with both a parallel (training) and a KV-cached (generation)
path selected by ``incremental_state``.
"""
import math

import torch
import torch.nn as nn

from autonmt.core.nn.layers import RMSNorm, SwiGLU
from autonmt.core.nn.layers.transformer.causal_self_attention import CausalSelfAttention
from autonmt.core.nn.lm import LitLM


def _default_ffn_dim(embed_dim, multiple_of=32):
    """SwiGLU hidden dim ~ 2/3 * 4 * d, rounded to a convenient multiple
    (keeps parameter count comparable to a 4x ReLU FFN)."""
    hidden = int(2 / 3 * 4 * embed_dim)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


class GPTBlock(nn.Module):
    """Pre-norm Transformer block: RMSNorm → causal self-attn, RMSNorm → SwiGLU."""

    def __init__(self, embed_dim, num_heads, ffn_dim, dropout, max_seq_len,
                 use_rope, norm_eps):
        super().__init__()
        self.attn_norm = RMSNorm(embed_dim, eps=norm_eps)
        self.attn = CausalSelfAttention(
            embed_dim, num_heads, dropout=dropout, bias=False,
            use_rope=use_rope, max_seq_len=max_seq_len,
        )
        self.ffn_norm = RMSNorm(embed_dim, eps=norm_eps)
        self.ffn = SwiGLU(embed_dim, ffn_dim, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, incremental_state=None):
        x = x + self.resid_dropout(self.attn(self.attn_norm(x), incremental_state=incremental_state))
        x = x + self.resid_dropout(self.ffn(self.ffn_norm(x)))
        return x


class GPT(LitLM):
    """Decoder-only Transformer language model.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size (usually inferred via :meth:`LitLM.from_corpus`).
    padding_idx : int, optional
        Padding / ignore id. Used as the criterion's ``ignore_index`` and to
        zero the corresponding embedding row.
    embed_dim : int, default 256
        Model (embedding) dimension.
    num_layers : int, default 4
        Number of stacked decoder blocks.
    num_heads : int, default 8
        Attention heads per block.
    ffn_dim : int, optional
        SwiGLU hidden dimension. Defaults to ``~2/3 * 4 * embed_dim`` rounded.
    dropout : float, default 0.1
        Dropout probability throughout.
    max_seq_len : int, default 1024
        Maximum context length (bounds RoPE / learned positions).
    block_size : int, optional
        Training context length (informational; must be ``<= max_seq_len``).
    use_rope : bool, default True
        Rotary position embeddings. When False, learned absolute positions.
    tie_embeddings : bool, default True
        Share the token-embedding weight with the output projection.
    norm_eps : float, default 1e-6
        RMSNorm epsilon.

    References
    ----------
    Radford et al. (2019). *Language Models are Unsupervised Multitask Learners.*
    (the GPT-2 decoder-only architecture)
    [OpenAI PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

    Touvron et al. (2023). *LLaMA: Open and Efficient Foundation Language Models.*
    (the RoPE + RMSNorm + SwiGLU pre-norm recipe used by default here)
    [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)

    Press & Wolf (2017). *Using the Output Embedding to Improve Language Models.*
    (weight tying between input embedding and output projection)
    [arXiv:1608.05859](https://arxiv.org/abs/1608.05859)
    """

    # Parity with the seq2seq models' decoding flag. ``lm_generate`` manages the
    # KV cache itself, but this advertises the capability.
    supports_incremental_decoding = True

    def __init__(self, vocab_size, padding_idx=None, embed_dim=256, num_layers=4,
                 num_heads=8, ffn_dim=None, dropout=0.1, max_seq_len=1024,
                 block_size=None, use_rope=True, tie_embeddings=True, norm_eps=1e-6,
                 **kwargs):
        super().__init__(vocab_size=vocab_size, padding_idx=padding_idx,
                         block_size=block_size or max_seq_len, architecture="gpt", **kwargs)
        ffn_dim = ffn_dim or _default_ffn_dim(embed_dim)
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope

        self.tok_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        # Learned absolute positions only when RoPE is disabled.
        self.pos_embeddings = None if use_rope else nn.Embedding(max_seq_len, embed_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            GPTBlock(embed_dim, num_heads, ffn_dim, dropout, max_seq_len, use_rope, norm_eps)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(embed_dim, eps=norm_eps)
        self.output_layer = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying (Press & Wolf 2017): share input embedding and output head.
        if tie_embeddings:
            self.output_layer.weight = self.tok_embeddings.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(self, x, incremental_state=None):
        B, L = x.shape
        assert L <= self.max_seq_len, (
            f"sequence length {L} exceeds max_seq_len {self.max_seq_len}"
        )

        h = self.tok_embeddings(x) * math.sqrt(self.embed_dim)
        if self.pos_embeddings is not None:
            past = 0
            if incremental_state is not None:
                past = incremental_state.get("_abs_pos", 0)
                incremental_state["_abs_pos"] = past + L
            positions = torch.arange(past, past + L, device=x.device)
            h = h + self.pos_embeddings(positions).unsqueeze(0)
        h = self.input_dropout(h)

        for block in self.blocks:
            h = block(h, incremental_state=incremental_state)

        h = self.final_norm(h)
        return self.output_layer(h)
