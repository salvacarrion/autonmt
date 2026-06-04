"""BERT-style encoder-only Transformer for masked language modelling.

A **bidirectional** Transformer encoder (full self-attention, no causal mask)
with a masked-LM head tied to the token embeddings. It reuses PyTorch's
well-tested ``nn.TransformerEncoder`` — the same bidirectional stack the
encoder side of the built-in :class:`~autonmt.core.nn.models.transformer.Transformer`
already uses — so the only genuinely new piece on the encoder-only path is the
training objective, not the attention.

Implements :class:`~autonmt.core.nn.mlm.LitMLM`'s single abstract method,
:meth:`forward`, returning per-position vocabulary logits.

References
----------
Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding.* [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

Vaswani et al. (2017). *Attention Is All You Need.* (the Transformer encoder)
[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
"""
import torch
import torch.nn as nn

from autonmt.core.nn.mlm import LitMLM


class MLMTransformer(LitMLM):
    """Encoder-only masked language model.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size (usually inferred via :meth:`LitMLM.from_corpus`). Must
        include the reserved ``<mask>`` piece.
    padding_idx : int, optional
        Padding / ignore id. Used as the criterion's ``ignore_index`` (the label
        value MLMDataset writes at non-masked positions) and to zero its embedding.
    embed_dim : int, default 256
        Model (embedding) dimension.
    num_layers : int, default 4
        Number of stacked encoder blocks.
    num_heads : int, default 8
        Attention heads per block.
    ffn_dim : int, optional
        Feed-forward inner dimension. Defaults to ``4 * embed_dim``.
    dropout : float, default 0.1
        Dropout probability throughout.
    max_seq_len : int, default 1024
        Maximum sequence length (bounds the learned positional table).
    block_size : int, optional
        Training context length (informational; must be ``<= max_seq_len``).
    activation : str, default "gelu"
        Feed-forward activation (``"gelu"`` as in BERT, or ``"relu"``).
    tie_embeddings : bool, default True
        Share the token-embedding weight with the MLM output projection.
    norm_first : bool, default False
        Pre-LN (norm before each sub-block) vs Post-LN (BERT's original).
    """

    def __init__(self, vocab_size, padding_idx=None, embed_dim=256, num_layers=4,
                 num_heads=8, ffn_dim=None, dropout=0.1, max_seq_len=1024,
                 block_size=None, activation="gelu", tie_embeddings=True,
                 norm_first=False, **kwargs):
        super().__init__(vocab_size=vocab_size, padding_idx=padding_idx,
                         block_size=block_size or max_seq_len, architecture="mlm-transformer", **kwargs)
        ffn_dim = ffn_dim or 4 * embed_dim
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.tok_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        # Learned absolute positions, indexed directly by arange(L) (no reserved
        # pad slot) — bidirectional, so no incremental position bookkeeping.
        self.pos_embeddings = nn.Embedding(max_seq_len, embed_dim)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation=activation, batch_first=True, norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, vocab_size)

        # Weight tying (Press & Wolf 2017): share input embedding with the MLM head.
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

    def forward(self, x, attention_mask=None):
        B, L = x.shape
        assert L <= self.max_seq_len, (
            f"sequence length {L} exceeds max_seq_len {self.max_seq_len}"
        )

        positions = torch.arange(L, device=x.device)
        h = self.tok_embeddings(x) + self.pos_embeddings(positions).unsqueeze(0)
        h = self.input_dropout(h)

        # nn.TransformerEncoder expects a key-padding mask where True = ignore.
        # ``attention_mask`` (when given) follows the True=keep convention, so invert.
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        return self.output_layer(h)
