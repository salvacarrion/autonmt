import math

import torch.nn as nn

from autonmt.core.nn.layers import (
    IncrementalTransformerDecoder,
    IncrementalTransformerDecoderLayer,
    PositionalEmbedding,
    pos_embedding_at,
)
from autonmt.core.nn.seq2seq import LitSeq2Seq


class Transformer(LitSeq2Seq):
    """Vanilla Transformer encoder-decoder.

    Stack of multi-head self-attention + feed-forward blocks on both sides;
    decoder adds masked self-attention and cross-attention to the encoder.
    Fully parallel during training, KV-cached during incremental decoding.

    References
    ----------
    Vaswani et al. (2017). *Attention Is All You Need.*
    [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
    """

    # Search algorithms can pass ``incremental_state={}`` to enable KV-cached
    # decoding; the model handles the contract change in ``forward_decoder``.
    supports_incremental_decoding = True

    def __init__(self,
                 src_vocab_size, tgt_vocab_size,
                 encoder_embed_dim=256,
                 decoder_embed_dim=256,
                 encoder_layers=3,
                 decoder_layers=3,
                 encoder_attention_heads=8,
                 decoder_attention_heads=8,
                 encoder_ffn_embed_dim=512,
                 decoder_ffn_embed_dim=512,
                 dropout=0.1,
                 activation_fn="relu",
                 max_src_positions=1024,
                 max_tgt_positions=1024,
                 padding_idx=None,
                 learned=False,
                 tie_embeddings=False,
                 norm_first=False,
                 **kwargs):
        """Build a Transformer encoder-decoder.

        Parameters
        ----------
        src_vocab_size, tgt_vocab_size : int
            Source / target vocabulary sizes (usually inferred from the vocabs
            via :meth:`~autonmt.core.nn.seq2seq.LitSeq2Seq.from_vocabs`).
        encoder_embed_dim, decoder_embed_dim : int, default 256
            Model (embedding) dimension on each side.
        encoder_layers, decoder_layers : int, default 3
            Number of stacked encoder / decoder blocks.
        encoder_attention_heads, decoder_attention_heads : int, default 8
            Number of attention heads per block.
        encoder_ffn_embed_dim, decoder_ffn_embed_dim : int, default 512
            Inner dimension of the position-wise feed-forward sublayer.
        dropout : float, default 0.1
            Dropout probability applied throughout.
        activation_fn : str, default "relu"
            Feed-forward activation (``"relu"`` or ``"gelu"``).
        max_src_positions, max_tgt_positions : int, default 1024
            Maximum sequence lengths supported by the positional embeddings.
        padding_idx : int, optional
            Padding token id (masked in attention and zeroed in the PE).
        learned : bool, default False
            Use learned absolute positional embeddings instead of sinusoidal.
        tie_embeddings : bool, default False
            Share weights between the target embedding and the output projection.
        norm_first : bool, default False
            Pre-norm (norm before each sublayer) instead of post-norm.
        """
        super().__init__(src_vocab_size, tgt_vocab_size, padding_idx, architecture="transformer", **kwargs)
        self.max_src_positions = max_src_positions
        self.max_tgt_positions = max_tgt_positions

        # Model
        self.src_embeddings = nn.Embedding(src_vocab_size, encoder_embed_dim)
        self.tgt_embeddings = nn.Embedding(tgt_vocab_size, decoder_embed_dim)
        # Vaswani et al. §3.4: token embeddings are scaled by sqrt(d_model) before
        # adding positional encodings so the two have comparable magnitudes.
        self.src_embed_scale = math.sqrt(encoder_embed_dim)
        self.tgt_embed_scale = math.sqrt(decoder_embed_dim)
        self.src_pos_embeddings = PositionalEmbedding(num_embeddings=max_src_positions, embedding_dim=encoder_embed_dim, padding_idx=padding_idx, learned=learned)
        self.tgt_pos_embeddings = PositionalEmbedding(num_embeddings=max_tgt_positions, embedding_dim=decoder_embed_dim, padding_idx=padding_idx, learned=learned)
        # Swap PyTorch's nn.TransformerDecoder for our KV-cache-aware one.
        # Parameter layout is identical → existing checkpoints load unchanged.
        decoder_layer = IncrementalTransformerDecoderLayer(
            d_model=decoder_embed_dim,
            nhead=decoder_attention_heads,
            dim_feedforward=decoder_ffn_embed_dim,
            dropout=dropout,
            activation=activation_fn,
            norm_first=norm_first,
        )
        # Pre-LN convention: add a final LayerNorm at the end of the decoder stack
        # (mirrors nn.Transformer's own pre-LN handling for its built-in decoder).
        decoder_norm = nn.LayerNorm(decoder_embed_dim) if norm_first else None
        custom_decoder = IncrementalTransformerDecoder(decoder_layer, num_layers=decoder_layers, norm=decoder_norm)
        # nn.Transformer's norm_first only configures its built-in encoder (we pass
        # custom_decoder, so its decoder code path is skipped) — that's exactly
        # what we need: the encoder gets Pre/Post-LN automatically, and we control
        # the decoder ourselves above.
        self.transformer = nn.Transformer(d_model=encoder_embed_dim,
                                          nhead=encoder_attention_heads,
                                          num_encoder_layers=encoder_layers,
                                          num_decoder_layers=decoder_layers,
                                          dim_feedforward=encoder_ffn_embed_dim,
                                          dropout=dropout,
                                          activation=activation_fn,
                                          norm_first=norm_first,
                                          custom_decoder=custom_decoder)
        self.output_layer = nn.Linear(encoder_embed_dim, tgt_vocab_size)
        self.input_dropout = nn.Dropout(dropout)

        # Checks
        assert encoder_embed_dim == decoder_embed_dim
        assert encoder_attention_heads == decoder_attention_heads
        assert encoder_ffn_embed_dim == decoder_ffn_embed_dim

        # Weight tying (Press & Wolf 2017): share the decoder input embedding
        # with the output projection. Cuts parameters and is standard in NMT.
        if tie_embeddings:
            self.output_layer.weight = self.tgt_embeddings.weight

    def forward_encoder(self, x, x_len, **kwargs):
        assert x.shape[1] <= self.max_src_positions

        # Build the source padding mask once and propagate it via the state
        # tuple. Without it both encoder self-attention and decoder cross-
        # attention compute over <pad> positions — 30–50% of attention work
        # on typical batches, and a quiet quality bug (representations get
        # contaminated by padding).
        src_key_padding_mask = (x == self.padding_idx)  # (B, L_src), True=pad

        x_pos = self.src_pos_embeddings(x)
        x_emb = self.src_embeddings(x) * self.src_embed_scale
        x_emb = (x_emb + x_pos).transpose(0, 1)

        memory = self.transformer.encoder(src=x_emb, mask=None,
                                          src_key_padding_mask=src_key_padding_mask)
        return None, (memory, src_key_padding_mask)

    @staticmethod
    def _unpack_state(states):
        # Tolerate older callers (e.g. checkpoints / external code) that still
        # pass a bare memory tensor as state.
        if isinstance(states, tuple) and len(states) == 2:
            return states
        return states, None

    def forward_decoder(self, y, y_len, states, incremental_state=None, **kwargs):
        if incremental_state is None:
            return self._forward_decoder_parallel(y, states)
        return self._forward_decoder_incremental(y, states, incremental_state)

    def _forward_decoder_parallel(self, y, states):
        assert y.shape[1] <= self.max_tgt_positions
        memory, src_key_padding_mask = self._unpack_state(states)

        y_pos = self.tgt_pos_embeddings(y)
        y_emb = self.tgt_embeddings(y) * self.tgt_embed_scale
        y_emb = (y_emb + y_pos).transpose(0, 1)

        tgt_mask = self.transformer.generate_square_subsequent_mask(y_emb.shape[0]).to(y_emb.device)

        output = self.transformer.decoder(tgt=y_emb, memory=memory, tgt_mask=tgt_mask, memory_mask=None,
                                          tgt_key_padding_mask=None,
                                          memory_key_padding_mask=src_key_padding_mask)

        output = output.transpose(0, 1)
        output = self.output_layer(output)
        return output, states

    def _forward_decoder_incremental(self, y, states, incremental_state):
        """One-token-per-step decoding with KV cache. ``y`` is (B, 1)."""
        assert y.shape[1] == 1, "incremental mode expects a single new token per call"
        memory, src_key_padding_mask = self._unpack_state(states)

        # ``step`` is the absolute position of the new token (0 = first after <sos>
        # if the search seeds the cache with <sos> at call 1).
        step = incremental_state.setdefault("step", 0)

        # Hand-roll the positional embedding for this absolute position: the
        # standard forward() expects a sequence and applies padding masking.
        y_pos = pos_embedding_at(self.tgt_pos_embeddings, step, y.device)   # (D,)
        y_emb = self.tgt_embeddings(y).squeeze(1) * self.tgt_embed_scale    # (B, D)
        y_emb = y_emb + y_pos.unsqueeze(0)
        y_emb = y_emb.unsqueeze(0)                                          # (1, B, D)

        # No causal tgt_mask: Q is just the new position, K/V from the cache.
        output = self.transformer.decoder(tgt=y_emb, memory=memory,
                                          memory_key_padding_mask=src_key_padding_mask,
                                          incremental_state=incremental_state)
        output = output.transpose(0, 1)                                     # (B, 1, D)
        output = self.output_layer(output)                                  # (B, 1, V)
        incremental_state["step"] = step + 1
        return output, states

    def forward_enc_dec(self, x, x_len, y, y_len, **kwargs):
        _, states = self.forward_encoder(x, x_len, **kwargs)
        output, _ = self.forward_decoder(y, y_len, states, **kwargs)
        return output
