import random

import torch
import torch.nn as nn

from autonmt.core.nn.seq2seq import LitSeq2Seq


class SimpleRNN(LitSeq2Seq):
    """Vanilla encoder-decoder RNN.

    The encoder consumes the source and compresses it into a final hidden
    state; the decoder is seeded with that state and generates the target
    token-by-token. No attention — the decoder sees the source only through
    the fixed-size hidden state. Supports plain RNN / LSTM / GRU as the
    underlying cell.

    Reference: Sutskever, Vinyals & Le, *Sequence to Sequence Learning with
    Neural Networks*, NeurIPS 2014 (arXiv:1409.3215).
    """

    BASE_RNNS = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}

    # ``forward_decoder`` already calls ``last_token(y)`` and threads the
    # recurrent state through ``states``, so the hidden state itself acts as
    # the per-step cache. Inherited by BahdanauRNN / LuongRNN / ContextRNN —
    # each of which also strips ``y`` to its last token before the RNN step.
    supports_incremental_decoding = True

    def __init__(self,
                 src_vocab_size, trg_vocab_size,
                 encoder_embed_dim=256,
                 decoder_embed_dim=256,
                 encoder_hidden_dim=512,
                 decoder_hidden_dim=512,
                 encoder_n_layers=2,
                 decoder_n_layers=2,
                 encoder_dropout=0.5,
                 decoder_dropout=0.5,
                 encoder_bidirectional=False,
                 decoder_bidirectional=False,
                 teacher_force_ratio=0.5,
                 padding_idx=None,
                 packed_sequence=False,
                 base_rnn="rnn",
                 **kwargs):
        super().__init__(src_vocab_size, trg_vocab_size, padding_idx, packed_sequence=packed_sequence,
                         base_rnn=base_rnn, architecture=f"{self.__class__.__name__}-{base_rnn.upper()}", **kwargs)
        self.base_rnn = base_rnn
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.encoder_n_layers = encoder_n_layers
        self.decoder_n_layers = decoder_n_layers
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.encoder_bidirectional = encoder_bidirectional
        self.decoder_bidirectional = decoder_bidirectional
        self.teacher_forcing_ratio = teacher_force_ratio

        # Model
        self.src_embeddings = nn.Embedding(src_vocab_size, encoder_embed_dim)
        self.trg_embeddings = nn.Embedding(trg_vocab_size, decoder_embed_dim)
        self.enc_dropout = nn.Dropout(encoder_dropout)
        self.dec_dropout = nn.Dropout(decoder_dropout)
        self.output_layer = nn.Linear(decoder_hidden_dim, trg_vocab_size)

        # RNN
        base_rnn_cls = self.resolve_base_rnn(self.base_rnn)
        self.encoder_rnn = base_rnn_cls(input_size=self.encoder_embed_dim,
                                        hidden_size=self.encoder_hidden_dim,
                                        num_layers=self.encoder_n_layers,
                                        dropout=self.encoder_dropout,
                                        bidirectional=self.encoder_bidirectional, batch_first=True)
        self.decoder_rnn = base_rnn_cls(input_size=self.decoder_embed_dim,
                                        hidden_size=self.decoder_hidden_dim,
                                        num_layers=self.decoder_n_layers,
                                        dropout=self.decoder_dropout,
                                        bidirectional=self.decoder_bidirectional, batch_first=True)

        # Checks
        assert encoder_embed_dim == decoder_embed_dim
        assert encoder_hidden_dim == decoder_hidden_dim
        assert encoder_n_layers == decoder_n_layers

    @classmethod
    def resolve_base_rnn(cls, base_rnn):
        """Map a string like ``"lstm"`` to the corresponding ``nn.*`` class.

        Override in a subclass to add custom cells (e.g. ``LayerNormLSTM``)
        without touching the base registry.
        """
        key = base_rnn.lower().strip()
        if key not in cls.BASE_RNNS:
            raise ValueError(f"Unknown base_rnn '{base_rnn}'. Expected one of: {sorted(cls.BASE_RNNS)}")
        return cls.BASE_RNNS[key]

    @staticmethod
    def last_token(y):
        """Normalise ``y`` to ``(batch, 1)`` holding only the most recent token.

        Decoder helpers receive either ``(batch,)``, ``(batch, 1)`` or the full
        ``(batch, t)`` history during teacher-forcing. The RNN cells only need
        the last step.
        """
        if y.dim() == 1:
            return y.unsqueeze(1)
        if y.dim() == 2 and y.shape[1] > 1:
            return y[:, -1].unsqueeze(1)
        return y

    def forward_encoder(self, x, x_len, **kwargs):
        # Encode trg: (batch, length) => (batch, length, emb_dim)
        x_emb = self.src_embeddings(x)
        x_emb = self.enc_dropout(x_emb)

        # Pack sequence
        if self.packed_sequence:  # Requires bucketing
            x_emb = nn.utils.rnn.pack_padded_sequence(x_emb, x_len.to('cpu'), batch_first=True, enforce_sorted=True)

        # input: (length, batch, emb_dim)
        # output: (length, batch, hidden_dim * n_directions)
        # hidden: (n_layers * n_directions, batch, hidden_dim)
        # cell: (n_layers * n_directions, batch, hidden_dim)
        output, states = self.encoder_rnn(x_emb)

        # Unpack sequence
        if self.packed_sequence:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, states

    def forward_decoder(self, y, y_len, states, **kwargs):
        y = self.last_token(y)

        # Decode trg: (batch, 1-length) => (batch, length, emb_dim)
        y_emb = self.trg_embeddings(y)
        y_emb = self.dec_dropout(y_emb)

        # intput: (batch, 1-length, emb_dim), (n_layers * n_directions, batch, hidden_dim) =>
        # output: (batch, length, hidden_dim * n_directions)
        # hidden: (n_layers * n_directions, batch, hidden_dim)
        # cell*: (n_layers * n_directions, batch, hidden_dim)
        output, states = self.decoder_rnn(y_emb, states)

        # Get output: (batch, 1-length, hidden_dim * n_directions) => (batch, 1-length, trg_vocab_size)
        output = self.output_layer(output)
        return output, states

    def forward_enc_dec(self, x, x_len, y, y_len, **kwargs):
        # Run encoder
        _, states = self.forward_encoder(x, x_len)

        y_pred = y[:, 0]  # <sos>
        outputs = []  # Doesn't contain <sos> token

        # Iterate over trg tokens
        x_pad_mask = (x != self.padding_idx) if self.packed_sequence else None  # Mask padding
        trg_length = y.shape[1]
        for t in range(trg_length):
            outputs_t, states = self.forward_decoder(y=y_pred, y_len=y_len, states=states, x_pad_mask=x_pad_mask, **kwargs)  # (B, L, E)
            outputs.append(outputs_t)  # (B, L, V)

            # Next input?
            teacher_force = random.random() < self.teacher_forcing_ratio
            top1 = outputs_t.argmax(2)  # Get most probable next-word (logits)
            y_pred = y[:, t] if teacher_force else top1  # Use ground-truth or predicted word

        # Concatenate outputs (B, 1, V) => (B, L, V)
        outputs = torch.concat(outputs, 1)
        return outputs
