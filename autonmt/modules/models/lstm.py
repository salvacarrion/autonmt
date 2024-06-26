import random
import torch
import torch.nn as nn

from autonmt.modules.layers import PositionalEmbedding
from autonmt.modules.seq2seq import LitSeq2Seq


class LSTM(LitSeq2Seq):
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
                 bidirectional=False,
                 teacher_force_ratio=0.5,
                 padding_idx=None,
                 **kwargs):
        super().__init__(src_vocab_size, trg_vocab_size, padding_idx, architecture="lstm", **kwargs)
        self.teacher_forcing_ratio = teacher_force_ratio

        # Model
        self.src_embeddings = nn.Embedding(src_vocab_size, encoder_embed_dim)
        self.trg_embeddings = nn.Embedding(trg_vocab_size, decoder_embed_dim)

        self.encoder_dropout = nn.Dropout(encoder_dropout)
        self.decoder_dropout = nn.Dropout(decoder_dropout)

        self.encoder_rnn = nn.LSTM(encoder_embed_dim, encoder_hidden_dim, encoder_n_layers, dropout=encoder_dropout, bidirectional=bidirectional, batch_first=True)
        self.decoder_rnn = nn.LSTM(decoder_embed_dim, decoder_hidden_dim, decoder_n_layers, dropout=decoder_dropout, bidirectional=bidirectional, batch_first=True)

        self.output_layer = nn.Linear(decoder_hidden_dim, trg_vocab_size)

        # Checks
        assert encoder_embed_dim == decoder_embed_dim
        assert encoder_hidden_dim == decoder_hidden_dim
        assert encoder_n_layers == decoder_n_layers

    def forward_encoder(self, x, **kwargs):
        # Encode trg: (batch, length) => (batch, length, emb_dim)
        x_emb = self.src_embeddings(x)
        x_emb = self.encoder_dropout(x_emb)

        # input: (length, batch, emb_dim)
        # output: (length, batch, hidden_dim * n_directions)
        # hidden: (n_layers * n_directions, batch, hidden_dim)
        # cell: (n_layers * n_directions, batch, hidden_dim]
        output, (hidden, cell) = self.encoder_rnn(x_emb)
        return output, (hidden, cell)

    def forward_decoder(self, y, hidden, cell, **kwargs):
        # Fix "y" dimensions
        if len(y.shape) == 1:  # (batch) => (batch, 1)
            y = y.unsqueeze(1)
        if len(y.shape) == 2 and y.shape[1] > 1:
            y = y[:, -1].unsqueeze(1)  # Get last value

        # Decode trg: (batch, 1-length) => (batch, length, emb_dim)
        y_emb = self.trg_embeddings(y)
        y_emb = self.decoder_dropout(y_emb)

        # (1-length, batch, emb_dim) =>
        # output: (batch, length, hidden_dim * n_directions)
        # hidden: (n_layers * n_directions, batch, hidden_dim]
        # cell: (n_layers * n_directions, batch, hidden_dim]
        output, (hidden, cell) = self.decoder_rnn(y_emb, (hidden, cell))

        # Get output: (length, batch, hidden_dim * n_directions) => (length, batch, trg_vocab_size)
        output = self.output_layer(output)
        return output, (hidden, cell)

    def forward_enc_dec(self, x, y):
        # Run encoder
        _, states = self.forward_encoder(x)

        y_pred = y[:, 0]  # <sos>
        outputs = []  # Doesn't contain <sos> token

        # Iterate over trg tokens
        trg_length = y.shape[1]
        for t in range(trg_length):
            outputs_t, states = self.forward_decoder(y_pred, *states)  # (B, L, E)
            outputs.append(outputs_t)  # (B, L, V)

            # Next input?
            teacher_force = random.random() < self.teacher_forcing_ratio
            top1 = outputs_t.argmax(2)  # Get most probable next-word (logits)
            y_pred = y[:, t] if teacher_force else top1  # Use ground-truth or predicted word

        # Concatenate outputs (B, 1, V) => (B, L, V)
        outputs = torch.concat(outputs, 1)
        return outputs
