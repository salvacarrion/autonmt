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
                 padding_idx=None,
                 **kwargs):
        super().__init__(src_vocab_size, trg_vocab_size, padding_idx, **kwargs)

        # Model
        self.src_embeddings = nn.Embedding(src_vocab_size, encoder_embed_dim)
        self.trg_embeddings = nn.Embedding(trg_vocab_size, decoder_embed_dim)

        self.encoder_dropout = nn.Dropout(encoder_dropout)
        self.decoder_dropout = nn.Dropout(decoder_dropout)

        self.encoder_rnn = nn.LSTM(encoder_embed_dim, encoder_hidden_dim, encoder_n_layers, dropout=encoder_dropout)
        self.decoder_rnn = nn.LSTM(decoder_embed_dim, decoder_hidden_dim, decoder_n_layers, dropout=decoder_dropout)

        self.output_layer = nn.Linear(encoder_embed_dim, trg_vocab_size)

        # Checks
        assert encoder_embed_dim == decoder_embed_dim
        assert encoder_hidden_dim == decoder_hidden_dim
        assert encoder_n_layers == decoder_n_layers

    def forward_encoder(self, x):
        # Encode src: (length, batch) => (length, batch, emb_dim)
        x_emb = self.src_embeddings(x)
        x_emb = self.encoder_dropout(x_emb)

        # input: (length, batch, emb_dim)
        # output: (length, batch, hidden_dim * n_directions)
        # hidden: (n_layers * n_directions, batch, hidden_dim)
        # cell: (n_layers * n_directions, batch, hidden_dim]
        outputs, (hidden, cell) = self.encoder_rnn(x_emb)
        return hidden, cell

    def forward_decoder(self, y, hidden, cell):
        # Encode trg: (1-length, batch) => (length, batch, emb_dim)
        y_emb = self.trg_embeddings(y)
        y_emb = self.decoder_dropout(y_emb)

        # (1-length, batch, emb_dim) =>
        # output: (length, batch, hidden_dim * n_directions)
        # hidden: (n_layers * n_directions, batch, hidden_dim]
        # cell: (n_layers * n_directions, batch, hidden_dim]
        output, (hidden, cell) = self.decoder_rnn(y_emb, (hidden, cell))

        # Get output: (length, batch, hidden_dim * n_directions) => (length, batch, trg_vocab_size)
        output = self.output_layer(output)
        return output
