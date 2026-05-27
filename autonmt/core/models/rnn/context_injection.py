import torch
import torch.nn as nn

from autonmt.core.models.rnn.simple_rnn import SimpleRNN


class ContextRNN(SimpleRNN):
    """Encoder-decoder RNN that re-injects the encoder context at every step.

    Unlike :class:`SimpleRNN`, the final encoder hidden state is concatenated
    to every decoder input and to the output projection — so the decoder never
    has to keep the whole source in its recurrent state. Default cell is GRU,
    matching the paper.

    Reference: Cho et al., *Learning Phrase Representations using RNN
    Encoder-Decoder for Statistical Machine Translation*, EMNLP 2014
    (arXiv:1406.1078). This is also the paper introducing the GRU.
    """

    def __init__(self, *args, base_rnn="gru", **kwargs):
        super().__init__(*args, base_rnn=base_rnn, **kwargs)
        base_rnn_cls = self.resolve_base_rnn(base_rnn)
        self.encoder_rnn = base_rnn_cls(input_size=self.encoder_embed_dim,
                                        hidden_size=self.encoder_hidden_dim,
                                        num_layers=self.encoder_n_layers,
                                        dropout=self.encoder_dropout,
                                        bidirectional=self.encoder_bidirectional, batch_first=True)
        self.decoder_rnn = base_rnn_cls(input_size=self.decoder_embed_dim + self.encoder_hidden_dim,
                                        hidden_size=self.decoder_hidden_dim,
                                        num_layers=self.decoder_n_layers,
                                        dropout=self.decoder_dropout,
                                        bidirectional=self.decoder_bidirectional, batch_first=True)
        self.output_layer = nn.Linear(self.decoder_embed_dim + self.decoder_hidden_dim * 2, self.trg_vocab_size)

    def forward_encoder(self, x, x_len, **kwargs):
        output, states = super().forward_encoder(x, x_len)

        # Clone states
        if isinstance(states, tuple):  # Trick to save the context (last hidden state of the encoder)
            context = tuple([s.clone() for s in states])
        else:
            context = states.clone()

        return output, (states, context)  # (states, context)

    def forward_decoder(self, y, y_len, states, **kwargs):
        states, context = states
        y = self.last_token(y)

        # Decode trg: (batch, 1-length) => (batch, length, emb_dim)
        y_emb = self.trg_embeddings(y)
        y_emb = self.dec_dropout(y_emb)

        # Add context (reduce to 1 layer)
        tmp_context = context[0] if isinstance(context, tuple) else context  # Get hidden state
        tmp_context = tmp_context.transpose(1, 0).sum(axis=1, keepdims=True)  # The paper has just 1 layer
        y_context = torch.cat((y_emb, tmp_context), dim=2)

        # intput: (batch, 1-length, emb_dim), (n_layers * n_directions, batch, hidden_dim)
        # output: (batch, length, hidden_dim * n_directions)
        # hidden: (n_layers * n_directions, batch, hidden_dim)
        output, states = self.decoder_rnn(y_context, states)

        # Add context
        tmp_hidden = states[0] if isinstance(states, tuple) else states  # Get hidden state
        tmp_hidden = tmp_hidden.transpose(1, 0).sum(axis=1, keepdims=True)  # The paper has just 1 layer
        output = torch.cat((y_emb, tmp_hidden, tmp_context), dim=2)

        # Get output: (batch, length, hidden_dim * n_directions) => (batch, length, trg_vocab_size)
        output = self.output_layer(output)
        return output, (states, context)
