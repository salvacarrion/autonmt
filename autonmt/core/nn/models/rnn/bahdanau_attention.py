import torch
import torch.nn as nn
import torch.nn.functional as F

from autonmt.core.nn.models.rnn.simple_rnn import SimpleRNN


class BahdanauRNN(SimpleRNN):
    """RNN encoder-decoder with additive (Bahdanau) attention.

    Bidirectional encoder produces a sequence of annotations; at each decoder
    step the *previous* hidden state attends over those annotations with an
    additive scoring MLP (``v · tanh(W·[s_{t-1}; h_j])``). The attended context
    is fed back into the decoder RNN **input** and into the output projection.
    Lifts the fixed-size bottleneck of :class:`SimpleRNN` / :class:`ContextRNN`.

    Contrast with :class:`LuongRNN`, which uses the *current* hidden state
    (post-RNN step) and a multiplicative score.

    Reference: Bahdanau, Cho & Bengio, *Neural Machine Translation by Jointly
    Learning to Align and Translate*, ICLR 2015 (arXiv:1409.0473).
    """

    def __init__(self, *args, base_rnn="gru", **kwargs):
        super().__init__(*args, base_rnn=base_rnn, **kwargs)
        base_rnn_cls = self.resolve_base_rnn(self.base_rnn)
        self.encoder_rnn = base_rnn_cls(input_size=self.encoder_embed_dim,
                                        hidden_size=self.encoder_hidden_dim,
                                        num_layers=self.encoder_n_layers,
                                        dropout=self.encoder_dropout,
                                        bidirectional=True, batch_first=True)
        self.decoder_rnn = base_rnn_cls(input_size=self.decoder_embed_dim + self.encoder_hidden_dim * 2,
                                        hidden_size=self.decoder_hidden_dim,
                                        num_layers=self.decoder_n_layers,
                                        dropout=self.decoder_dropout,
                                        bidirectional=False, batch_first=True)

        # Attention
        self.attn = nn.Linear((self.encoder_hidden_dim * 2) + self.decoder_hidden_dim, self.decoder_hidden_dim)
        self.v = nn.Linear(self.decoder_hidden_dim, 1, bias=False)

        self.enc_ffn = nn.Linear(self.encoder_hidden_dim * self.encoder_n_layers * 2,
                                 self.decoder_hidden_dim * self.decoder_n_layers)
        self.output_layer = nn.Linear(self.decoder_embed_dim + self.decoder_hidden_dim + self.encoder_hidden_dim * 2,
                                      self.trg_vocab_size)

    def forward_encoder(self, x, x_len, **kwargs):
        # input: (B, L) =>
        # output: (B, L, hidden_dim * n_directions)
        # hidden: (n_layers * n_directions, batch, hidden_dim)
        output, states = super().forward_encoder(x, x_len)

        # Reshape hidden to (batch, n_layers * n_directions * hidden_dim)
        states = states if isinstance(states, tuple) else (states,)  # Get hidden state
        states = list(states)
        for i in range(len(states)):
            states[i] = states[i].transpose(0, 1).contiguous().view(states[i].size(1), -1)

            # Apply the linear transformation
            states[i] = torch.tanh(self.enc_ffn(states[i]))

            # Reshape back to (n_layers, batch, decoder_hidden_dim)
            states[i] = states[i].view(self.encoder_n_layers, -1, self.decoder_hidden_dim)

        # Fix states shape
        states = tuple(states) if len(states) > 1 else states[0]
        return output, (states, output)

    def forward_decoder(self, y, y_len, states, x_pad_mask=None, **kwargs):
        states, enc_outputs = states
        y = self.last_token(y)

        # Decode trg: (batch, 1-length) => (batch, length, emb_dim)
        y_emb = self.trg_embeddings(y)
        y_emb = self.dec_dropout(y_emb)

        # Attention (using only the top layer of hidden state)
        attn = self.attention(states, enc_outputs, x_pad_mask)
        attn = attn.unsqueeze(1)  # (B, L) => (B, 1, L)
        weighted = torch.bmm(attn, enc_outputs)  # (B, 1, L) x (B, L, H) => (B, 1, H)

        # intput: (batch, 1-length, emb_dim+w_emb_dim), (1, batch, hidden_dim)
        # output: (batch, length, hidden_dim * n_directions)
        # hidden: (n_layers * n_directions, batch, hidden_dim)
        rnn_input = torch.cat((y_emb, weighted), dim=2)
        output, states = self.decoder_rnn(rnn_input, states)

        # Get output: => (B, 1-length, H+H+H)
        output = torch.cat((output, weighted, y_emb), dim=2)
        output = self.output_layer(output)  # (B, 1, H+H+H) => (B, 1, V)

        return output, (states, enc_outputs)  # pass enc_outputs (trick)

    def attention(self, states, encoder_outputs, x_pad_mask):
        hidden = states[0][-1] if isinstance(states, tuple) else states[-1]  # Get hidden state
        src_len = encoder_outputs.shape[1]

        # Repeat decoder hidden state "src_len" times: (B, emb) => (B, src_len, hid_dim)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # Compute energy
        energy = torch.cat((hidden, encoder_outputs), dim=2)  # => (B, L, hid_dim+hid_dim)
        energy = self.attn(energy)  # => (B, L, hid_dim)
        energy = torch.tanh(energy)

        # Compute attention
        attention = self.v(energy).squeeze(2)  # (B, L, H) => (B, L)  # "weight logits"

        # Mask attention
        if x_pad_mask is not None:
            attention = attention.masked_fill(x_pad_mask == 0, -1e10)

        return F.softmax(attention, dim=1)  # (B, L): normalized between 0..1 (attention)
