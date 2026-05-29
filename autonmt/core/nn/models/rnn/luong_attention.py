import torch
import torch.nn as nn
import torch.nn.functional as F

from autonmt.core.nn.models.rnn.simple_rnn import SimpleRNN


class LuongRNN(SimpleRNN):
    """RNN encoder-decoder with multiplicative (Luong) attention.

    Unlike :class:`BahdanauRNN`, attention is computed *after* the decoder RNN
    step using the *current* hidden state ``h_t`` (not ``h_{t-1}``). Score is
    multiplicative ("general" variant from the paper)::

        score(h_t, h_j) = h_t^T · W_a · h_j

    The attended context is then combined with ``h_t`` into an "attentional
    vector"::

        h̃_t = tanh(W_c · [c_t; h_t])

    which feeds the output projection. The encoder is unidirectional in the
    paper (a stacked LSTM); this implementation follows that default but the
    parent class exposes ``encoder_bidirectional`` if you want to experiment.

    Reference: Luong, Pham & Manning, *Effective Approaches to Attention-based
    Neural Machine Translation*, EMNLP 2015 (arXiv:1508.04025).
    """

    def __init__(self, *args, base_rnn="lstm", encoder_bidirectional=False, **kwargs):
        super().__init__(*args, base_rnn=base_rnn,
                         encoder_bidirectional=encoder_bidirectional, **kwargs)

        # When the encoder is bidirectional its output dim is 2*H; otherwise H.
        enc_out_dim = self.encoder_hidden_dim * (2 if self.encoder_bidirectional else 1)

        # "general" scoring: maps each encoder annotation into the decoder
        # hidden space so a dot-product is well-defined.
        self.attn_W = nn.Linear(enc_out_dim, self.decoder_hidden_dim, bias=False)

        # W_c · [context; h_t] → h̃_t. Output projection from SimpleRNN
        # (decoder_hidden_dim → vocab) is reused unchanged on top of h̃_t.
        self.attn_combine = nn.Linear(enc_out_dim + self.decoder_hidden_dim,
                                      self.decoder_hidden_dim, bias=False)

    def forward_encoder(self, x, x_len, **kwargs):
        output, states = super().forward_encoder(x, x_len)
        # Same trick as BahdanauRNN: stash enc_outputs alongside the recurrent
        # state so the decoder gets them through the standard ``states`` arg.
        return output, (states, output)

    def forward_decoder(self, y, y_len, states, x_pad_mask=None, **kwargs):
        states, enc_outputs = states
        y = self.last_token(y)

        # (B, 1) → (B, 1, E)
        y_emb = self.dec_dropout(self.tgt_embeddings(y))

        # Luong: RNN step FIRST, then attention on the resulting h_t.
        h_t, states = self.decoder_rnn(y_emb, states)               # (B, 1, H_dec)

        attn = self._score(h_t, enc_outputs, x_pad_mask)            # (B, 1, L_src)
        context = torch.bmm(attn, enc_outputs)                      # (B, 1, H_enc)

        # Attentional vector h̃_t = tanh(W_c · [c_t; h_t]).
        h_tilde = torch.tanh(self.attn_combine(torch.cat([context, h_t], dim=2)))

        return self.output_layer(h_tilde), (states, enc_outputs)

    def _score(self, h_t, enc_outputs, x_pad_mask):
        """Multiplicative ("general") alignment scores normalised over src."""
        # (B, L_src, H_enc) → (B, L_src, H_dec) → dot with h_t for (B, 1, L_src).
        proj = self.attn_W(enc_outputs)
        scores = torch.bmm(h_t, proj.transpose(1, 2))

        if x_pad_mask is not None:
            scores = scores.masked_fill(x_pad_mask.unsqueeze(1) == 0, -1e10)

        return F.softmax(scores, dim=-1)
