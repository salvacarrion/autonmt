import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from autonmt.modules.layers import PositionalEmbedding
from autonmt.modules.seq2seq import LitSeq2Seq


class BaseRNN(LitSeq2Seq):
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
                 architecture="base_rnn",
                 **kwargs):
        super().__init__(src_vocab_size, trg_vocab_size, padding_idx, architecture=architecture, **kwargs)
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.encoder_n_layers = encoder_n_layers
        self.decoder_n_layers = decoder_n_layers
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.bidirectional = bidirectional
        self.teacher_forcing_ratio = teacher_force_ratio

        # Model
        self.src_embeddings = nn.Embedding(src_vocab_size, encoder_embed_dim)
        self.trg_embeddings = nn.Embedding(trg_vocab_size, decoder_embed_dim)
        self.enc_dropout = nn.Dropout(encoder_dropout)
        self.dec_dropout = nn.Dropout(decoder_dropout)
        self.encoder_rnn = None
        self.decoder_rnn = None
        self.output_layer = nn.Linear(decoder_hidden_dim, trg_vocab_size)

        # Checks
        assert encoder_embed_dim == decoder_embed_dim
        assert encoder_hidden_dim == decoder_hidden_dim
        assert encoder_n_layers == decoder_n_layers

    def forward_encoder(self, x):
        # Encode trg: (batch, length) => (batch, length, emb_dim)
        x_emb = self.src_embeddings(x)
        x_emb = self.enc_dropout(x_emb)

        # input: (length, batch, emb_dim)
        # output: (length, batch, hidden_dim * n_directions)
        # hidden: (n_layers * n_directions, batch, hidden_dim)
        # cell: (n_layers * n_directions, batch, hidden_dim)
        output, states = self.encoder_rnn(x_emb)
        return output, states

    def forward_decoder(self, y, states):
        # Fix "y" dimensions
        if len(y.shape) == 1:  # (batch) => (batch, 1)
            y = y.unsqueeze(1)
        if len(y.shape) == 2 and y.shape[1] > 1:
            y = y[:, -1].unsqueeze(1)  # Get last value

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

    def forward_enc_dec(self, x, y):
        # Run encoder
        _, states = self.forward_encoder(x)

        y_pred = y[:, 0]  # <sos>
        outputs = []  # Doesn't contain <sos> token

        # Iterate over trg tokens
        trg_length = y.shape[1]
        for t in range(trg_length):
            outputs_t, states = self.forward_decoder(y_pred, states)  # (B, L, E)
            outputs.append(outputs_t)  # (B, L, V)

            # Next input?
            teacher_force = random.random() < self.teacher_forcing_ratio
            top1 = outputs_t.argmax(2)  # Get most probable next-word (logits)
            y_pred = y[:, t] if teacher_force else top1  # Use ground-truth or predicted word

        # Concatenate outputs (B, 1, V) => (B, L, V)
        outputs = torch.concat(outputs, 1)
        return outputs


class GenericRNN(BaseRNN):

    def __init__(self, architecture="lstm", **kwargs):
        super().__init__(architecture=architecture, **kwargs)

        # Choose architecture
        architecture = architecture.lower().strip()
        if architecture == "rnn":
            base_rnn = nn.RNN
        elif architecture == "lstm":
            base_rnn = nn.LSTM
        elif architecture == "gru":
            base_rnn = nn.GRU
        else:
            raise ValueError(f"Invalid architecture: {architecture}. Choose: 'rnn', 'lstm' or 'gru'")

        self.encoder_rnn = base_rnn(input_size=self.encoder_embed_dim,
                                    hidden_size=self.encoder_hidden_dim,
                                    num_layers=self.encoder_n_layers,
                                    dropout=self.encoder_dropout,
                                    bidirectional=self.bidirectional, batch_first=True)
        self.decoder_rnn = base_rnn(input_size=self.decoder_embed_dim,
                                    hidden_size=self.decoder_hidden_dim,
                                    num_layers=self.decoder_n_layers,
                                    dropout=self.decoder_dropout,
                                    bidirectional=self.bidirectional, batch_first=True)


class GRU(BaseRNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, architecture="gru", **kwargs)
        self.encoder_rnn = nn.GRU(input_size=self.encoder_embed_dim,
                                  hidden_size=self.encoder_hidden_dim,
                                  num_layers=self.encoder_n_layers,
                                  dropout=self.encoder_dropout,
                                  bidirectional=self.bidirectional, batch_first=True)
        self.decoder_rnn = nn.GRU(input_size=self.decoder_embed_dim + self.encoder_hidden_dim,
                                  hidden_size=self.decoder_hidden_dim,
                                  num_layers=self.decoder_n_layers,
                                  dropout=self.decoder_dropout,
                                  bidirectional=self.bidirectional, batch_first=True)
        self.output_layer = nn.Linear(self.decoder_embed_dim + self.decoder_hidden_dim*2, self.trg_vocab_size)

    def forward_encoder(self, x):
        output, states = super().forward_encoder(x)
        context = states.clone()
        return output, (states, context)  # (hidden, context)

    def forward_decoder(self, y, states):
        hidden, context = states

        # Fix "y" dimensions
        if len(y.shape) == 1:  # (batch) => (batch, 1)
            y = y.unsqueeze(1)
        if len(y.shape) == 2 and y.shape[1] > 1:
            y = y[:, -1].unsqueeze(1)  # Get last value

        # Decode trg: (batch, 1-length) => (batch, length, emb_dim)
        y_emb = self.trg_embeddings(y)
        y_emb = self.dec_dropout(y_emb)

        # Add context
        tmp_context = context.transpose(1, 0).sum(axis=1, keepdims=True)  # The paper has just 1 layer
        y_context = torch.cat((y_emb, tmp_context), dim=2)

        # intput: (batch, 1-length, emb_dim), (n_layers * n_directions, batch, hidden_dim)
        # output: (batch, length, hidden_dim * n_directions)
        # hidden: (n_layers * n_directions, batch, hidden_dim)
        output, hidden = self.decoder_rnn(y_context, hidden)

        # Add context
        tmp_hidden = hidden.transpose(1, 0).sum(axis=1, keepdims=True)   # The paper has just 1 layer
        output = torch.cat((y_emb, tmp_hidden, tmp_context), dim=2)

        # Get output: (batch, length, hidden_dim * n_directions) => (batch, length, trg_vocab_size)
        output = self.output_layer(output)
        return output, (hidden, context)

class Attention(nn.Module):

    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn = nn.Linear((encoder_hidden_dim * 2) + decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]

        # Repeat decoder hidden state "src_len" times: (B, emb) => (B, src_len, hid_dim)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # Compute energy
        energy = torch.cat((hidden, encoder_outputs), dim=2)  # => (B, L, hid_dim+hid_dim)
        energy = self.attn(energy)  # => (B, L, hid_dim)
        energy = torch.tanh(energy)

        # Compute attention
        attention = self.v(energy).squeeze(2)  # (B, L, H) => (B, L)  # "weight logits"
        return F.softmax(attention, dim=1)  # (B, L): normalized between 0..1 (attention)

class GenericRNNAttention(BaseRNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, architecture="gru", **kwargs)
        self.encoder_rnn = nn.GRU(input_size=self.encoder_embed_dim,
                                  hidden_size=self.encoder_hidden_dim,
                                  num_layers=1,
                                  dropout=self.encoder_dropout,
                                  bidirectional=True, batch_first=True)
        self.decoder_rnn = nn.GRU(input_size=self.decoder_embed_dim + self.encoder_hidden_dim*2,
                                  hidden_size=self.decoder_hidden_dim,
                                  num_layers=1,
                                  dropout=self.decoder_dropout,
                                  bidirectional=False, batch_first=True)
        self.attention = Attention(self.encoder_hidden_dim, self.decoder_hidden_dim)
        self.enc_ffn = nn.Linear(self.encoder_hidden_dim * 2, self.decoder_hidden_dim)
        self.output_layer = nn.Linear(self.decoder_embed_dim + self.decoder_hidden_dim + self.encoder_hidden_dim*2, self.trg_vocab_size)

    def forward_encoder(self, x):
        # input: (B, L) =>
        # output: (B, L, hidden_dim * n_directions)
        # hidden: (n_layers * n_directions, batch, hidden_dim)
        output, hidden = super().forward_encoder(x)

        # bidirectional hidden is stacked [forward_1, backward_1, forward_2, backward_2,...]
        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN
        # Concat hidden layers (back and forward) from the last layer: (B, emb) => (B, emb*2)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

        # Apply the linear transformation to each layer to match the decoder hidden size
        hidden = torch.tanh(self.enc_ffn(hidden))
        return output, (hidden, output)

    def forward_decoder(self, y, states):
        hidden, enc_outputs = states

        # Fix "y" dimensions
        if len(y.shape) == 1:  # (batch) => (batch, 1)
            y = y.unsqueeze(1)
        if len(y.shape) == 2 and y.shape[1] > 1:
            y = y[:, -1].unsqueeze(1)  # Get last value

        # Decode trg: (batch, 1-length) => (batch, length, emb_dim)
        y_emb = self.trg_embeddings(y)
        y_emb = self.dec_dropout(y_emb)

        # Attention
        attn = self.attention(hidden, enc_outputs)
        attn = attn.unsqueeze(1)  # (B, L) => (B, 1, L)
        weighted = torch.bmm(attn, enc_outputs)  # (B, 1, L) x (B, L, H) => (B, 1, H)

        # intput: (batch, 1-length, emb_dim+w_emb_dim), (1, batch, hidden_dim)
        # output: (batch, length, hidden_dim * n_directions)
        # hidden: (n_layers * n_directions, batch, hidden_dim)
        rnn_input = torch.cat((y_emb, weighted), dim=2)
        output, hidden = self.decoder_rnn(rnn_input, hidden.unsqueeze(0))

        # Get output: => (B, 1-length, H+H+H)
        output = torch.cat((output, weighted, y_emb), dim=2)
        output = self.output_layer(output)  # (B, 1, H+H+H) => (B, 1, V)
        return output, (hidden.squeeze(0), enc_outputs)  # pass enc_outputs (trick)
