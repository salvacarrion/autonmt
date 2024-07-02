import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from autonmt.modules.layers import PositionalEmbedding
from autonmt.modules.seq2seq import LitSeq2Seq


class Conv(LitSeq2Seq):
    def __init__(self,
                 src_vocab_size, trg_vocab_size,
                 encoder_kernel_size=3,
                 decoder_kernel_size=3,
                 encoder_embed_dim=256,
                 decoder_embed_dim=256,
                 encoder_layers=10,
                 decoder_layers=10,
                 encoder_hidden_dim=512,
                 decoder_hidden_dim=512,
                 encoder_dropout=0.25,
                 decoder_dropout=0.25,
                 max_src_positions=100,
                 max_trg_positions=100,
                 padding_idx=None,
                 **kwargs):
        super().__init__(src_vocab_size, trg_vocab_size, padding_idx, architecture="convolutional", **kwargs)
        self.encoder_kernel_size = encoder_kernel_size
        self.decoder_kernel_size = decoder_kernel_size
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.max_src_positions = max_src_positions
        self.max_trg_positions = max_trg_positions

        assert encoder_kernel_size % 2 == 1, "Kernel size must be odd!"
        assert decoder_kernel_size % 2 == 1, "Kernel size must be odd!"

        # Encoder
        self.encoder_scale = math.sqrt(0.5)
        self.encoder_tok_embedding = nn.Embedding(src_vocab_size, encoder_embed_dim)
        self.encoder_pos_embedding = nn.Embedding(max_src_positions, encoder_embed_dim)
        self.encoder_emb2hid = nn.Linear(encoder_embed_dim, encoder_hidden_dim)
        self.encoder_hid2emb = nn.Linear(encoder_hidden_dim, encoder_embed_dim)
        self.encoder_convs = nn.ModuleList([nn.Conv1d(in_channels=encoder_hidden_dim,
                                              out_channels=2 * encoder_hidden_dim,
                                              kernel_size=encoder_kernel_size,
                                              padding=(encoder_kernel_size - 1) // 2)
                                    for _ in range(encoder_layers)])
        self.encoder_dropout = nn.Dropout(encoder_dropout)

        # Decoder
        self.decoder_scale = math.sqrt(0.5)
        self.decoder_tok_embedding = nn.Embedding(trg_vocab_size, decoder_embed_dim)
        self.decoder_pos_embedding = nn.Embedding(max_trg_positions, decoder_embed_dim)
        self.decoder_emb2hid = nn.Linear(decoder_embed_dim, decoder_hidden_dim)
        self.decoder_hid2emb = nn.Linear(decoder_hidden_dim, decoder_embed_dim)
        self.decoder_attn_hid2emb = nn.Linear(decoder_hidden_dim, decoder_embed_dim)
        self.decoder_attn_emb2hid = nn.Linear(decoder_embed_dim, decoder_hidden_dim)
        self.decoder_convs = nn.ModuleList([nn.Conv1d(in_channels=decoder_hidden_dim,
                                              out_channels=2 * decoder_hidden_dim,
                                              kernel_size=decoder_kernel_size)
                                    for _ in range(decoder_layers)])
        self.decoder_dropout = nn.Dropout(decoder_dropout)
        self.output_layer = nn.Linear(decoder_embed_dim, trg_vocab_size)


    def forward_encoder(self, x, x_len, **kwargs):
        assert x.shape[1] <= self.max_src_positions
        batch_size = x.shape[0]
        src_len = x.shape[1]

        # Create position tensor: (1...len) x batch size => (B, len)
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(x.device)

        # Encode src
        x_pos = self.encoder_pos_embedding(pos)  # (B, L, emb dim)
        x_emb = self.encoder_tok_embedding(x)  # (B, L, emb dim)
        x_emb = self.encoder_dropout(x_emb + x_pos)  # (B, L, emb dim)

        # Convert from emb dim to hid dim
        conv_input = self.encoder_emb2hid(x_emb)  # (B, L, hid dim)
        conv_input = conv_input.permute(0, 2, 1)  # (B, hid dim, L)

        # Convolutional blocks
        conved = None  # Dummy placeholder
        for i, conv in enumerate(self.encoder_convs):
            conved = self.encoder_dropout(conv_input)  # (B, hid dim, L)
            conved = conv(conved)  # (B, 2 * hid dim, L)
            conved = F.glu(conved, dim=1)  # Reduce hid dim by half: (B, hid dim, L)
            conved = (conved + conv_input) * self.encoder_scale  # Residual connection
            conv_input = conved  # Set input for next layer

        # Permute and convert back from hid dim to emb dim
        conved = conved.permute(0, 2, 1)  # (B, hid dim, L) => (B, L, hid dim)
        conved = self.encoder_hid2emb(conved)  # (B, L, emb dim)
        combined = (conved + x_emb) * self.encoder_scale

        return None, (conved, combined)

    def calculate_attention(self, y_emb, conved, encoder_conved, encoder_combined):
        conved_emb = self.decoder_attn_hid2emb(conved.permute(0, 2, 1))
        combined = (conved_emb + y_emb) * self.decoder_scale
        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1))
        attention = F.softmax(energy, dim=2)

        attended_encoding = torch.matmul(attention, encoder_combined)
        attended_encoding = self.decoder_attn_emb2hid(attended_encoding)
        attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.decoder_scale
        return attention, attended_combined

    def forward_decoder(self, y, y_len, states, **kwargs):
        assert y.shape[1] <= self.max_trg_positions
        batch_size = y.shape[0]
        trg_len = y.shape[1]
        encoder_conved, encoder_combined = states

        # Create position tensor: (1...len) x batch size => (B, len)
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(y.device)

        # Encode trg
        y_pos = self.decoder_pos_embedding(pos)  # (B, L, emb dim)
        y_emb = self.decoder_tok_embedding(y)  # (B, L, emb dim)
        y_emb = self.decoder_dropout(y_emb + y_pos)  # (B, L, emb dim)

        # Convert from emb dim to hid dim
        conv_input = self.decoder_emb2hid(y_emb)  # (B, L, hid dim)
        conv_input = conv_input.permute(0, 2, 1)  # (B, hid dim, L)

        conved = None  # Dummy placeholder
        for i, conv in enumerate(self.decoder_convs):
            conv_input = self.decoder_dropout(conv_input)  # (B, hid dim, L)

            # Pad the input so decoder can't look ahead: Pad => (B, hid dim, K-1) + Conv => (B, hid dim, L)
            padding = torch.zeros(batch_size, self.decoder_hidden_dim, self.decoder_kernel_size - 1).fill_(self.padding_idx).to(y.device)
            padded_conv_input = torch.cat((padding, conv_input), dim=2)  # (B, hid dim, L + K - 1)

            conved = conv(padded_conv_input)  # (B, 2 * hid dim, L)
            conved = F.glu(conved, dim=1)  # Reduce hid dim by half: (B, hid dim, L)

            # Calculate attention
            attention, conved = self.calculate_attention(y_emb, conved, encoder_conved, encoder_combined)

            # apply residual connection
            conved = (conved + conv_input) * self.decoder_scale  # Residual connection
            conv_input = conved  # Set input for next layer

        # Permute and convert back from hid dim to emb dim
        conved = conved.permute(0, 2, 1)  # (B, hid dim, L) => (B, L, hid dim)
        conved = self.decoder_hid2emb(conved)  # (B, L, emb dim)
        output = self.output_layer(self.decoder_dropout(conved))  # (B, L, vocab size)

        return output, (encoder_conved, encoder_combined)  # Return state for compatibility

    def forward_enc_dec(self, x, x_len, y, y_len, **kwargs):
        _, states = self.forward_encoder(x, x_len, **kwargs)
        output, _ = self.forward_decoder(y, y_len, states, **kwargs)
        return output
