import torch
import torch.nn as nn
from autonmt.tasks.translation.models import Seq2Seq


class Transformer(Seq2Seq):
    def __init__(self,
                 src_vocab_size, trg_vocab_size,
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
                 max_sequence_length=256,
                 **kwargs):
        super().__init__(src_vocab_size, trg_vocab_size, **kwargs)
        self.max_sequence_length = max_sequence_length

        # Model
        self.src_embeddings = nn.Embedding(src_vocab_size, encoder_embed_dim)
        self.trg_embeddings = nn.Embedding(trg_vocab_size, encoder_embed_dim)
        self.pos_embeddings = nn.Embedding(max_sequence_length, encoder_embed_dim)
        self.transformer = nn.Transformer(d_model=encoder_embed_dim,
                                          nhead=encoder_attention_heads,
                                          num_encoder_layers=encoder_layers,
                                          num_decoder_layers=decoder_layers,
                                          dim_feedforward=encoder_ffn_embed_dim,
                                          dropout=dropout,
                                          activation=activation_fn)
        self.output_layer = nn.Linear(encoder_embed_dim, src_vocab_size)

        # Checks
        assert encoder_embed_dim == decoder_embed_dim
        assert encoder_attention_heads == decoder_attention_heads
        assert encoder_ffn_embed_dim == decoder_ffn_embed_dim

    def forward(self, X, Y):
        assert X.shape[1] <= self.max_sequence_length
        assert Y.shape[1] <= self.max_sequence_length

        # Encode src
        X = self.src_embeddings(X)
        X_positional = torch.arange(X.shape[1], device=X.device).repeat((X.shape[0], 1))
        X_positional = self.pos_embeddings(X_positional)
        X = (X + X_positional).transpose(0, 1)

        # Encode trg
        Y = self.trg_embeddings(Y)
        Y_positional = torch.arange(Y.shape[1], device=Y.device).repeat((Y.shape[0], 1))
        Y_positional = self.pos_embeddings(Y_positional)
        Y = (Y + Y_positional).transpose(0, 1)

        # Make trg mask
        mask = self.transformer.generate_square_subsequent_mask(Y.shape[0]).to(Y.device)

        # Forward model
        output = self.transformer.forward(src=X, tgt=Y, tgt_mask=mask)

        # Get output
        output = output.transpose(0, 1)
        output = self.output_layer(output)
        return output
