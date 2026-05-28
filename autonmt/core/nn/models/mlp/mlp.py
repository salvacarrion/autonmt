import torch
import torch.nn as nn

from autonmt.core.nn.seq2seq import LitSeq2Seq


_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


def _build_mlp(in_dim, hidden_dim, out_dim, num_layers, activation_cls, dropout):
    """Stack of ``(Linear → activation → Dropout)`` blocks ending in a Linear
    projection to ``out_dim``. ``num_layers`` counts Linear layers including the
    final projection, so ``num_layers=1`` is a single Linear with no hidden."""
    assert num_layers >= 1
    layers = []
    dim = in_dim
    for _ in range(num_layers - 1):
        layers += [nn.Linear(dim, hidden_dim), activation_cls(), nn.Dropout(dropout)]
        dim = hidden_dim
    layers += [nn.Linear(dim, out_dim)]
    return nn.Sequential(*layers)


class MLP(LitSeq2Seq):
    """Position-agnostic MLP seq2seq baseline.

    Encoder: masked mean-pool of source token embeddings → MLP → fixed-size
    context vector ``(B, D)``.
    Decoder: per-step ``concat(prev_token_emb, context)`` → MLP → vocab logits.

    No recurrence, no attention. Intended as a "no sequential inductive bias"
    baseline for benchmarking — not expected to translate well past toy
    datasets.

    In the spirit of Mikolov, Le & Sutskever, *Exploiting Similarities among
    Languages for Machine Translation*, 2013 (arXiv:1309.4168), which framed
    translation as a learned linear transformation between embedding spaces
    (W·X ≈ Y) — this model generalises that idea to a non-linear MLP over
    pooled source embeddings.
    """

    # Decoder output at position t depends only on ``y[:, t]`` and the (fixed)
    # context — no inter-position interaction — so feeding a length-1 slice
    # gives the same logits as a full-prefix pass. No cache needed; the flag
    # alone makes the search loop pass only the last token (O(L) vs O(L^2)).
    supports_incremental_decoding = True

    def __init__(self,
                 src_vocab_size, trg_vocab_size,
                 encoder_embed_dim=256,
                 decoder_embed_dim=256,
                 encoder_hidden_dim=512,
                 decoder_hidden_dim=512,
                 encoder_layers=2,
                 decoder_layers=2,
                 dropout=0.1,
                 activation_fn="relu",
                 padding_idx=None,
                 **kwargs):
        super().__init__(src_vocab_size, trg_vocab_size, padding_idx, architecture="mlp", **kwargs)

        assert encoder_embed_dim == decoder_embed_dim
        activation_cls = _ACTIVATIONS[activation_fn.lower()]

        self.src_embeddings = nn.Embedding(src_vocab_size, encoder_embed_dim)
        self.trg_embeddings = nn.Embedding(trg_vocab_size, decoder_embed_dim)
        self.input_dropout = nn.Dropout(dropout)

        # Encoder MLP: pooled (B, E) -> (B, E)  [context lives in embedding space]
        self.encoder_mlp = _build_mlp(
            in_dim=encoder_embed_dim, hidden_dim=encoder_hidden_dim, out_dim=encoder_embed_dim,
            num_layers=encoder_layers, activation_cls=activation_cls, dropout=dropout,
        )

        # Decoder MLP: (prev_emb || context) -> (B, L, H)
        self.decoder_mlp = _build_mlp(
            in_dim=decoder_embed_dim + encoder_embed_dim, hidden_dim=decoder_hidden_dim,
            out_dim=decoder_hidden_dim, num_layers=decoder_layers,
            activation_cls=activation_cls, dropout=dropout,
        )
        self.output_layer = nn.Linear(decoder_hidden_dim, trg_vocab_size)

    def forward_encoder(self, x, x_len, **kwargs):
        x_emb = self.input_dropout(self.src_embeddings(x))  # (B, L, E)

        # Masked mean-pool: ignore padding so the context isn't diluted by it.
        if self.padding_idx is not None:
            mask = (x != self.padding_idx).unsqueeze(-1).type_as(x_emb)  # (B, L, 1)
            pooled = (x_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = x_emb.mean(dim=1)

        context = self.encoder_mlp(pooled)  # (B, E)
        return None, context

    def forward_decoder(self, y, y_len, states, **kwargs):
        context = states  # (B, E)

        y_emb = self.input_dropout(self.trg_embeddings(y))                   # (B, L, E)
        ctx = context.unsqueeze(1).expand(-1, y_emb.shape[1], -1)            # (B, L, E)

        h = self.decoder_mlp(torch.cat([y_emb, ctx], dim=2))                 # (B, L, H)
        return self.output_layer(h), context                                 # (B, L, V)

    def forward_enc_dec(self, x, x_len, y, y_len, **kwargs):
        _, states = self.forward_encoder(x, x_len, **kwargs)
        output, _ = self.forward_decoder(y, y_len, states, **kwargs)
        return output
