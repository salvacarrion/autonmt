"""Parity tests for the incremental Transformer decoder.

Two invariants:
  1. Module structure matches nn.TransformerDecoderLayer (state_dict transfers).
  2. Step-by-step incremental decoding produces the same logits as a single
     parallel pass on the full prefix.
"""
import torch
import torch.nn as nn

from autonmt.core.layers.incremental_transformer import (
    IncrementalTransformerDecoder,
    IncrementalTransformerDecoderLayer,
)


def _build_pair(d_model=16, nhead=2, dim_ff=32, num_layers=2, dropout=0.0):
    """Returns (custom_decoder, reference_decoder) seeded identically."""
    torch.manual_seed(0)
    custom_layer = IncrementalTransformerDecoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
        dropout=dropout, activation="relu")
    custom = IncrementalTransformerDecoder(custom_layer, num_layers=num_layers)

    torch.manual_seed(0)
    ref_layer = nn.TransformerDecoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
        dropout=dropout, activation="relu", batch_first=False)
    ref = nn.TransformerDecoder(ref_layer, num_layers=num_layers)

    # Bring weights into alignment via state_dict transfer — proves the
    # submodule layout is the same.
    custom.load_state_dict(ref.state_dict())

    custom.eval()
    ref.eval()
    return custom, ref


def test_parallel_mode_matches_pytorch_decoder():
    """With the same weights and ``incremental_state=None``, the custom layer
    must produce numerically identical outputs to ``nn.TransformerDecoder``."""
    d_model, T, L_src, B = 16, 5, 7, 3
    custom, ref = _build_pair(d_model=d_model)

    tgt = torch.randn(T, B, d_model)
    memory = torch.randn(L_src, B, d_model)
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(T)

    with torch.no_grad():
        out_custom = custom(tgt, memory, tgt_mask=tgt_mask)
        out_ref = ref(tgt, memory, tgt_mask=tgt_mask)

    assert torch.allclose(out_custom, out_ref, atol=1e-6), \
        f"max diff = {(out_custom - out_ref).abs().max().item()}"


def test_incremental_matches_parallel():
    """Feeding the prefix one token at a time with the cache must reproduce
    the logits the parallel pass would produce at the same positions."""
    d_model, T, L_src, B = 16, 6, 7, 2
    custom, _ = _build_pair(d_model=d_model)

    tgt = torch.randn(T, B, d_model)
    memory = torch.randn(L_src, B, d_model)
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(T)

    with torch.no_grad():
        # Parallel: one pass over the whole prefix.
        out_parallel = custom(tgt, memory, tgt_mask=tgt_mask)        # (T, B, D)

        # Incremental: feed tokens one at a time.
        incremental_state = {}
        out_incr = []
        for t in range(T):
            step_out = custom(tgt[t:t+1], memory, incremental_state=incremental_state)
            out_incr.append(step_out)
        out_incr = torch.cat(out_incr, dim=0)                        # (T, B, D)

    diff = (out_parallel - out_incr).abs().max().item()
    assert torch.allclose(out_parallel, out_incr, atol=1e-5), \
        f"parallel and incremental diverged: max diff = {diff}"


def test_transformer_model_incremental_matches_parallel():
    """End-to-end through ``Transformer.forward_decoder``: feeding the prefix
    one token at a time with ``incremental_state`` must reproduce the logits
    of the parallel pass at the same final position."""
    from autonmt.core.models.transformer import Transformer

    torch.manual_seed(0)
    src_v, trg_v = 20, 20
    model = Transformer(
        src_vocab_size=src_v, trg_vocab_size=trg_v,
        encoder_embed_dim=16, decoder_embed_dim=16,
        encoder_layers=2, decoder_layers=2,
        encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_embed_dim=32, decoder_ffn_embed_dim=32,
        dropout=0.0, padding_idx=0,
    )
    model.eval()

    B, L_src, T = 2, 5, 4
    x = torch.randint(1, src_v, (B, L_src))
    x_len = torch.tensor([L_src] * B)
    y = torch.randint(1, trg_v, (B, T))

    with torch.no_grad():
        _, states = model.forward_encoder(x=x, x_len=x_len)

        out_parallel, _ = model.forward_decoder(y=y, y_len=None, states=states)

        incremental_state = {}
        out_incr_steps = []
        for t in range(T):
            step_out, _ = model.forward_decoder(
                y=y[:, t:t+1], y_len=None, states=states,
                incremental_state=incremental_state,
            )
            out_incr_steps.append(step_out)
        out_incr = torch.cat(out_incr_steps, dim=1)  # (B, T, V)

    diff = (out_parallel - out_incr).abs().max().item()
    assert torch.allclose(out_parallel, out_incr, atol=1e-5), \
        f"model parallel vs incremental diverged: max diff = {diff}"


def _tiny_transformer_dataset(B=2, L_src=4, src_v=20, trg_v=20):
    """Returns (model, dataset) seeded so the search loop has something to chew."""
    from autonmt.core.models.transformer import Transformer

    torch.manual_seed(0)
    model = Transformer(
        src_vocab_size=src_v, trg_vocab_size=trg_v,
        encoder_embed_dim=16, decoder_embed_dim=16,
        encoder_layers=2, decoder_layers=2,
        encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_embed_dim=32, decoder_ffn_embed_dim=32,
        dropout=0.0, padding_idx=0,
    )
    model.eval()

    src = torch.randint(1, src_v, (B, L_src))

    class _DS:
        def __init__(self, src, vocab_size):
            self.src = src
            self.src_len = torch.tensor([src.shape[1]] * src.shape[0])
            self.trg_vocab = list(range(vocab_size))

        def __len__(self):
            return self.src.shape[0]

        def __getitem__(self, idx):
            return idx

        def get_collate_fn(self, max_tokens):
            def collate(_batch):
                return (self.src, None), (self.src_len, None)
            return collate

    return model, _DS(src, trg_v)


def test_greedy_search_incremental_matches_parallel():
    """End-to-end: ``GreedySearch.decode`` over the real Transformer must
    produce the same token sequence whether the model exposes the
    incremental path or not."""
    from autonmt.core.decoding import GreedySearch

    model, dataset = _tiny_transformer_dataset()

    common = dict(
        model=model, dataset=dataset,
        sos_id=1, eos_id=2, pad_id=0,
        batch_size=2, max_tokens=None,
        max_len_a=0, max_len_b=6, num_workers=0,
    )

    model.supports_incremental_decoding = False
    out_parallel, _ = GreedySearch().decode(**common)
    model.supports_incremental_decoding = True
    out_incr, _ = GreedySearch().decode(**common)

    assert out_parallel == out_incr


def test_beam_search_incremental_matches_parallel():
    """End-to-end: ``BeamSearch.decode`` over the real Transformer must
    produce the same best hypothesis with and without the incremental path,
    including under beam reordering (which forces cache reordering too)."""
    from autonmt.core.decoding import BeamSearch

    model, dataset = _tiny_transformer_dataset()

    common = dict(
        model=model, dataset=dataset,
        sos_id=1, eos_id=2, pad_id=0,
        batch_size=2, max_tokens=None,
        max_len_a=0, max_len_b=6, beam_width=3, num_workers=0,
    )

    model.supports_incremental_decoding = False
    out_parallel, _ = BeamSearch(length_penalty=1.0).decode(**common)
    model.supports_incremental_decoding = True
    out_incr, _ = BeamSearch(length_penalty=1.0).decode(**common)

    assert out_parallel == out_incr
