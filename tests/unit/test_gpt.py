"""GPT (decoder-only) model invariants.

  1. The LitBase refactor: both LitSeq2Seq and LitLM/GPT share the same base.
  2. forward produces ``(B, L, vocab)`` logits.
  3. KV-cached incremental decoding matches the parallel forward (the key
     correctness property of the new CausalSelfAttention block).
  4. ``_step`` applies the supervise mask and yields a finite loss.
"""
import torch

from autonmt.core.nn.base import LitBase
from autonmt.core.nn.lm import LitLM
from autonmt.core.nn.seq2seq import LitSeq2Seq
from autonmt.core.nn.models import GPT


def _model(vocab=40, **kw):
    torch.manual_seed(0)
    defaults = dict(vocab_size=vocab, padding_idx=3, embed_dim=64, num_layers=2,
                    num_heads=4, dropout=0.0, max_seq_len=64)
    defaults.update(kw)
    m = GPT(**defaults)
    m.eval()
    return m


def test_shared_litbase():
    assert issubclass(LitSeq2Seq, LitBase)
    assert issubclass(LitLM, LitBase)
    assert issubclass(GPT, LitLM)


def test_forward_shape():
    V, B, L = 40, 3, 12
    m = _model(vocab=V)
    logits = m(torch.randint(0, V, (B, L)))
    assert logits.shape == (B, L, V)


def test_incremental_matches_parallel():
    V, B, L = 40, 2, 10
    m = _model(vocab=V)
    x = torch.randint(0, V, (B, L))
    with torch.no_grad():
        parallel = m(x)
        inc = {}
        m(x[:, :L - 1], incremental_state=inc)         # prefill
        step = m(x[:, L - 1:L], incremental_state=inc)  # one more token
    assert torch.allclose(step[:, -1, :], parallel[:, -1, :], atol=1e-4)


def test_tied_embeddings():
    m = _model(tie_embeddings=True)
    assert m.output_layer.weight is m.tok_embeddings.weight


def test_step_supervise_mask():
    V, B, L = 40, 3, 12
    m = _model(vocab=V)
    m.configure_criterion("cross_entropy")
    x = torch.randint(0, V, (B, L))
    y = torch.randint(0, V, (B, L))

    sup = torch.ones(B, L, dtype=torch.long)
    sup[:, : L // 2] = 0  # mask the "prompt" half (instruct-style)
    loss = m._step((x, y, sup), log_prefix="train")
    assert torch.isfinite(loss)

    # Pure-LM mask (all ones) also yields a finite loss.
    loss_full = m._step((x, y, torch.ones(B, L, dtype=torch.long)), log_prefix="train")
    assert torch.isfinite(loss_full)
