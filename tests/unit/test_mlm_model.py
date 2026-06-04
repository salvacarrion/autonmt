"""MLMTransformer (encoder-only) model invariants.

  1. Shared LitBase across the three model families.
  2. forward produces ``(B, L, vocab)`` logits.
  3. The encoder is **bidirectional** (no causal mask): changing a later token
     affects an earlier position's logits — the defining difference from GPT.
  4. ``_step`` computes loss only over non-ignored (masked) positions.
"""
import torch

from autonmt.core.nn.base import LitBase
from autonmt.core.nn.mlm import LitMLM
from autonmt.core.nn.lm import LitLM
from autonmt.core.nn.seq2seq import LitSeq2Seq
from autonmt.core.nn.models import MLMTransformer


def _model(vocab=50, **kw):
    torch.manual_seed(0)
    defaults = dict(vocab_size=vocab, padding_idx=3, embed_dim=64, num_layers=2,
                    num_heads=4, dropout=0.0, max_seq_len=64)
    defaults.update(kw)
    m = MLMTransformer(**defaults)
    m.eval()
    return m


def test_shared_litbase():
    for cls in (LitSeq2Seq, LitLM, LitMLM):
        assert issubclass(cls, LitBase)
    assert issubclass(MLMTransformer, LitMLM)


def test_forward_shape():
    V, B, L = 50, 2, 12
    logits = _model(vocab=V)(torch.randint(5, V, (B, L)))
    assert logits.shape == (B, L, V)


def test_bidirectional():
    V, B, L = 50, 2, 12
    m = _model(vocab=V)
    x = torch.randint(5, V, (B, L))
    x2 = x.clone()
    x2[:, -1] = (x[:, -1] + 1) % V          # perturb the LAST token
    with torch.no_grad():
        l1, l2 = m(x), m(x2)
    # An earlier position must react → attention is not causal.
    assert (l1[:, 0, :] - l2[:, 0, :]).abs().max().item() > 1e-6


def test_tied_embeddings():
    m = _model(tie_embeddings=True)
    assert m.output_layer.weight is m.tok_embeddings.weight


def test_step_masked_loss():
    V, B, L = 50, 3, 12
    m = _model(vocab=V)
    m.configure_criterion("cross_entropy")     # ignore_index = padding_idx (3)
    x = torch.randint(5, V, (B, L))
    y = torch.full((B, L), 3, dtype=torch.long)  # all ignored...
    y[:, 3] = x[:, 3]                            # ...except two masked targets
    y[:, 8] = x[:, 8]
    loss = m._step((x, y), log_prefix="train")
    assert torch.isfinite(loss)
