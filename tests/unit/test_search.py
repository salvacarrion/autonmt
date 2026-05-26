"""Regression tests for autonmt.core.search.{Greedy,Beam}Search."""
import torch
import torch.nn as nn

from autonmt.core.search import BeamSearch, GreedySearch


class _FixedBatchDataset:
    """Dataset whose collate_fn yields the same precomputed (src, src_len) batch,
    matching the ((x, y), (x_len, y_len)) shape that greedy_search unpacks."""

    def __init__(self, src, vocab_size=10):
        self.src = src
        self.src_len = torch.tensor([src.shape[1]] * src.shape[0])
        # beam_search reads ``len(dataset.trg_vocab)`` for the vocab-size factor
        # in its topk reshape; greedy doesn't need it, but a shared fixture must.
        self.trg_vocab = list(range(vocab_size))

    def __len__(self):
        return self.src.shape[0]

    def __getitem__(self, idx):
        return idx

    def get_collate_fn(self, max_tokens):
        def collate(_batch):
            return (self.src, None), (self.src_len, None)
        return collate


class _ConstPredModel(nn.Module):
    """Decoder always assigns the highest logit to ``const_id``."""

    def __init__(self, vocab_size, const_id):
        super().__init__()
        self._param = nn.Parameter(torch.zeros(1))  # gives parameters().device a target
        self.vocab_size = vocab_size
        self.const_id = const_id
        self.packed_sequence = False

    def forward_encoder(self, x, x_len):
        return None, None

    def forward_decoder(self, y, y_len, states, x_pad_mask):
        logits = torch.zeros(y.shape[0], y.shape[1], self.vocab_size)
        logits[:, :, self.const_id] = 10.0
        return logits, states


def test_greedy_search_breaks_on_eos_and_keeps_eos_token():
    """When every sequence emits EOS the loop short-circuits; the EOS token
    itself must remain in the output so downstream decode() can strip it."""
    sos_id, eos_id, pad_id = 1, 2, 0
    src = torch.tensor([[3, 4, 5]])  # B=1, L=3
    model = _ConstPredModel(vocab_size=10, const_id=eos_id)

    out, _ = GreedySearch().decode(
        model=model, dataset=_FixedBatchDataset(src),
        sos_id=sos_id, eos_id=eos_id, pad_id=pad_id,
        batch_size=1, max_tokens=None, max_len_a=0, max_len_b=8, num_workers=0,
    )

    # i=1 predicts EOS → break with max_iter=1 → slice y_pred[:, :2]
    assert out == [[sos_id, eos_id]]


def test_greedy_search_includes_final_token_at_length_cap():
    """Regression: when no EOS is produced, the slice must include the token
    written at the final iteration. The previous off-by-one (``:max_iter``
    instead of ``:max_iter + 1``) silently truncated the last predicted token."""
    sos_id, eos_id, pad_id = 1, 2, 0
    predicted = 5  # never EOS
    src = torch.tensor([[3, 4]])  # B=1, L=2
    model = _ConstPredModel(vocab_size=10, const_id=predicted)

    out, _ = GreedySearch().decode(
        model=model, dataset=_FixedBatchDataset(src),
        sos_id=sos_id, eos_id=eos_id, pad_id=pad_id,
        batch_size=1, max_tokens=None, max_len_a=0, max_len_b=4, num_workers=0,
    )

    # max_gen_length = int(0*2 + 4) = 4 → loop i=1,2,3 → max_iter=3 → slice [:, :4]
    assert out == [[sos_id, predicted, predicted, predicted]]


def test_beam_search_emits_eos_in_best_hypothesis():
    """When every decoder step picks EOS, the best beam must carry EOS at the
    position immediately after <sos>. Trailing tokens may appear because a
    beam-search loop only terminates once *all* beams of *all* sentences are
    finished — finished beams are locked to <pad> and keep accumulating."""
    sos_id, eos_id, pad_id = 1, 2, 0
    src = torch.tensor([[3, 4, 5]])  # B=1, L=3
    model = _ConstPredModel(vocab_size=10, const_id=eos_id)

    out, _ = BeamSearch().decode(
        model=model, dataset=_FixedBatchDataset(src),
        sos_id=sos_id, eos_id=eos_id, pad_id=pad_id,
        batch_size=1, max_tokens=None, max_len_a=0, max_len_b=8,
        beam_width=2, num_workers=0,
    )

    assert len(out) == 1
    assert out[0][0] == sos_id
    assert out[0][1] == eos_id


def test_beam_search_runs_to_length_cap_without_eos():
    """When no beam ever emits EOS the loop runs to the full length budget and
    returns a sequence of length ``max_gen_length`` with no EOS token in it."""
    sos_id, eos_id, pad_id = 1, 2, 0
    predicted = 5  # never EOS
    src = torch.tensor([[3, 4]])  # B=1, L=2
    model = _ConstPredModel(vocab_size=10, const_id=predicted)

    max_len_b = 4
    out, _ = BeamSearch().decode(
        model=model, dataset=_FixedBatchDataset(src),
        sos_id=sos_id, eos_id=eos_id, pad_id=pad_id,
        batch_size=1, max_tokens=None, max_len_a=0, max_len_b=max_len_b,
        beam_width=2, num_workers=0,
    )

    # max_gen_length = int(0*L + max_len_b) = 4 → 1 sos + 3 extension steps
    assert len(out) == 1
    assert len(out[0]) == max_len_b
    assert eos_id not in out[0]
    assert out[0][0] == sos_id
