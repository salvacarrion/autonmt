"""Regression tests for autonmt.core.decoding.*"""
import torch
import torch.nn as nn

from autonmt.core.decoding import (
    BeamSearch,
    GreedySearch,
    MultinomialSampling,
    TopKSampling,
    TopPSampling,
)


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

    def forward_decoder(self, y, y_len, states, x_pad_mask, **kwargs):
        logits = torch.zeros(y.shape[0], y.shape[1], self.vocab_size)
        logits[:, :, self.const_id] = 10.0
        return logits, states


class _StepModel(nn.Module):
    """Decoder whose next-token logits depend only on the prefix length, not on
    its content. Useful to engineer specific beam-search trajectories where the
    same prefix-length always returns the same logits across beams."""

    def __init__(self, vocab_size, logits_per_step):
        super().__init__()
        self._param = nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size
        self.logits_per_step = logits_per_step  # list of (V,) tensors
        self.packed_sequence = False

    def forward_encoder(self, x, x_len):
        return None, None

    def forward_decoder(self, y, y_len, states, x_pad_mask, **kwargs):
        idx = min(y.shape[1] - 1, len(self.logits_per_step) - 1)
        out = torch.zeros(y.shape[0], y.shape[1], self.vocab_size)
        out[:, -1, :] = self.logits_per_step[idx]
        return out, states


def test_greedy_search_breaks_on_eos_and_keeps_eos_token():
    """When every sequence emits EOS the first generated token must be EOS so
    downstream ``Vocabulary._strip_special_tokens`` can truncate the sequence
    at the first EOS. The loop probes early-stop only every ``EARLY_STOP_EVERY``
    steps to avoid a per-step CUDA sync — so trailing tokens after EOS are
    expected and harmless (they're stripped on decode)."""
    sos_id, eos_id, pad_id = 1, 2, 0
    src = torch.tensor([[3, 4, 5]])  # B=1, L=3
    model = _ConstPredModel(vocab_size=10, const_id=eos_id)

    out, _ = GreedySearch().decode(
        model=model, dataset=_FixedBatchDataset(src),
        sos_id=sos_id, eos_id=eos_id, pad_id=pad_id,
        batch_size=1, max_tokens=None, max_len_a=0, max_len_b=8, num_workers=0,
    )

    assert len(out) == 1
    assert out[0][0] == sos_id
    assert out[0][1] == eos_id


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


def test_multinomial_sampling_picks_dominant_token():
    """With one logit dwarfing the rest, the softmax is effectively a delta and
    multinomial sampling must always return that token (no seed needed)."""
    logits = torch.full((4, 10), -50.0)
    logits[:, 7] = 50.0  # token 7 dominates every row
    out = MultinomialSampling().pick_next_token(logits)
    assert out.tolist() == [7, 7, 7, 7]


def test_topp_sampling_keeps_only_boundary_token():
    """Logits ``[10, 9, 0, 0, 0]`` ⇒ softmax ≈ ``[0.73, 0.27, 0, 0, 0]``. With
    ``top_p=0.5`` only the top-1 falls inside the nucleus (its 0.73 alone
    already exceeds 0.5), so sampling is deterministically the top-1 token."""
    logits = torch.zeros(3, 5)
    logits[:, 0] = 10.0
    logits[:, 1] = 9.0
    out = TopPSampling(top_p=0.5).pick_next_token(logits)
    assert out.tolist() == [0, 0, 0]


def test_beam_search_length_penalty_flips_best_beam():
    """With ``length_penalty=0`` (raw score) the short hypothesis wins; with
    ``length_penalty=1`` (linear normalization) the longer hypothesis wins
    because its per-token average log-probability is higher.

    Setup: step-1 logits favor <eos> only slightly over token A; step-2 logits
    are dominated by <eos>. So beam 0 = [sos, eos] (length 1) just barely beats
    beam 1 = [sos, A, eos] (length 2) on raw cumulative log-prob — but beam 1's
    per-token mean is roughly twice as high, so length-1 norm flips the winner."""
    sos_id, eos_id, pad_id, A = 1, 2, 0, 3
    V = 10
    src = torch.tensor([[5, 6, 7]])

    step1 = torch.full((V,), -10.0)
    step1[eos_id] = 1.0
    step1[A] = 0.9
    step2 = torch.full((V,), -10.0)
    step2[eos_id] = 10.0
    model = _StepModel(vocab_size=V, logits_per_step=[step1, step2])

    common = dict(
        model=model, dataset=_FixedBatchDataset(src),
        sos_id=sos_id, eos_id=eos_id, pad_id=pad_id,
        batch_size=1, max_tokens=None, max_len_a=0, max_len_b=4,
        beam_width=2, num_workers=0,
    )

    out_raw, _ = BeamSearch(length_penalty=0.0).decode(**common)
    out_norm, _ = BeamSearch(length_penalty=1.0).decode(**common)

    assert out_raw[0][1] == eos_id            # short [sos, eos, ...]
    assert out_norm[0][1] == A                # longer [sos, A, eos]
    assert out_norm[0][2] == eos_id


def test_topk_sampling_only_samples_from_top_k():
    """With one logit dominating its top-k slice, top-k sampling becomes
    deterministic — even though the masked-out tokens have higher *raw* logits
    than the ones outside the kept set, only the kept set is sampled from."""
    logits = torch.full((3, 10), -100.0)
    logits[:, 2] = 100.0  # token 2 dominates within the top-k slice
    out = TopKSampling(top_k=3).pick_next_token(logits)
    assert out.tolist() == [2, 2, 2]
