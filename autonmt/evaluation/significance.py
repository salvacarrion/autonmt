"""Statistical significance tests for translation systems.

Reviewers in ACL/EMNLP/WMT won't accept a +0.4 BLEU improvement reported
without a p-value. This module provides a paired bootstrap test you can call
on the ``hyp.txt`` / ``ref.txt`` files produced by ``predict()``.
"""
from __future__ import annotations

import random
from typing import Dict, List

from autonmt.evaluation.metrics import score_sacrebleu


def paired_bootstrap_bleu(hyp_a: List[str], hyp_b: List[str], ref: List[str],
                          n_samples: int = 1000, seed: int = 42) -> Dict[str, float]:
    """Paired bootstrap significance test between two systems on the same test set.

    Both systems are re-scored on the *same* bootstrap resamples so the variance
    isolates the systems' difference from the noise of the test set itself.
    Returns ``{bleu_a, bleu_b, delta, p_value, n_samples}``; ``p_value`` is the
    fraction of resamples in which system B did *not* beat A (one-sided
    H0: ``score(B) <= score(A)``). Cost is ``n_samples`` calls to sacrebleu —
    a few seconds for a 3k-line test set, minutes for tens of thousands. Use
    ``n_samples >= 1000`` for reportable results.
    """
    if not (len(hyp_a) == len(hyp_b) == len(ref)):
        raise ValueError(f"hyp_a/hyp_b/ref must be line-aligned "
                         f"(got {len(hyp_a)}/{len(hyp_b)}/{len(ref)})")
    n = len(ref)
    rng = random.Random(seed)

    score_a = score_sacrebleu(hyp_a, ref, metrics={"bleu"})[0]["score"]
    score_b = score_sacrebleu(hyp_b, ref, metrics={"bleu"})[0]["score"]

    wins_b = 0
    for _ in range(n_samples):
        idx = [rng.randrange(n) for _ in range(n)]
        ra = [ref[i] for i in idx]
        ha = [hyp_a[i] for i in idx]
        hb = [hyp_b[i] for i in idx]
        sa = score_sacrebleu(ha, ra, metrics={"bleu"})[0]["score"]
        sb = score_sacrebleu(hb, ra, metrics={"bleu"})[0]["score"]
        if sb > sa:
            wins_b += 1

    return {
        "bleu_a": score_a,
        "bleu_b": score_b,
        "delta": score_b - score_a,
        "p_value": 1.0 - wins_b / n_samples,
        "n_samples": n_samples,
    }
