"""Train <-> test leakage detection.

Catches the most common contamination: identical sentences appearing in both
train and test. Default is exact string match; pass ``key_fn=`` to relax it
(lowercase, NFKC-normalize, drop punctuation...) or plug in your own scheme.

Intentionally a stand-alone utility — not wired into ``DatasetBuilder`` — so
you call it on whatever lines you care about, with whatever policy you want
(warn / raise / filter). For paired (src, tgt) matching, join the two sides
into one string and pass that as the line.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable, List, Optional, Tuple

from autonmt.utils.logger import get_logger

log = get_logger(__name__)


def find_leaked_lines(
    train_lines: Iterable[str],
    test_lines: Iterable[str],
    *,
    key_fn: Optional[Callable[[str], object]] = None,
) -> List[Tuple[int, str, int]]:
    """Return ``(test_idx, test_line, train_match_count)`` for every test line
    whose ``key_fn(line)`` is also present in ``train_lines``.

    ``key_fn`` must return a hashable key. Defaults to identity (exact match).
    Examples: ``str.lower``, a Unicode/whitespace normalizer, a hash of the
    sentence's content n-grams. O(N_train + N_test) with one pass per side.
    """
    key_fn = key_fn or (lambda x: x)
    train_keys = Counter(key_fn(line) for line in train_lines)
    leaked = []
    for i, line in enumerate(test_lines):
        k = key_fn(line)
        if k in train_keys:
            leaked.append((i, line, train_keys[k]))
    return leaked


def warn_on_leakage(
    train_lines: Iterable[str],
    test_lines: Iterable[str],
    *,
    key_fn: Optional[Callable[[str], object]] = None,
    label: str = "",
    max_examples: int = 3,
) -> List[Tuple[int, str, int]]:
    """``find_leaked_lines`` + ``log.warning`` with a short preview.

    Returns the same list so the caller can decide what to do (filter test set,
    abort, just log). ``label`` disambiguates when you check multiple sides
    or splits in the same run.
    """
    leaked = find_leaked_lines(train_lines, test_lines, key_fn=key_fn)
    if leaked:
        prefix = f"[{label}] " if label else ""
        log.warning(f"{prefix}Found {len(leaked)} test line(s) also present in train.")
        for idx, line, n in leaked[:max_examples]:
            preview = line[:80] + ("..." if len(line) > 80 else "")
            log.warning(f"\t- test[{idx}] (x{n} train hits): {preview!r}")
        if len(leaked) > max_examples:
            log.warning(f"\t- ... and {len(leaked) - max_examples} more")
    return leaked
