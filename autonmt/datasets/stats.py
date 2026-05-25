"""Statistical helpers and small data-manipulation utilities.

Pure functions, no I/O beyond reading the files the caller hands in. Keep this
module free of dataset / plot concerns.
"""
import random
from collections import Counter

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Token-level stats over a file
# ---------------------------------------------------------------------------

def count_tokens_per_sentence(filename, split_fn=None):
    if split_fn is None:
        split_fn = lambda x: x.strip().split(' ')
    with open(filename, 'r') as f:
        return [len(split_fn(line)) for line in f.readlines()]


def basic_stats(tokens, prefix=""):
    """Return min/max/avg/percentile summary of an integer-valued numpy array.

    The percentile tiers (99.671, 99.749, 99.982, 99.995) match the I/II/III/IV
    NMT-sentence-length buckets we report on; do not rename without checking
    downstream report consumers.
    """
    assert isinstance(tokens, np.ndarray)
    return {
        f"{prefix}total_sentences": len(tokens),
        f"{prefix}total_tokens": int(tokens.sum()),
        f"{prefix}max_tokens": int(np.max(tokens)),
        f"{prefix}min_tokens": int(np.min(tokens)),
        f"{prefix}avg_tokens": float(np.average(tokens)),
        f"{prefix}std_tokens": float(np.std(tokens)),
        f"{prefix}percentile5_tokens": int(np.percentile(tokens, 5)),
        f"{prefix}percentile50_tokens": int(np.percentile(tokens, 50)),
        f"{prefix}percentile95_tokens": int(np.percentile(tokens, 95)),
        f"{prefix}percentile99_tokens": int(np.percentile(tokens, 99)),
        f"{prefix}percentile99.671_tokens": int(np.percentile(tokens, 99.671)),  # TIER I
        f"{prefix}percentile99.749_tokens": int(np.percentile(tokens, 99.749)),  # TIER II
        f"{prefix}percentile99.982_tokens": int(np.percentile(tokens, 99.982)),  # TIER III
        f"{prefix}percentile99.995_tokens": int(np.percentile(tokens, 99.995)),  # TIER IV
    }


# ---------------------------------------------------------------------------
# Vocabulary counters
# ---------------------------------------------------------------------------

def build_counter_low_mem(filename, split_fn):
    """Stream-count tokens from a file without loading it into memory."""
    c = Counter()
    with open(filename, 'r') as f:
        for line in tqdm(f):
            c.update(split_fn(line.strip()))
    return c


def norm_counter(c):
    """Return a copy of ``c`` with values rescaled so they sum to 1."""
    c = Counter(c)
    total = sum(c.values(), 0.0)
    for key in c:
        c[key] /= total
    return c


# ---------------------------------------------------------------------------
# Misc small helpers used by preprocessing
# ---------------------------------------------------------------------------

def parse_split_size(ds_size, max_ds_size):
    """Resolve a split-size spec to an absolute number of lines.

    Accepts:
      - ``int``        -> taken as-is
      - ``float``      -> fraction of ``max_ds_size``
      - ``(frac, cap)``-> min(frac * max_ds_size, cap)
    """
    if isinstance(ds_size, tuple):
        return int(min(float(ds_size[0]) * max_ds_size, ds_size[1]))
    if isinstance(ds_size, float):
        return float(ds_size) * max_ds_size
    if isinstance(ds_size, int):
        return ds_size
    raise TypeError("'ds_size' can be a tuple(float, int), float or int")


def shuffle_in_order(list1, list2):
    """Co-shuffle two equal-length sequences in lockstep."""
    paired = list(zip(list1, list2))
    random.shuffle(paired)
    a, b = zip(*paired)
    return list(a), list(b)
