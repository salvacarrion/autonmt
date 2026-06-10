"""Split-preparation helpers: resolve a variant's size spec and co-shuffle the
two sides of a parallel split in lockstep. Used by the builder and the
preprocessing stage — kept out of :mod:`autonmt.datasets.analysis.stats`, which
is purely descriptive.
"""
import random


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
