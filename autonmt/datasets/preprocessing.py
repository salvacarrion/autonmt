"""Subword-agnostic text preparation: filter, normalize, dedupe, shuffle.

These functions produce the contents of ``2_preprocessed/`` and are independent
of the subword model choice (same output whether the run uses ``word``, ``bpe``,
``unigram``, etc.). Subword-dependent file ops live in
:mod:`autonmt.datasets.encoding`.
"""
import collections
import os
import random

import numpy as np

from tokenizers import normalizers
from tokenizers.normalizers import NFKC, Strip

from autonmt.utils.fileio import read_file_lines, write_file_lines
from autonmt.utils.logger import get_logger
from autonmt.datasets.stats import shuffle_in_order
from autonmt.datasets import tokenizers

log = get_logger(__name__)


def _log_removed(label, before, after):
    pct = (1 - after / before) * 100 if before else 0
    log.info(f"\t\t\t- Removed {before - after:,} lines ({label}; {pct:.3f}%)")


def _filter_lengths_pairs(src_lines, tgt_lines, min_len, max_len, max_len_percentile):
    min_len = 0 if min_len is None else min_len
    max_len = float('inf') if max_len is None else max_len
    max_len_src = max_len_tgt = max_len

    if max_len_percentile and max_len_percentile < 100:
        src_lengths, tgt_lengths = zip(*[(len(s), len(t)) for s, t in zip(src_lines, tgt_lines)])
        max_len_src = np.percentile(np.array(src_lengths), max_len_percentile)
        max_len_tgt = np.percentile(np.array(tgt_lengths), max_len_percentile)

    log.info("\t\t- Checking lengths...")
    before = len(src_lines)
    pairs = [(s, t) for s, t in zip(src_lines, tgt_lines)
             if min_len <= len(s) <= max_len_src and min_len <= len(t) <= max_len_tgt]
    if not pairs:
        _log_removed("invalid lengths", before, 0)
        return [], []
    src_lines, tgt_lines = zip(*pairs)
    assert len(src_lines) == len(tgt_lines)
    _log_removed("invalid lengths", before, len(src_lines))
    return list(src_lines), list(tgt_lines)


def _filter_length_ratio(src_lines, tgt_lines, max_len_ratio_percentile, safe_len_ratio):
    log.info("\t\t- Removing pairs whose length ratios differ too much...")
    before = len(src_lines)
    ratios = np.array([max(len(s), len(t)) / min(len(s), len(t))
                       for s, t in zip(src_lines, tgt_lines)])
    threshold = np.percentile(ratios, max_len_ratio_percentile)
    if threshold < safe_len_ratio:
        log.info(f"\t\t\t- Percentile threshold overruled (safe threshold: {safe_len_ratio:.2f})")
        threshold = max(threshold, safe_len_ratio)
    log.info(f"\t\t\t- Threshold: {threshold:.2f} (percentile: {max_len_ratio_percentile:.2f})")
    pairs = [(s, t) for s, t in zip(src_lines, tgt_lines)
             if max(len(s), len(t)) / min(len(s), len(t)) <= threshold]
    src_lines, tgt_lines = list(zip(*pairs)) if pairs else ([], [])
    assert len(src_lines) == len(tgt_lines)
    _log_removed("length difference", before, len(src_lines))
    return list(src_lines), list(tgt_lines)


def _dedupe_pairs(src_lines, tgt_lines):
    log.info("\t\t- Removing duplicates...")
    before = len(src_lines)
    pairs = list(collections.Counter(zip(src_lines, tgt_lines)).keys())
    src_lines, tgt_lines = list(zip(*pairs)) if pairs else ([], [])
    assert len(src_lines) == len(tgt_lines)
    _log_removed("duplicates", before, len(src_lines))
    return list(src_lines), list(tgt_lines)


def preprocess_pairs(src_lines, tgt_lines, normalize_fn=None, min_len=None, max_len=None,
                     max_len_percentile=None, remove_duplicates=False,
                     max_len_ratio_percentile=100, safe_len_ratio=2.0, shuffle_lines=False):
    assert len(src_lines) == len(tgt_lines)
    total0 = len(src_lines)

    if normalize_fn:
        log.info(f"\t\t- Normalizing {len(src_lines):,} pairs...")
        src_lines = normalize_fn(src_lines)
        tgt_lines = normalize_fn(tgt_lines)

    if min_len is not None or max_len is not None or max_len_percentile is not None:
        src_lines, tgt_lines = _filter_lengths_pairs(
            src_lines, tgt_lines, min_len, max_len, max_len_percentile)

    if remove_duplicates:
        src_lines, tgt_lines = _dedupe_pairs(src_lines, tgt_lines)

    if max_len_ratio_percentile and max_len_ratio_percentile < 100:
        src_lines, tgt_lines = _filter_length_ratio(
            src_lines, tgt_lines, max_len_ratio_percentile, safe_len_ratio)

    if shuffle_lines:
        log.info(f"\t\t- Shuffling {len(src_lines):,} pairs...")
        src_lines, tgt_lines = shuffle_in_order(src_lines, tgt_lines)

    _log_removed("total", total0, len(src_lines))
    return src_lines, tgt_lines


def preprocess_lines(lines, normalize_fn=None, min_len=None, max_len=None,
                     remove_duplicates=False, shuffle_lines=False):
    total0 = len(lines)
    min_len = 0 if min_len is None else min_len
    max_len = float('inf') if max_len is None else max_len

    if normalize_fn:
        log.info(f"\t\t- Normalizing {len(lines):,} lines...")
        lines = normalize_fn(lines)

    log.info("\t\t- Checking lengths...")
    before = len(lines)
    lines = [l for l in lines if min_len <= len(l) <= max_len]
    _log_removed("invalid lengths", before, len(lines))

    if remove_duplicates:
        log.info("\t\t- Removing duplicates...")
        before = len(lines)
        lines = list(collections.Counter(lines).keys())
        _log_removed("duplicates", before, len(lines))

    if shuffle_lines:
        log.info(f"\t\t- Shuffling {len(lines):,} lines...")
        random.shuffle(lines)

    _log_removed("total", total0, len(lines))
    return lines


def normalize_lines(lines, seq=None):
    if seq is None:
        seq = [NFKC(), Strip()]
    normalizer = normalizers.Sequence(seq)
    return [normalizer.normalize_str(line) for line in lines]


def preprocess_predict_file(input_file, output_file, preprocess_fn, pretokenize,
                            input_lang, vocab_lang, ds, force_overwrite):
    if not force_overwrite and os.path.exists(output_file):
        return
    lines = read_file_lines(input_file, autoclean=True)
    if preprocess_fn:
        data = {"lang": input_lang, "lines": lines}
        lines = preprocess_fn(data, ds)
    if pretokenize:
        lines = tokenizers.moses_tokenize(lines, lang=vocab_lang)
    write_file_lines(lines=lines, filename=output_file, insert_break_line=True, encoding="utf-8")
    assert os.path.exists(output_file)
