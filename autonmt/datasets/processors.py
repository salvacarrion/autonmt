"""Text-level processors invoked by the builder and at predict time.

Two layers:

  * ``preprocess_pairs`` / ``preprocess_lines`` — opinionated cleaning pipelines
    composed from small filters (length, dedupe, length-ratio, shuffle).
  * ``pretokenize_file`` / ``encode_file`` / ``decode_file`` — file-level
    wrappers that delegate to :mod:`autonmt.datasets.tokenizers`.
"""
import collections
import os
import random
import shutil

import numpy as np

from tokenizers import normalizers
from tokenizers.normalizers import NFKC, Strip

from autonmt.utils.enums import has_vocab, is_bytes_only
from autonmt.utils.fileio import read_file_lines, write_file_lines, text2hex
from autonmt.utils.logger import get_logger
from autonmt.datasets.stats import shuffle_in_order
from autonmt.datasets import tokenizers

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Cleaning pipelines
# ---------------------------------------------------------------------------

def _log_removed(label, before, after):
    pct = (1 - after / before) * 100 if before else 0
    log.info(f"\t\t\t- Removed {before - after:,} lines ({label}; {pct:.3f}%)")


def _filter_lengths_pairs(src_lines, tgt_lines, min_len, max_len, max_len_percentile):
    min_len = 0 if min_len is None else min_len
    max_len = float('inf') if max_len is None else max_len
    max_len_src = max_len_trg = max_len

    if max_len_percentile and max_len_percentile < 100:
        src_lengths, trg_lengths = zip(*[(len(s), len(t)) for s, t in zip(src_lines, tgt_lines)])
        max_len_src = np.percentile(np.array(src_lengths), max_len_percentile)
        max_len_trg = np.percentile(np.array(trg_lengths), max_len_percentile)

    log.info("\t\t- Checking lengths...")
    before = len(src_lines)
    pairs = [(s, t) for s, t in zip(src_lines, tgt_lines)
             if min_len <= len(s) <= max_len_src and min_len <= len(t) <= max_len_trg]
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


# ---------------------------------------------------------------------------
# File-level wrappers used during the build / predict pipeline
# ---------------------------------------------------------------------------

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


def pretokenize_file(input_file, output_file, lang, force_overwrite, **kwargs):
    if force_overwrite or not os.path.exists(output_file):
        tokenizers.moses_tokenizer_file(input_file=input_file, output_file=output_file, lang=lang)
        assert os.path.exists(output_file)


def encode_file(input_file, output_file, model_vocab_path, subword_model,
                force_overwrite, **kwargs):
    if not force_overwrite and os.path.exists(output_file):
        return

    if not has_vocab(subword_model) and not is_bytes_only(subword_model):
        shutil.copyfile(input_file, output_file)
    elif is_bytes_only(subword_model):
        # Save file as UTF8 and make sure everything uses NFKC
        lines = read_file_lines(input_file, autoclean=True)
        lines = [NFKC().normalize_str(line) for line in lines]
        lines = [text2hex(line, return_str=True) for line in lines]
        write_file_lines(lines=lines, filename=output_file, insert_break_line=True)
    else:
        tokenizers.spm_encode_file(spm_model_path=model_vocab_path,
                                   input_file=input_file, output_file=output_file)

    assert os.path.exists(output_file)


def decode_file(input_file, output_file, lang, subword_model, pretok_flag,
                model_vocab_path, force_overwrite, remove_unk_hyphen=False, **kwargs):
    if not force_overwrite and os.path.exists(output_file):
        return

    if not has_vocab(subword_model):
        # Rename or copy files (tok==txt) — applies to None and bytes alike.
        shutil.copyfile(input_file, output_file)
    else:
        tokenizers.spm_decode_file(model_vocab_path, input_file=input_file, output_file=output_file)

    if pretok_flag:
        tokenizers.moses_detokenizer_file(input_file=output_file, output_file=output_file, lang=lang)

    assert os.path.exists(output_file)


def decode_lines(lines, lang, subword_model, pretok_flag, spm_model=None):
    if has_vocab(subword_model):
        lines = tokenizers.spm_decode(lines, spm_model)
    if pretok_flag:
        lines = tokenizers.moses_detokenize(lines, lang=lang)
    return lines
