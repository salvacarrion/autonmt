"""Vocabulary-stage helpers: bytes pseudo-vocab, SPM training, frequency export.

Each public function takes a :class:`~autonmt.datasets.dataset.Dataset`
and is responsible for one disk artifact under ``vocabs/<sw>/<vs>/``. They are
called by :class:`~autonmt.datasets.dataset_builder.DatasetBuilder` but kept here
to keep the builder small.
"""
import os
import random
from collections import Counter

from autonmt.utils.enums import SubwordModel
from autonmt.utils.fileio import read_file_lines, write_file_lines
from autonmt.utils.logger import get_logger
from autonmt.datasets.stats import build_counter_low_mem, norm_counter
from autonmt.datasets import tokenizers

log = get_logger(__name__)


# Special tokens written for the synthetic "bytes" vocab. Kept in sync with
# Vocabulary's defaults (unk/sos/eos/pad) so the layout matches what SPM emits.
_BYTES_SPECIAL_TOKENS = ("<unk>", "<s>", "</s>", "<pad>")


def _lang_files(ds):
    """Vocab filename stems: one combined entry when merge_vocabs, else one per language."""
    if ds.merge_vocabs:
        return [f"{ds.src_lang}-{ds.tgt_lang}"]
    return [ds.src_lang, ds.tgt_lang]


def write_bytes_vocab(ds, force_overwrite):
    """Synthesize a fixed 256-byte vocab on disk (no SPM training needed)."""
    tokens = [f'0x{byte:02x}' for byte in range(256)]
    lines = [f"{tok}\t0" for tok in (*_BYTES_SPECIAL_TOKENS, *tokens)]
    for ext in _lang_files(ds):
        output_file = ds.get_vocab_file(lang=ext) + ".vocab"
        if force_overwrite or not os.path.exists(output_file):
            write_file_lines(lines, filename=output_file, insert_break_line=True)


def train_spm(ds, force_overwrite, input_sentence_size, character_coverage, split_digits):
    """Train one SentencePiece model per language (or one combined if merge_vocabs)."""
    src_train = _train_input_path(ds, ds.src_lang)
    tgt_train = _train_input_path(ds, ds.tgt_lang)

    if ds.merge_vocabs:
        merged_path = os.path.join(ds.get_vocab_path(base=True), "_tmp",
                                   f"{ds.train_name}.{ds.src_lang}-{ds.tgt_lang}")
        os.makedirs(os.path.dirname(merged_path), exist_ok=True)
        _concat_for_spm(src_train, tgt_train, merged_path, force_overwrite)
        files = [(merged_path, f"{ds.src_lang}-{ds.tgt_lang}")]
    else:
        files = [(src_train, ds.src_lang), (tgt_train, ds.tgt_lang)]

    for input_file, ext in files:
        output_file = ds.get_vocab_file(lang=ext)  # without extension
        if force_overwrite or not os.path.exists(f"{output_file}.model"):
            tokenizers.spm_train_file(
                input_file=input_file, model_prefix=output_file,
                subword_model=ds.subword_model, vocab_size=ds.vocab_size,
                input_sentence_size=input_sentence_size,
                character_coverage=character_coverage, split_digits=split_digits,
                byte_fallback=ds.byte_fallback,
            )
            assert os.path.exists(f"{output_file}.model")


def _train_input_path(ds, lang):
    path_fn = ds.get_pretok_path if ds.pretok_flag else ds.get_splits_auto_path
    return path_fn(fname=f"{ds.train_name}.{lang}")


def _concat_for_spm(src_path, tgt_path, out_path, force_overwrite):
    if not force_overwrite and os.path.exists(out_path):
        return
    lines = read_file_lines(src_path, autoclean=True) + read_file_lines(tgt_path, autoclean=True)
    # spm_train_file loads the first N lines of the corpus by default; shuffle so
    # the trained vocab isn't biased toward one language's prefix.
    random.shuffle(lines)
    write_file_lines(lines=lines, filename=out_path, insert_break_line=True)


def export_frequencies(ds, force_overwrite, normalize_freq=False):
    """Write one ``.vocabf`` file per language with token frequencies (for plotting).

    SentencePiece can leave words in its vocab that never appear during the
    encoded-training-set tokenisation; we count only tokens that *do* appear in
    the SPM vocab so the frequencies match what the model actually sees.
    """
    files = _lang_files(ds)
    vocab_paths = [ds.get_vocab_path(fname=f) + ".vocabf" for f in files]
    if not force_overwrite and all(os.path.exists(f) for f in vocab_paths):
        return

    src_counter = build_counter_low_mem(
        ds.get_encoded_path(f"{ds.train_name}.{ds.src_lang}"),
        split_fn=lambda x: x.split(' '))
    tgt_counter = build_counter_low_mem(
        ds.get_encoded_path(f"{ds.train_name}.{ds.tgt_lang}"),
        split_fn=lambda x: x.split(' '))

    uses_spm = ds.subword_model is not SubwordModel.BYTES
    if not uses_spm:
        counters = ([src_counter + tgt_counter] if ds.merge_vocabs
                    else [src_counter, tgt_counter])
    else:
        per_lang = ([(src_counter + tgt_counter, f"{ds.src_lang}-{ds.tgt_lang}")]
                    if ds.merge_vocabs else [(src_counter, ds.src_lang),
                                             (tgt_counter, ds.tgt_lang)])
        counters = []
        for counter, lang_file in per_lang:
            spm_vocab = tokenizers.spm_read_vocab_file(
                vocab_path=ds.get_vocab_path(fname=lang_file) + ".vocab")
            counters.append(Counter({k: v for k, v in counter.items() if k in spm_vocab}))

    for counter, vocab_path in zip(counters, vocab_paths):
        if normalize_freq:
            counter = norm_counter(counter)
        if force_overwrite or not os.path.exists(vocab_path):
            lines = [f"{tok}\t{freq}" for tok, freq in counter.most_common()]
            write_file_lines(lines=lines, filename=vocab_path, insert_break_line=True)
