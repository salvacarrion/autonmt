"""Tokenizer primitives (Moses + SentencePiece) and their file-level wrappers.

In-memory helpers (``moses_tokenize`` / ``spm_encode`` / ``moses_detokenize`` /
``spm_decode``) operate on lists of strings. The ``*_file`` wrappers are thin
"read → transform → write" adapters used by the build pipeline.

References
----------
Koehn et al. (2007). *Moses: Open Source Toolkit for Statistical Machine
Translation.* https://aclanthology.org/P07-2045/

Sennrich, Haddow & Birch (2016). *Neural Machine Translation of Rare Words
with Subword Units* (BPE). [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)

Kudo (2018). *Subword Regularization: Improving Neural Network Translation
Models with Multiple Subword Candidates* (unigram LM).
[arXiv:1804.10959](https://arxiv.org/abs/1804.10959)

Kudo & Richardson (2018). *SentencePiece: A simple and language independent
subword tokenizer and detokenizer for Neural Text Processing.*
[arXiv:1808.06226](https://arxiv.org/abs/1808.06226)
"""
from tqdm import tqdm

from sacremoses import MosesTokenizer, MosesDetokenizer
import sentencepiece as spm

from autonmt.utils.fileio import read_file_lines, write_file_lines


# ---------------------------------------------------------------------------
# In-memory primitives
# ---------------------------------------------------------------------------

def moses_tokenize(lines, lang, show_progress=False):
    mt = MosesTokenizer(lang=lang)
    return [mt.tokenize(line, return_str=True)
            for line in tqdm(lines, total=len(lines), disable=not show_progress)]


def moses_detokenize(lines, lang, show_progress=False):
    mt = MosesDetokenizer(lang=lang)
    return [mt.detokenize(line.split())
            for line in tqdm(lines, total=len(lines), disable=not show_progress)]


def spm_encode(lines, sp, show_progress=False):
    encoded = sp.encode(lines, out_type=str)
    return [' '.join(line)
            for line in tqdm(encoded, total=len(encoded), disable=not show_progress)]


def spm_decode(lines, sp, show_progress=False):
    split = [line.split(' ')
             for line in tqdm(lines, total=len(lines), disable=not show_progress)]
    return sp.decode_pieces(split, out_type=str)


# Backwards-compatible aliases (some callers and tests still import these names).
_moses_tokenizer = moses_tokenize
_moses_detokenizer = moses_detokenize
_spm_encode = spm_encode
_spm_decode = spm_decode


# ---------------------------------------------------------------------------
# Shared file wrapper
# ---------------------------------------------------------------------------

def _process_file(input_file, output_file, transform_fn):
    lines = read_file_lines(input_file, autoclean=True)
    lines = transform_fn(lines)
    write_file_lines(lines=lines, filename=output_file, insert_break_line=True)


def moses_tokenizer_file(input_file, output_file, lang):
    _process_file(input_file, output_file,
                  lambda lines: moses_tokenize(lines, lang, show_progress=True))


def moses_detokenizer_file(input_file, output_file, lang):
    _process_file(input_file, output_file,
                  lambda lines: moses_detokenize(lines, lang, show_progress=True))


def spm_encode_file(spm_model_path, input_file, output_file):
    sp = spm.SentencePieceProcessor(model_file=spm_model_path)
    _process_file(input_file, output_file,
                  lambda lines: spm_encode(lines, sp, show_progress=True))


def spm_decode_file(spm_model_path, input_file, output_file):
    sp = spm.SentencePieceProcessor(model_file=spm_model_path)
    _process_file(input_file, output_file,
                  lambda lines: spm_decode(lines, sp, show_progress=True))


def truncate_file(input_file, output_file, max_tokens):
    _process_file(
        input_file, output_file,
        lambda lines: [" ".join(line.split(' ')[:max_tokens]).strip() for line in lines],
    )


# ---------------------------------------------------------------------------
# SentencePiece training + vocab parsing
# ---------------------------------------------------------------------------

def spm_train_file(input_file, model_prefix, subword_model, vocab_size, input_sentence_size,
                   character_coverage, split_digits, byte_fallback=False):
    # Numbers are not included in the vocabulary (...and digits are not split, even with: --split_digits)
    spm.SentencePieceTrainer.train(
        input=input_file, model_prefix=model_prefix,
        model_type=str(subword_model), vocab_size=vocab_size,
        input_sentence_size=input_sentence_size, byte_fallback=byte_fallback,
        character_coverage=character_coverage, split_digits=split_digits,
        pad_id=3,
    )


def spm_read_vocab_file(vocab_path, ignore_special_tokens=4):
    """Load an SPM ``.vocab`` file as ``{piece: log_prob}``."""
    lines = read_file_lines(vocab_path, autoclean=False)
    if ignore_special_tokens > 0:
        lines = lines[ignore_special_tokens:]
    vocab = {}
    for line in lines:
        cols = line.split('\t')
        vocab[cols[0]] = float(cols[-1].strip())
    return vocab
