"""Subword-dependent file ops: Moses pretokenization, SPM/bytes encode, decode.

These functions produce ``3_pretokenized/`` and ``4_encoded/<subword>/<vocab>/``
and vary with the subword model choice. For ``word`` runs Moses pretokenization
*is* the encoding step (``encode_file`` then just copies the file). Subword-
agnostic preparation lives in :mod:`autonmt.datasets.preprocessing`.
"""
import os
import shutil

from tokenizers.normalizers import NFKC

from autonmt.utils.enums import has_vocab, is_bytes_only
from autonmt.utils.fileio import read_file_lines, write_file_lines, text2hex
from autonmt.datasets import tokenizers


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
