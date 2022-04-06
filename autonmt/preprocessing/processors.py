import shutil
from itertools import islice

import numpy as np
import pandas as pd

from autonmt.bundle.utils import *
from autonmt.bundle import utils, plots
from autonmt.preprocessing.dataset import Dataset
from collections import Counter

from autonmt.api import py_cmd_api

from tokenizers import normalizers
from tokenizers.normalizers import NFKC


def normalize_file(input_file, output_file, normalizer, force_overwrite, limit=None):
    if force_overwrite or not os.path.exists(output_file):
        lines = read_file_lines(input_file, autoclean=True)
        lines = lines if not limit else lines[:limit]
        lines = lines if not normalizer else [normalizer.normalize_str(line) for line in lines]
        write_file_lines(lines=lines, filename=output_file, insert_break_line=True, encoding="utf-8")
        assert os.path.exists(output_file)


def pretokenize_file(input_file, output_file, lang, force_overwrite, **kwargs):
    # Tokenize
    if force_overwrite or not os.path.exists(output_file):
        py_cmd_api.moses_tokenizer(input_file=input_file, output_file=output_file, lang=lang, **kwargs)
        assert os.path.exists(output_file)


def encode_file(ds, input_file, output_file, lang, merge_vocabs, truncate_at, force_overwrite, **kwargs):
    # Check if file exists
    if force_overwrite or not os.path.exists(output_file):

        # Apply preprocessing

        # Copy file
        if ds.subword_model in {None, "none"}:
            shutil.copyfile(input_file, output_file)

        elif ds.subword_model in {"bytes"}:
            # Save file as UTF8 and make sure everything uses NFKC
            lines = read_file_lines(input_file, autoclean=True)
            lines = [NFKC().normalize_str(line) for line in lines]
            lines = [" ".join([hex(x) for x in line.encode()]) for line in lines]
            write_file_lines(lines=lines, filename=output_file, insert_break_line=True)

        else:
            # Select model
            if merge_vocabs:
                model_path = ds.get_vocab_file() + ".model"
            else:
                model_path = ds.get_vocab_file(lang=lang) + ".model"

            # Encode files
            py_cmd_api.spm_encode(spm_model_path=model_path,
                                  input_file=input_file, output_file=output_file, **kwargs)

        # Truncate if needed
        if truncate_at:
            lines = read_file_lines(output_file, autoclean=True)
            lines = [" ".join(line.split(' ')[:truncate_at]).strip() for line in lines]
            write_file_lines(lines=lines, filename=output_file, insert_break_line=True)

        # Check that the output file exist
        assert os.path.exists(output_file)


def decode_file(input_file, output_file, lang, subword_model, pretok_flag, model_vocab_path, force_overwrite,
                use_cmd, venv_path, remove_unk_hyphen=False, **kwargs):
    if force_overwrite or not os.path.exists(output_file):

        # Detokenize
        if subword_model in {None, "none"}:
            # Rename or copy files (tok==txt)
            shutil.copyfile(input_file, output_file)

        elif subword_model in {"bytes"}:
            # Decode files
            lines = read_file_lines(input_file, autoclean=True)
            lines = [clean_file_line(bytes([int(x, base=16) for x in line.split(' ')])) for line in lines]

            # Write files
            write_file_lines(lines=lines, filename=output_file, insert_break_line=True)

        else:
            # Decode files
            py_cmd_api.spm_decode(model_vocab_path + ".model", input_file=input_file, output_file=output_file,
                                  use_cmd=use_cmd, venv_path=venv_path)

            # Remove the hyphen of unknown words when needed
            if remove_unk_hyphen:
                replace_in_file('▁', ' ', output_file)

        # Detokenize with moses
        if pretok_flag:
            py_cmd_api.moses_detokenizer(input_file=output_file, output_file=output_file, lang=lang,
                                         use_cmd=use_cmd, venv_path=venv_path)


        # Check that the output file exist
        assert os.path.exists(output_file)


def decode_lines(lines, lang, subword_model, pretok_flag, model_vocab_path,  remove_unk_hyphen=False):
    # Detokenize
    if subword_model in {None, "none"}:
        # Rename or copy files (tok==txt)
        lines = lines

    elif subword_model in {"bytes"}:
        # Decode files
        lines = [utils.clean_file_line(bytes([int(x, base=16) for x in line.split(' ')])) for line in lines]

    else:
        # Decode files
        lines = py_cmd_api._spm_decode(lines, model_vocab_path + ".model")

        # Remove the hyphen of unknown words when needed
        if remove_unk_hyphen:
            lines = [line.replace('▁', ' ') for line in lines]

    # Detokenize with moses
    if pretok_flag:
        lines = py_cmd_api._moses_detokenizer(lines, lang=lang)

    return lines
