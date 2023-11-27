import collections
import numpy as np

from tokenizers import normalizers
from tokenizers.normalizers import NFKC, Strip, Lowercase

from autonmt.preprocessing import tokenizers
from autonmt.bundle import utils
from autonmt.bundle.utils import *


def preprocess_pairs(src_lines, tgt_lines, normalize_fn=None, min_len=None, max_len=None, max_len_percentile=None,
                     remove_duplicates=False, max_len_ratio_percentile=100, safe_len_ratio=2.0, shuffle_lines=False):
    assert len(src_lines) == len(tgt_lines)
    total_lines0 = len(src_lines)

    # Normalize lines: lowercase, strip, NFKC
    if normalize_fn:
        print(f"\t\t- Normalizing {len(src_lines):,} pairs...")
        src_lines = normalize_fn(src_lines)
        tgt_lines = normalize_fn(tgt_lines)

    # Remove source and target lines that are too long or too short
    # Set default values
    if min_len is not None or max_len is not None or max_len_percentile is not None:
        min_len = 0 if min_len is None else min_len
        max_len = float('inf') if max_len is None else max_len
        min_len_src = min_len_trg = min_len
        max_len_src = max_len_trg = max_len

        # Compute percentiles
        if max_len_percentile and max_len_percentile < 100:
            src_lengths, trg_lengths = list(zip(*[(len(src), len(trg)) for (src, trg) in zip(src_lines, tgt_lines)]))
            max_len_src = np.percentile(np.array(src_lengths), max_len_ratio_percentile)
            max_len_trg = np.percentile(np.array(trg_lengths), max_len_ratio_percentile)

        print("\t\t- Checking lengths...")
        total_lines = len(src_lines)
        src_lines, tgt_lines = zip(*[(src, tgt) for src, tgt in zip(src_lines, tgt_lines) if
                                     min_len_src <= len(src) <= max_len_src and
                                     min_len_trg <= len(tgt) <= max_len_trg])
        assert len(src_lines) == len(tgt_lines)
        print(f"\t\t\t- Removed {total_lines - len(src_lines):,} lines with invalid lengths")

    # Remove duplicate pairs of source and target lines
    if remove_duplicates:
        print("\t\t- Removing duplicates...")
        total_lines = len(src_lines)
        src_lines, tgt_lines = list(zip(*[item for item, count in collections.Counter(zip(src_lines, tgt_lines)).items()]))
        assert len(src_lines) == len(tgt_lines)
        print(f"\t\t\t- Removed {total_lines - len(src_lines):,} duplicate lines")

    # Remove language pairs that differ too much in length
    if max_len_ratio_percentile and max_len_ratio_percentile < 100:  # Percentile 99.95 -> 2.5 x src_len (aprox.)
        print("\t\t- Removing pairs whose length ratios differ too much...")
        total_lines = len(src_lines)
        diff_ratios = np.array([max(len(src), len(trg)) / min(len(src), len(trg)) for (src, trg) in zip(src_lines, tgt_lines)])
        threshold = np.percentile(diff_ratios, max_len_ratio_percentile)
        if threshold < safe_len_ratio:  # Do not remove pairs below this threshold
            print(f"\t\t\t- Percentile threshold overruled (safe threshold: {safe_len_ratio:.2f})")
            threshold = max(threshold, safe_len_ratio)
        print("\t\t\t- Threshold: {:.2f} (percentile: {:.2f})".format(threshold, max_len_ratio_percentile))
        src_lines, tgt_lines = list(zip(*[(src, trg) for (src, trg) in zip(src_lines, tgt_lines) if max(len(src), len(trg))/min(len(src), len(trg)) <= threshold]))
        assert len(src_lines) == len(tgt_lines)
        print(f"\t\t\t- Removed {total_lines - len(src_lines):,} lines due to the length difference")

    # Shuffle lines
    if shuffle_lines:
        print(f"\t\t- Shuffling {len(src_lines):,} pairs...")
        src_lines, tgt_lines = shuffle_in_order(src_lines, tgt_lines)

    # Summary
    print(f"\t\t- Total lines removed {total_lines0-len(src_lines):,} ({1-len(src_lines)/total_lines0:.3f}%)")
    return src_lines, tgt_lines

def preprocess_lines(lines, normalize_fn=None, min_len=None, max_len=None, remove_duplicates=False, shuffle_lines=False):
    total_lines0 = len(lines)

    # Set default values
    min_len = 0 if min_len is None else min_len
    max_len = float('inf') if max_len is None else max_len

    # Normalize lines: lowercase, strip, NFKC
    if normalize_fn:
        print(f"\t\t- Normalizing {len(lines):,} lines...")
        lines = normalize_fn(lines)

    # Remove source and target lines that are too long or too short
    if min_len is not None or max_len is not None:
        print("\t\t- Checking lengths...")
        total_lines = len(lines)
        lines = [l for l in lines if min_len <= len(l) <= max_len]
        print(f"\t\t\t- Removed {total_lines - len(lines):,} lines with invalid lengths")

    # Remove duplicate pairs of source and target lines
    if remove_duplicates:
        print("\t\t- Removing duplicates...")
        total_lines = len(lines)
        lines = [item for item, count in collections.Counter(lines).items()]
        print(f"\t\t\t- Removed {total_lines - len(lines):,} duplicate lines")

    # Shuffle lines
    if shuffle_lines:
        print(f"\t\t- Shuffling {len(lines):,} lines...")
        random.shuffle(lines)

    # Summary
    print(f"\t\t- Total lines removed {total_lines0-len(lines):,} ({1-len(lines)/total_lines0:.3f}%)")
    return lines

def normalize_lines(lines, seq=None):
    # Default sequence
    if seq is None:
        seq = [NFKC(), Strip()]

    # Normalize lines
    normalizer = normalizers.Sequence(seq)
    lines = [normalizer.normalize_str(line) for line in lines]
    return lines

def preprocess_predict_file(input_file, output_file, preprocess_fn, pretokenize, input_lang, vocab_lang, ds, force_overwrite):
    if force_overwrite or not os.path.exists(output_file):
        lines = read_file_lines(input_file, autoclean=True)

        # preprocess_fn
        if preprocess_fn:
            data = {"lang": input_lang, "lines": lines}
            lines = preprocess_fn(data, ds)

        # Pretokenize
        if pretokenize:
            lines = tokenizers._moses_tokenizer(lines, lang=vocab_lang)

        write_file_lines(lines=lines, filename=output_file, insert_break_line=True, encoding="utf-8")
        assert os.path.exists(output_file)

def pretokenize_file(input_file, output_file, lang, force_overwrite, **kwargs):
    # Tokenize
    if force_overwrite or not os.path.exists(output_file):
        tokenizers.moses_tokenizer_file(input_file=input_file, output_file=output_file, lang=lang)
        assert os.path.exists(output_file)


def encode_file(input_file, output_file, model_vocab_path, subword_model, force_overwrite, **kwargs):
    # Check if file exists
    if force_overwrite or not os.path.exists(output_file):

        # Copy file
        if subword_model in {None, "none"}:
            shutil.copyfile(input_file, output_file)

        elif subword_model in {"bytes"}:
            # Save file as UTF8 and make sure everything uses NFKC
            lines = read_file_lines(input_file, autoclean=True)
            lines = [NFKC().normalize_str(line) for line in lines]
            lines = [" ".join([hex(x) for x in line.encode()]) for line in lines]
            write_file_lines(lines=lines, filename=output_file, insert_break_line=True)

        else:

            # Encode files
            tokenizers.spm_encode_file(spm_model_path=model_vocab_path, input_file=input_file, output_file=output_file)

        # Check that the output file exist
        assert os.path.exists(output_file)


def decode_file(input_file, output_file, lang, subword_model, pretok_flag, model_vocab_path, force_overwrite,
                remove_unk_hyphen=False, **kwargs):
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
            tokenizers.spm_decode_file(model_vocab_path, input_file=input_file, output_file=output_file)

            # Remove the hyphen of unknown words when needed
            if remove_unk_hyphen:
                replace_in_file('▁', ' ', output_file)

        # Detokenize with moses
        if pretok_flag:
            tokenizers.moses_detokenizer_file(input_file=output_file, output_file=output_file, lang=lang)

        # Check that the output file exist
        assert os.path.exists(output_file)


def decode_lines(lines, lang, subword_model, pretok_flag, spm_model=None, remove_unk_hyphen=False):
    # Detokenize
    if subword_model in {None, "none"}:
        # Rename or copy files (tok==txt)
        lines = lines

    elif subword_model in {"bytes"}:
        # Decode files
        lines = [utils.clean_file_line(bytes([int(x, base=16) for x in line.split(' ')])) for line in lines]
    else:
        # Decode files
        lines = tokenizers._spm_decode(lines, spm_model)

        # Remove the hyphen of unknown words when needed
        if remove_unk_hyphen:
            lines = [line.replace('▁', ' ') for line in lines]

    # Detokenize with moses
    if pretok_flag:
        lines = tokenizers._moses_detokenizer(lines, lang=lang)

    return lines
