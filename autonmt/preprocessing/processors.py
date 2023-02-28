import shutil

from tokenizers import normalizers
from tokenizers.normalizers import NFKC, Strip, Lowercase

from autonmt.preprocessing import tokenizers
from autonmt.bundle import utils
from autonmt.bundle.utils import *


def preprocess_pairs(src_lines, tgt_lines, normalize_fn=None, min_len=None, max_len=None, remove_duplicates=False,
                     shuffle_lines=False):
    assert len(src_lines) == len(tgt_lines)

    # Set default values
    min_len = 0 if min_len is None else min_len
    max_len = float('inf') if max_len is None else max_len
    if remove_duplicates:
        print("\t\t- [WARNING] Removing duplicates will lose the order of the lines")

    # Normalize lines: lowercase, strip, NFKC
    if normalize_fn:
        print("\t\t- Normalizing lines...")
        src_lines = normalize_fn(src_lines)
        tgt_lines = normalize_fn(tgt_lines)

    # Remove source and target lines that are too long or too short
    if min_len is not None or max_len is not None:
        print("\t\t- Checking lengths...")
        total_lines = len(src_lines)
        src_lines, tgt_lines = zip(*[(src, tgt) for src, tgt in zip(src_lines, tgt_lines) if
                                     min_len <= len(src) <= max_len and
                                     min_len <= len(tgt) <= max_len])
        assert len(src_lines) == len(tgt_lines)
        print("\t\t- Removed {} lines with invalid lengths".format(total_lines - len(src_lines)))

    # Remove duplicate pairs of source and target lines
    if remove_duplicates:
        print("\t\t- Removing duplicates... (order is lost)")
        total_lines = len(src_lines)
        src_lines, tgt_lines = zip(*set(zip(src_lines, tgt_lines)))
        assert len(src_lines) == len(tgt_lines)
        print("\t\t- Removed {} duplicate lines".format(total_lines - len(src_lines)))

    # Shuffle lines
    if shuffle_lines:
        src_lines, tgt_lines = shuffle_in_order(src_lines, tgt_lines)

    return src_lines, tgt_lines

def preprocess_lines(lines, normalize_fn=None, min_len=None, max_len=None, remove_duplicates=False, shuffle_lines=False):

    # Set default values
    min_len = 0 if min_len is None else min_len
    max_len = float('inf') if max_len is None else max_len
    if remove_duplicates:
        print("\t\t- [WARNING] Removing duplicates will lose the order of the lines")

    # Normalize lines: lowercase, strip, NFKC
    if normalize_fn:
        print("\t\t- Normalizing lines...")
        lines = normalize_fn(lines)

    # Remove source and target lines that are too long or too short
    if min_len is not None or max_len is not None:
        print("\t\t- Checking lengths...")
        total_lines = len(lines)
        lines = [l for l in lines if min_len <= len(l) <= max_len]
        print("\t\t- Removed {} lines with invalid lengths".format(total_lines - len(lines)))

    # Remove duplicate pairs of source and target lines
    if remove_duplicates:
        print("\t\t- Removing duplicates... (order is lost)")
        total_lines = len(lines)
        lines = list(set(lines))
        print("\t\t- Removed {} duplicate lines".format(total_lines - len(lines)))

    # Shuffle lines
    if shuffle_lines:
        random.shuffle(lines)

    return lines
def normalize_lines(lines):
    normalizer = normalizers.Sequence([NFKC(), Strip()])
    lines = [normalizer.normalize_str(line) for line in lines]
    return lines

def preprocess_predict_file(input_file, output_file, preprocess_fn, pretokenize, lang, force_overwrite):
    if force_overwrite or not os.path.exists(output_file):
        lines = read_file_lines(input_file, autoclean=True)

        # preprocess_fn
        if preprocess_fn:
            lines = preprocess_fn(lines)

        # Pretokenize
        if pretokenize:
            lines = tokenizers._moses_tokenizer(lines, lang=lang)

        write_file_lines(lines=lines, filename=output_file, insert_break_line=True, encoding="utf-8")
        assert os.path.exists(output_file)

def pretokenize_file(input_file, output_file, lang, force_overwrite, **kwargs):
    # Tokenize
    if force_overwrite or not os.path.exists(output_file):
        tokenizers.moses_tokenizer(input_file=input_file, output_file=output_file, lang=lang)
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
            tokenizers.spm_encode(spm_model_path=model_path, input_file=input_file, output_file=output_file)

        # Truncate if needed
        if truncate_at:
            lines = read_file_lines(output_file, autoclean=True)
            lines = [" ".join(line.split(' ')[:truncate_at]).strip() for line in lines]
            write_file_lines(lines=lines, filename=output_file, insert_break_line=True)

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
            tokenizers.spm_decode(model_vocab_path + ".model", input_file=input_file, output_file=output_file)

            # Remove the hyphen of unknown words when needed
            if remove_unk_hyphen:
                replace_in_file('▁', ' ', output_file)

        # Detokenize with moses
        if pretok_flag:
            tokenizers.moses_detokenizer(input_file=output_file, output_file=output_file, lang=lang)

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
        lines = tokenizers._spm_decode(lines, model_vocab_path + ".model")

        # Remove the hyphen of unknown words when needed
        if remove_unk_hyphen:
            lines = [line.replace('▁', ' ') for line in lines]

    # Detokenize with moses
    if pretok_flag:
        lines = tokenizers._moses_detokenizer(lines, lang=lang)

    return lines
