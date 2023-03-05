from tqdm import tqdm

from sacremoses import MosesTokenizer, MosesDetokenizer
import sentencepiece as spm

from autonmt.bundle import utils


def _moses_tokenizer(lines, lang):
    mt = MosesTokenizer(lang=lang)
    return [mt.tokenize(line, return_str=True) for line in tqdm(lines, total=len(lines))]

def _moses_detokenizer(lines, lang):
    mt = MosesDetokenizer(lang=lang)
    return [mt.detokenize(line.split()) for line in tqdm(lines, total=len(lines))]

def _spm_encode(lines, model_file):
    sp = spm.SentencePieceProcessor(model_file=model_file)
    lines = sp.encode(lines, out_type=str)
    lines = [' '.join(line) for line in tqdm(lines, total=len(lines))]
    return lines

def _spm_decode(lines, model_file):
    sp = spm.SentencePieceProcessor(model_file=model_file)
    lines = sp.encode(lines, out_type=str)
    lines = [line.split(' ') for line in tqdm(lines, total=len(lines))]
    return lines

def moses_tokenizer_file(input_file, output_file, lang):
    # Read, tokenizer and write lines
    lines = utils.read_file_lines(input_file, autoclean=True)
    lines = _moses_tokenizer(lines, lang)
    utils.write_file_lines(lines=lines, filename=output_file, insert_break_line=True)

def moses_detokenizer_file(input_file, output_file, lang):
    # Read, detokenizer and write lines
    lines = utils.read_file_lines(input_file, autoclean=True)
    lines = _moses_detokenizer(lines, lang)
    utils.write_file_lines(lines=lines, filename=output_file, insert_break_line=True)

def spm_train_file(input_file, model_prefix, subword_model, vocab_size, input_sentence_size, character_coverage, split_digits):
    # Enable
    byte_fallback = False
    if "+bytes" in subword_model:
        subword_model = subword_model.replace("+bytes", "")
        byte_fallback = True

    # Numbers are not included in the vocabulary (...and digits are not split, even with: --split_digits)
    spm.SentencePieceTrainer.train(input=input_file, model_prefix=model_prefix,
                                   model_type=subword_model, vocab_size=vocab_size,
                                   input_sentence_size=input_sentence_size, byte_fallback=byte_fallback,
                                   character_coverage=character_coverage, split_digits=split_digits,
                                   pad_id=3)


def spm_encode_file(spm_model_path, input_file, output_file):
    # Read, encode and write lines
    lines = utils.read_file_lines(input_file, autoclean=True)
    lines = _spm_encode(lines, spm_model_path)
    utils.write_file_lines(lines=lines, filename=output_file, insert_break_line=True)


def spm_decode_file(spm_model_path, input_file, output_file):
    # Read, decode and write lines
    lines = utils.read_file_lines(input_file, autoclean=True)
    lines = _spm_decode(lines, spm_model_path)
    utils.write_file_lines(lines=lines, filename=output_file, insert_break_line=True)


def truncate_file(input_file, output_file, max_tokens):
    lines = utils.read_file_lines(input_file, autoclean=True)
    lines = [" ".join(line.split(' ')[:max_tokens]).strip() for line in lines]
    utils.write_file_lines(lines=lines, filename=output_file, insert_break_line=True)

def smp_read_vocab_file(vocab_path, ignore_special_tokens=4):
    # Get the exact vocab from SPM
    spm_vocab_lines = utils.read_file_lines(vocab_path, autoclean=False)

    # Ignore special tokens
    if ignore_special_tokens > 0:
        spm_vocab_lines = spm_vocab_lines[ignore_special_tokens:]

    # Parse vocab
    spm_vocab = {}
    for line in spm_vocab_lines:
        cols = line.split('\t')
        spm_vocab[cols[0]] = int(cols[-1].strip())  # word -> id
    return spm_vocab