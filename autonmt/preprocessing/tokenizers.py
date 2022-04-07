from tqdm import tqdm

from sacremoses import MosesTokenizer, MosesDetokenizer
import sentencepiece as spm

from autonmt.bundle import utils


def moses_tokenizer(input_file, output_file, lang):
    # Read lines
    lines = utils.read_file_lines(input_file, autoclean=True)

    # Tokenize
    mt = MosesTokenizer(lang=lang)
    lines = [mt.tokenize(line, return_str=True) for line in tqdm(lines, total=len(lines))]

    # Save file
    utils.write_file_lines(lines=lines, filename=output_file, insert_break_line=True)


def moses_detokenizer(input_file, output_file, lang):
    # Read lines
    lines = utils.read_file_lines(input_file, autoclean=True)

    # Detokenize
    mt = MosesDetokenizer(lang=lang)
    lines = [mt.detokenize(line.split()) for line in tqdm(lines, total=len(lines))]

    # Save file
    utils.write_file_lines(lines=lines, filename=output_file, insert_break_line=True)


def _moses_detokenizer(lines, lang):
    mt = MosesDetokenizer(lang=lang)
    return [mt.detokenize(line.split()) for line in lines]


def spm_train(input_file, model_prefix, subword_model, vocab_size, input_sentence_size, character_coverage):
    # Enable
    byte_fallback = False
    if "+bytes" in subword_model:
        subword_model = subword_model.replace("+bytes", "")
        byte_fallback = True

    # Numbers are not included in the vocabulary (...and digits are not split, even with: --split_digits)
    spm.SentencePieceTrainer.train(input=input_file, model_prefix=model_prefix,
                                   model_type=subword_model, vocab_size=vocab_size,
                                   input_sentence_size=input_sentence_size, byte_fallback=byte_fallback,
                                   character_coverage=character_coverage,
                                   pad_id=3)


def spm_encode(spm_model_path, input_file, output_file):
    # Read lines
    lines = utils.read_file_lines(input_file, autoclean=True)

    # Encode
    s = spm.SentencePieceProcessor(model_file=spm_model_path)
    lines = [' '.join(s.encode(line, out_type=str)) for line in tqdm(lines, total=len(lines))]

    # Save file
    utils.write_file_lines(lines=lines, filename=output_file, insert_break_line=True)


def spm_decode(spm_model_path, input_file, output_file):
    # Read lines
    lines = utils.read_file_lines(input_file, autoclean=True)

    # Decode
    s = spm.SentencePieceProcessor(model_file=spm_model_path)
    lines = [s.decode_pieces(line.split(' ')) for line in tqdm(lines, total=len(lines))]

    # Save file
    utils.write_file_lines(lines=lines, filename=output_file, insert_break_line=True)


def _spm_decode(lines, spm_model_path):
    s = spm.SentencePieceProcessor(model_file=spm_model_path)
    return [s.decode_pieces(line.split(' ')) for line in lines]





