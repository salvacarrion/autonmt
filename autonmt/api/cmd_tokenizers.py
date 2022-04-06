import subprocess

from autonmt.api import NO_VENV_MSG


def cmd_spm_encode(model_path, input_file, output_file, venv_path=None):
    print("\t- [INFO]: Using 'SentencePiece' from the command line.")

    env = f"{venv_path}" if venv_path else NO_VENV_MSG
    cmd = f"spm_encode --model={model_path} --output_format=piece < {input_file} > {output_file}"  # --vocabulary={model_path}.vocab --vocabulary_threshold={min_vocab_frequency}
    subprocess.call(['/bin/bash', '-c', f"{env} && {cmd}"])
    return cmd


def cmd_spm_decode(model_path, input_file, output_file, venv_path=None):
    print("\t- [INFO]: Using 'SentencePiece' from the command line.")

    env = f"{venv_path}" if venv_path else NO_VENV_MSG
    cmd = f"spm_decode --model={model_path} --input_format=piece < {input_file} > {output_file}"
    subprocess.call(['/bin/bash', '-c', f"{env} && {cmd}"])
    return cmd


def cmd_spm_train(input_file, model_prefix, subword_model, vocab_size, input_sentence_size, character_coverage, byte_fallback, venv_path=None):
    print("\t- [INFO]: Using 'SentencePiece' from the command line.")

    # Add extra options
    extra = ""
    extra += " --byte_fallback" if byte_fallback else ""
    extra += f" --character_coverage={character_coverage}" if character_coverage else ""

    # Numbers are not included in the vocabulary (...and digits are not split, even with: --split_digits)
    # https://github.com/google/sentencepiece/blob/master/doc/options.md
    env = f"{venv_path}" if venv_path else NO_VENV_MSG
    cmd = f"spm_train --input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type={subword_model} --input_sentence_size={input_sentence_size} --pad_id=3 {extra}"
    subprocess.call(['/bin/bash', '-c', f"{env} && {cmd}"])
    return cmd


def cmd_moses_tokenizer(input_file, output_file, lang, venv_path=None):
    print("\t- [INFO]: Using 'Sacremoses' from the command line.")

    env = f"{venv_path}" if venv_path else NO_VENV_MSG
    cmd = f"sacremoses -l {lang} -j$(nproc) tokenize < {input_file} > {output_file}"
    subprocess.call(['/bin/bash', '-c', f"{env} && {cmd}"])
    return cmd


def cmd_moses_detokenizer(input_file, output_file, lang, venv_path=None):
    print("\t- [INFO]: Using 'Sacremoses' from the command line.")

    env = f"{venv_path}" if venv_path else NO_VENV_MSG
    cmd = f"sacremoses -l {lang} -j$(nproc) detokenize < {input_file} > {output_file}"
    subprocess.call(['/bin/bash', '-c', f"{env} && {cmd}"])
    return cmd

