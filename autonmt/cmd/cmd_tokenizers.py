import subprocess
from autonmt.cmd import NO_CONDA_MSG


def cmd_spm_encode(model_path, input_file, output_file, conda_env_name=None):
    print("\t- [INFO]: Using 'SentencePiece' from the command line.")

    env = f"conda activate {conda_env_name}" if conda_env_name else NO_CONDA_MSG
    cmd = f"spm_encode --model={model_path} --output_format=piece < {input_file} > {output_file}"  # --vocabulary={model_path}.vocab --vocabulary_threshold={min_vocab_frequency}
    subprocess.call(['/bin/bash', '-i', '-c', f"{env} && {cmd}"])
    return cmd


def cmd_spm_decode(model_path, input_file, output_file, conda_env_name=None):
    print("\t- [INFO]: Using 'SentencePiece' from the command line.")

    env = f"conda activate {conda_env_name}" if conda_env_name else NO_CONDA_MSG
    cmd = f"spm_decode --model={model_path} --input_format=piece < {input_file} > {output_file}"
    subprocess.call(['/bin/bash', '-i', '-c', f"{env} && {cmd}"])
    return cmd


def cmd_spm_train(input_file, model_prefix, subword_model, vocab_size, input_sentence_size, conda_env_name=None):
    print("\t- [INFO]: Using 'SentencePiece' from the command line.")

    # https://github.com/google/sentencepiece/blob/master/doc/options.md
    env = f"conda activate {conda_env_name}" if conda_env_name else NO_CONDA_MSG
    cmd = f"spm_train --input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type={subword_model} --input_sentence_size={input_sentence_size} --pad_id=3"
    subprocess.call(['/bin/bash', '-i', '-c', f"{env} && {cmd}"])
    return cmd


def cmd_moses_tokenizer(input_file, output_file, lang, conda_env_name=None):
    print("\t- [INFO]: Using 'Sacremoses' from the command line.")

    env = f"conda activate {conda_env_name}" if conda_env_name else NO_CONDA_MSG
    cmd = f"sacremoses -l {lang} -j$(nproc) tokenize < {input_file} > {output_file}"
    subprocess.call(['/bin/bash', '-i', '-c', f"{env} && {cmd}"])
    return cmd


def cmd_moses_detokenizer(lang, input_file, output_file, conda_env_name=None):
    print("\t- [INFO]: Using 'Sacremoses' from the command line.")

    env = f"conda activate {conda_env_name}" if conda_env_name else NO_CONDA_MSG
    cmd = f"sacremoses -l {lang} -j$(nproc) detokenize < {input_file} > {output_file}"
    subprocess.call(['/bin/bash', '-i', '-c', f"{env} && {cmd}"])
    return cmd

