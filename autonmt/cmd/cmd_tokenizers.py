import subprocess


def spm_encode(spm_model_path, input_file, output_file, conda_env_name=None):
    env = f"conda activate {conda_env_name}" if conda_env_name else "echo \"No conda env. Using '/bin/bash'\""
    cmd = f"spm_encode --model={spm_model_path} --output_format=piece < {input_file} > {output_file}"  # --vocabulary={spm_model_path}.vocab --vocabulary_threshold={min_vocab_frequency}
    subprocess.call(['/bin/bash', '-i', '-c', f"{env} && {cmd}"])
    print(f"\t- Command used: {cmd}")


def spm_decode(spm_model_path, input_file, output_file, conda_env_name=None):
    env = f"conda activate {conda_env_name}" if conda_env_name else "echo \"No conda env. Using '/bin/bash'\""
    cmd = f"spm_decode --model={spm_model_path} --input_format=piece < {input_file} > {output_file}"
    subprocess.call(['/bin/bash', '-i', '-c', f"{env} && {cmd}"])
    print(f"\t- Command used: {cmd}")


def spm_train(input_file, model_prefix, vocab_size, character_coverage, subword_model, input_sentence_size=1000000, conda_env_name=None):
    # https://github.com/google/sentencepiece/blob/master/doc/options.md
    env = f"conda activate {conda_env_name}" if conda_env_name else "echo \"No conda env. Using '/bin/bash'\""
    cmd = f"spm_train --input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --character_coverage={character_coverage} --model_type={subword_model} --input_sentence_size={input_sentence_size} --pad_id=3"
    subprocess.call(['/bin/bash', '-i', '-c', f"{env} && {cmd}"])
    print(f"\t- Command used: {cmd}")


def moses_tokenizer(lang, input_file, output_file, conda_env_name=None):
    env = f"conda activate {conda_env_name}" if conda_env_name else "echo \"No conda env. Using '/bin/bash'\""
    cmd = f"sacremoses -l {lang} -j$(nproc) tokenize < {input_file} > {output_file}"
    subprocess.call(['/bin/bash', '-i', '-c', f"{env} && {cmd}"])
    print(f"\t- Command used: {cmd}")


def moses_detokenizer(lang, input_file, output_file, conda_env_name=None):
    env = f"conda activate {conda_env_name}" if conda_env_name else "echo \"No conda env. Using '/bin/bash'\""
    cmd = f"sacremoses -l {lang} -j$(nproc) detokenize < {input_file} > {output_file}"
    subprocess.call(['/bin/bash', '-i', '-c', f"{env} && {cmd}"])
    print(f"\t- Command used: {cmd}")


def replace_in_file(search_string, replace_string, filename):
    cmd = f"sed -i 's/{search_string}/{replace_string}/' {filename}"
    subprocess.call(['/bin/bash', '-i', '-c', f"{cmd}"])
    print(f"\t- Command used: {cmd}")
