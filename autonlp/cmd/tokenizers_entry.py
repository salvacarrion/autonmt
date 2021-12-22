import json
import os
import random
import subprocess
from pathlib import Path


random.seed(123)

CONDA_ENVNAME = "mltests"


def cmd_sacrebleu(ref_file, hyp_file, output_file, metrics, conda_env_name=None):
    # The max ngram (max_ngram_order:) is default to 4 as the it was found to be the highest correlation with monolingual human judgements
    # Source: https://towardsdatascience.com/machine-translation-evaluation-with-sacrebleu-and-bertscore-d7fdb0c47eb3

    # Set args
    sb_m = ""
    sb_m += "bleu " if "bleu" in metrics else ""
    sb_m += "chrf " if "chrf" in metrics else ""
    sb_m += "ter " if "ter" in metrics else ""

    # Run command
    env = f"conda activate {conda_env_name}" if conda_env_name else "echo \"No conda env. Using '/bin/bash'\""
    cmd = f"sacrebleu {ref_file} -i {hyp_file} -m {sb_m} -w 5 > {output_file}"  # bleu chrf ter
    subprocess.call(['/bin/bash', '-i', '-c', f"{env} && {cmd}"])
    print(f"\t- Command used: {cmd}")


def cmd_bertscore(ref_file, hyp_file, output_file, trg_lang, conda_env_name=None):
    # Run command
    env = f"conda activate {conda_env_name}" if conda_env_name else "echo \"No conda env. Using '/bin/bash'\""
    cmd = f"bert-score -r {ref_file} -c {hyp_file} --lang {trg_lang} > {output_file}"
    subprocess.call(['/bin/bash', '-i', '-c', f"{env} && {cmd}"])
    print(f"\t- Command used: {cmd}")


def cmd_cometscore(src_file, ref_file, hyp_file, output_file, conda_env_name=None):
    # Run command
    env = f"conda activate {conda_env_name}" if conda_env_name else "echo \"No conda env. Using '/bin/bash'\""
    cmd = f"comet-score -s {src_file} -t {hyp_file} -r {ref_file} > {output_file}"
    subprocess.call(['/bin/bash', '-i', '-c', f"{env} && {cmd}"])
    print(f"\t- Command used: {cmd}")


def cmd_beer(ref_file, hyp_file, output_file, conda_env_name=None):
    # Run command
    env = f"conda activate {conda_env_name}" if conda_env_name else "echo \"No conda env. Using '/bin/bash'\""
    cmd = f"beer -s {hyp_file} -r {ref_file} > {output_file}"
    subprocess.call(['/bin/bash', '-i', '-c', f"{env} && {cmd}"])
    print(f"\t- Command used: {cmd}")

    
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
