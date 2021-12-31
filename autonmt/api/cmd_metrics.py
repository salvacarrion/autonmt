import subprocess

from autonmt.api import NO_CONDA_MSG


def cmd_sacrebleu(ref_file, hyp_file, output_file, metrics, conda_env_name=None):
    print("\t- [INFO]: Using 'sacrebleu' from the command line.")

    # The max ngram (max_ngram_order:) is default to 4 as the it was found to be the highest correlation with monolingual human judgements
    # Source: https://towardsdatascience.com/machine-translation-evaluation-with-sacrebleu-and-bertscore-d7fdb0c47eb3

    # Set args
    sb_m = ""
    sb_m += "bleu " if "sacrebleu" in metrics or "bleu" in metrics else ""
    sb_m += "chrf " if "chrf" in metrics else ""
    sb_m += "ter " if "ter" in metrics else ""

    # Run command
    env = f"conda activate {conda_env_name}" if conda_env_name else NO_CONDA_MSG
    cmd = f"sacrebleu {ref_file} -i {hyp_file} -m {sb_m} -w 5 > {output_file}"  # bleu chrf ter
    subprocess.call(['/bin/bash', '-i', '-c', f"{env} && {cmd}"])
    return cmd


def cmd_bertscore(ref_file, hyp_file, output_file, trg_lang, conda_env_name=None):
    print("\t- [INFO]: Using 'bertscore' from the command line.")

    # Run command
    env = f"conda activate {conda_env_name}" if conda_env_name else NO_CONDA_MSG
    cmd = f"bert-score -r {ref_file} -c {hyp_file} --lang {trg_lang} > {output_file}"
    subprocess.call(['/bin/bash', '-i', '-c', f"{env} && {cmd}"])
    return cmd


def cmd_cometscore(src_file, ref_file, hyp_file, output_file, conda_env_name=None):
    print("\t- [INFO]: Using 'cometscore' from the command line.")

    # Run command
    env = f"conda activate {conda_env_name}" if conda_env_name else NO_CONDA_MSG
    cmd = f"comet-score -s {src_file} -t {hyp_file} -r {ref_file} > {output_file}"
    subprocess.call(['/bin/bash', '-i', '-c', f"{env} && {cmd}"])
    return cmd


def cmd_beer(ref_file, hyp_file, output_file, conda_env_name=None):
    print("\t- [INFO]: Using 'beer' from the command line.")

    # Run command
    env = f"conda activate {conda_env_name}" if conda_env_name else NO_CONDA_MSG
    cmd = f"beer -s {hyp_file} -r {ref_file} > {output_file}"
    subprocess.call(['/bin/bash', '-i', '-c', f"{env} && {cmd}"])
    return cmd