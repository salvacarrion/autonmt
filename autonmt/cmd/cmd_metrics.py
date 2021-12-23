import subprocess

from autonmt import utils
from sacrebleu.metrics import BLEU, CHRF, TER
import bert_score


def sacrebleu_scores(ref_file, hyp_file, output_file, metrics):
    # Read file
    ref_lines = utils.read_file_lines(ref_file)
    hyp_lines = utils.read_file_lines(hyp_file)

    scores = []
    if "bleu" in metrics:
        bleu = BLEU()
        d = bleu.corpus_score(hyp_lines, [ref_lines]).__dict__
        d["signature"] = str(bleu.get_signature())
        scores.append(d)

    if "chrf" in metrics:
        chrf = CHRF()
        d = chrf.corpus_score(hyp_lines, [ref_lines]).__dict__
        d["signature"] = str(chrf.get_signature())
        scores.append(d)

    if "ter" in metrics:
        ter = TER()
        d = ter.corpus_score(hyp_lines, [ref_lines]).__dict__
        d["signature"] = str(ter.get_signature())
        scores.append(d)

    # Save json
    utils.save_json(scores, output_file)

def cmd_bertscore(ref_file, hyp_file, output_file, trg_lang, **kwargs):
    # Read file
    ref_lines = utils.read_file_lines(ref_file)
    hyp_lines = utils.read_file_lines(hyp_file)

    # Score
    p, r, f1 = bert_score.score(hyp_lines, ref_lines, lang=trg_lang)

    scores = [
        {"name": "bertscore",
         "precision": float(p.mean()),
         "recall": float(r.mean()),
         "f1": float(f1.mean()),
         }
    ]

    # Save json
    utils.save_json(scores, output_file)


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


def _cmd_bertscore(ref_file, hyp_file, output_file, trg_lang, conda_env_name=None):
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