import json
import os
import random
import subprocess
from pathlib import Path


random.seed(123)

CONDA_ENVNAME = "fairseq"


def preprocess(src_lang, trg_lang, output_path, train_path, val_path, test_path, src_vocab_path, trg_vocab_path, *args, **kwargs):
    # Define command
    cmd = f"fairseq-preprocess " \
          f"--source-lang {src_lang} " \
          f"--target-lang {trg_lang} " \
          f"--trainpref {train_path} " \
          f"--testpref {test_path} " \
          f"--destdir {output_path} " \
          f"--workers $(nproc)"
    cmd += f" --validpref {val_path}" if val_path else ""
    cmd += f" --srcdict {src_vocab_path}" if src_vocab_path else ""
    cmd += f" --tgtdict {trg_vocab_path}" if trg_vocab_path else ""

    # Run command
    subprocess.call(['/bin/bash', '-i', '-c', f"conda activate {CONDA_ENVNAME} && {cmd}"])


def train(ds_path, src_lang, trg_lang, subword_model, vocab_size, force_overwrite, interactive, run_name, model_path, num_gpus):
    pass


def translate(data_path, checkpoint_path, output_path, src_lang, trg_lang, beam_width, max_gen_length, force_overwrite, interactive):
    pass


def score():
    pass