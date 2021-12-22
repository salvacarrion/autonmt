import os
import random
import shutil
import subprocess
from abc import ABC
from pathlib import Path

from autonlp.tasks.translation.base import BaseTranslator
from autonlp.cmd import tokenizers_entry


def _parse_args(**kwargs):
    cmd = []
    reserved_args = {"fairseq-preprocess", "fairseq-train", "fairseq-generate",
                     "--save-dir", "--tensorboard-logdir"}

    # Add args
    fairseq_args = kwargs.get("fairseq_args")
    if not fairseq_args or not isinstance(fairseq_args, list):
        raise ValueError("No fairseq args were provided.\n"
                         "You can add them with 'model.fit(fairseq_args=FARSEQ_ARGS)', where 'FAIRSEQ_ARGS' is a "
                         "list with the fairseq parameters (['--arch transformer', '--lr 0.001',...])")
    else:
        if any([x in fairseq_args for x in reserved_args]):
            raise ValueError(f"A reserved fairseq arg was used. List of reserved args: {str(reserved_args)}")
        else:  # Add args
            cmd = fairseq_args
    return cmd


class FairseqTranslator(BaseTranslator):

    def __init__(self, *args, conda_fairseq_env_name=None, **kwargs):
        super().__init__(*args, engine="fairseq", **kwargs)

        # Vars
        self.conda_fairseq_env_name = conda_fairseq_env_name

        # Check conda environment
        if conda_fairseq_env_name is None:
            print("=> [INFO] 'FairseqTranslator' needs the fairseq commands to be accessible from the '/bin/bash'\n"
                  "   - You can also install it in a conda environment, and set 'conda_env_name=MYCONDAENV'")
            # raise ValueError("'FairseqTranslator' needs the name of the conda environment where it is installed")

    def _preprocess(self, src_lang, trg_lang, output_path, train_path, val_path, test_path, src_vocab_path,
                   trg_vocab_path, *args, **kwargs):
        # Reformat vocab files for fairseq
        new_src_vocab_path = ""
        new_trg_vocab_path = ""
        if src_vocab_path:
            new_src_vocab_path = os.path.join(output_path, f"dict.{src_lang}.txt")
            shutil.copyfile(f"{src_vocab_path.strip()}.vocabf", new_src_vocab_path)
            tokenizers_entry.replace_in_file(search_string='\t', replace_string=' ', filename=new_src_vocab_path)
        if trg_vocab_path:
            new_trg_vocab_path = os.path.join(output_path, f"dict.{trg_lang}.txt")
            shutil.copyfile(f"{trg_vocab_path.strip()}.vocabf", new_trg_vocab_path)
            tokenizers_entry.replace_in_file(search_string='\t', replace_string=' ', filename=new_trg_vocab_path)

        # Trick for generation. A train path is required
        if kwargs.get("external_data"):
            train_path = test_path

        # Write command
        cmd = [f"fairseq-preprocess"]

        # Base params
        cmd += [
            f"--source-lang {src_lang}",
            f"--target-lang {trg_lang}",
            f"--trainpref '{train_path}'",
            f"--testpref '{test_path}'",
            f"--destdir '{output_path}'",
        ]

        # Optional params
        cmd += [f" --validpref '{val_path}'"] if val_path else []
        cmd += [f" --srcdict '{new_src_vocab_path}'"] if new_src_vocab_path else []
        cmd += [f" --tgtdict '{new_trg_vocab_path}'"] if new_trg_vocab_path else []

        # Parse fairseq args
        # cmd += _parse_args(**kwargs)

        # Run command
        cmd = " ".join([] + cmd)
        env = f"conda activate {self.conda_fairseq_env_name}" if self.conda_fairseq_env_name else "echo \"No conda env. Using '/bin/bash'\""
        subprocess.call(['/bin/bash', '-i', '-c', f"{env} && {cmd}"])

    def _train(self, data_bin_path, checkpoints_path, logs_path, *args, **kwargs):
        # Write command
        cmd = [f"fairseq-train '{data_bin_path}'"]
        cmd += [f"--save-dir '{checkpoints_path}'"] if checkpoints_path else []
        cmd += [f"--tensorboard-logdir '{logs_path}'"] if logs_path else []

        # Parse fairseq args
        cmd += _parse_args(**kwargs)

        # Parse gpu flag
        num_gpus = f"CUDA_VISIBLE_DEVICES={','.join([str(i) for i in range(self.num_gpus)])}" if self.num_gpus else ""

        # Run command
        cmd = " ".join([num_gpus] + cmd)
        env = f"conda activate {self.conda_fairseq_env_name}" if self.conda_fairseq_env_name else "echo \"No conda env. Using '/bin/bash'\""
        subprocess.call(['/bin/bash', '-i', '-c', f"{env} && {cmd}"])

    def _translate(self, src_lang, trg_lang, data_path, output_path, checkpoint_path, src_spm_model_path,
                   trg_spm_model_path, beam_width, max_gen_length,
                   *args, **kwargs):
        # Write command
        cmd = [f"fairseq-generate {data_path}"]

        # Add stuff
        cmd += [
            f"--source-lang {src_lang}",
            f"--target-lang {trg_lang}",
            f"--path '{checkpoint_path}'",
            f"--results-path '{output_path}'",
            f"--beam {beam_width}",
            f"--max-len-a {0}",  # max_len = ax+b
            f"--max-len-b {max_gen_length}",
            f"--nbest 1",
            "--scoring sacrebleu",
            "--skip-invalid-size-inputs-valid-test",
        ]

        # Parse fairseq args
        # cmd += _parse_args(**kwargs)

        # Parse gpu flag
        num_gpus = f"CUDA_VISIBLE_DEVICES={','.join([str(i) for i in range(self.num_gpus)])}" if self.num_gpus else ""

        # Run command
        cmd = " ".join([num_gpus] + cmd)
        env = f"conda activate {self.conda_fairseq_env_name}" if self.conda_fairseq_env_name else "echo \"No conda env. Using '/bin/bash'\""
        subprocess.call(['/bin/bash', '-i', '-c', f"{env} && {cmd}"])

        # Extract sentences from generate-test.txt
        gen_test_path = os.path.join(output_path, "generate-test.txt")
        src_tok_path = os.path.join(output_path, "src.tok")
        ref_tok_path = os.path.join(output_path, "ref.tok")
        hyp_tok_path = os.path.join(output_path, "hyp.tok")
        subprocess.call(['/bin/bash', '-i', '-c', f"grep ^S {gen_test_path} | cut -f2- > {src_tok_path}"])
        subprocess.call(['/bin/bash', '-i', '-c', f"grep ^T {gen_test_path} | cut -f2- > {ref_tok_path}"])
        subprocess.call(['/bin/bash', '-i', '-c', f"grep ^H {gen_test_path} | cut -f3- > {hyp_tok_path}"])

        # Replace "<<unk>>" with "<unk>" in ref.tok
        tokenizers_entry.replace_in_file(search_string="<<unk>>", replace_string="<unk>", filename=ref_tok_path)

        # Detokenize
        src_txt_path = os.path.join(output_path, "src.txt")
        ref_txt_path = os.path.join(output_path, "ref.txt")
        hyp_txt_path = os.path.join(output_path, "hyp.txt")
        tokenizers_entry.spm_decode(src_spm_model_path, input_file=src_tok_path, output_file=src_txt_path, conda_env_name=self.conda_env_name)
        tokenizers_entry.spm_decode(trg_spm_model_path, input_file=ref_tok_path, output_file=ref_txt_path, conda_env_name=self.conda_env_name)
        tokenizers_entry.spm_decode(trg_spm_model_path, input_file=hyp_tok_path, output_file=hyp_txt_path, conda_env_name=self.conda_env_name)
