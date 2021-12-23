import os
import shutil
import subprocess

from autonmt.tasks.translation.base import BaseTranslator
from autonmt import utils
from autonmt.cmd import NO_CONDA_MSG


def _parse_args(**kwargs):
    cmd = []

    # Set reserved args
    reserved_args = {"fairseq-preprocess", "fairseq-train", "fairseq-generate", "--save-dir", "--tensorboard-logdir"}
    # reserved_args.update(autonmt2fairseq.keys())

    # Check fairseq args
    fairseq_args = kwargs.get("fairseq_args")
    if not fairseq_args or not isinstance(fairseq_args, list):
        raise ValueError("No fairseq args were provided.\n"
                         "You can add them with 'model.fit(fairseq_args=FARSEQ_ARGS)', where 'FAIRSEQ_ARGS' is a "
                         "list with the fairseq parameters (['--arch transformer', '--lr 0.001',...])")
    else:
        if any([x in fairseq_args for x in reserved_args]):
            raise ValueError(f"A reserved fairseq arg was used. List of reserved args: {str(reserved_args)}")

    # Convert AutoNLP args to Fairseq
    autonmt2fairseq = {'batch_size': "--batch-size", 'max_tokens': "--max-tokens", 'max_epochs': "--max-epoch",
                       'learning_rate': "--lr", 'clip_norm': "--clip-norm", 'patience': "--patience",
                       'criterion': "--criterion", 'optimizer': "--optimizer"}
    proposed_args = []
    for autonmt_arg_name, autonmt_arg_value in kwargs.items():
        faisreq_arg_name = autonmt2fairseq.get(autonmt_arg_name)
        if autonmt_arg_value is not None and faisreq_arg_name:
            proposed_args.append(f"{faisreq_arg_name} {str(autonmt_arg_value)}")

    # Add params: Fairseq params have preference over the autonmt params, but autonmt params have preference over
    # the default fairseq params
    proposed_fairseq_keys = [arg.split(' ')[0] for arg in proposed_args]
    fairseq_keys = set([arg.split(' ')[0] for arg in fairseq_args])
    for autonmt_arg, autonmt_key in zip(proposed_args, proposed_fairseq_keys):
        if autonmt_key not in fairseq_keys:
            cmd += [autonmt_arg]
    cmd += fairseq_args
    return cmd


def _postprocess_output(output_path):
    # Extract sentences from generate-test.txt
    gen_test_path = os.path.join(output_path, "generate-test.txt")
    src_tok_path = os.path.join(output_path, "src.tok")
    ref_tok_path = os.path.join(output_path, "ref.tok")
    hyp_tok_path = os.path.join(output_path, "hyp.tok")
    subprocess.call(['/bin/bash', '-i', '-c', f"grep ^S {gen_test_path} | cut -f2- > {src_tok_path}"])
    subprocess.call(['/bin/bash', '-i', '-c', f"grep ^T {gen_test_path} | cut -f2- > {ref_tok_path}"])
    subprocess.call(['/bin/bash', '-i', '-c', f"grep ^H {gen_test_path} | cut -f3- > {hyp_tok_path}"])

    # Replace "<<unk>>" with "<unk>" in ref.tok
    utils.replace_in_file(search_string="<<unk>>", replace_string="<unk>", filename=ref_tok_path)


class FairseqTranslator(BaseTranslator):

    def __init__(self, conda_fairseq_env_name=None, **kwargs):
        super().__init__(engine="fairseq", **kwargs)

        # Vars
        self.conda_fairseq_env_name = conda_fairseq_env_name

        # Check conda environment
        if conda_fairseq_env_name is None:
            print("=> [INFO] 'FairseqTranslator' needs the fairseq commands to be accessible from the '/bin/bash'\n"
                  "   - You can also install it in a conda environment, and set 'conda_env_name=MYCONDAENV'")
            # raise ValueError("'FairseqTranslator' needs the name of the conda environment where it is installed")

    def _preprocess(self, src_lang, trg_lang, output_path, train_path, val_path, test_path, src_vocab_path,
                    trg_vocab_path, **kwargs):
        # Reformat vocab files for fairseq
        new_src_vocab_path = ""
        new_trg_vocab_path = ""
        if src_vocab_path:
            new_src_vocab_path = os.path.join(output_path, f"dict.{src_lang}.txt")
            shutil.copyfile(f"{src_vocab_path.strip()}.vocabf", new_src_vocab_path)
            utils.replace_in_file(search_string='\t', replace_string=' ', filename=new_src_vocab_path)
        if trg_vocab_path:
            new_trg_vocab_path = os.path.join(output_path, f"dict.{trg_lang}.txt")
            shutil.copyfile(f"{trg_vocab_path.strip()}.vocabf", new_trg_vocab_path)
            utils.replace_in_file(search_string='\t', replace_string=' ', filename=new_trg_vocab_path)

        # Trick for generation.
        # Fairseq always requires a train path, but during evaluation there is no need.
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
        env = f"conda activate {self.conda_fairseq_env_name}" if self.conda_fairseq_env_name else NO_CONDA_MSG
        print(f"\t- [INFO]: Command used: {cmd}")
        subprocess.call(['/bin/bash', '-i', '-c', f"{env} && {cmd}"])

    def _train(self, data_bin_path, checkpoints_path, logs_path, **kwargs):
        # Write command
        cmd = [f"fairseq-train '{data_bin_path}'"]
        cmd += [f"--save-dir '{checkpoints_path}'"] if checkpoints_path else []
        cmd += [f"--tensorboard-logdir '{logs_path}'"] if logs_path else []

        # Parse fairseq args
        cmd += _parse_args(**kwargs)

        # Parse gpu flag
        num_gpus = kwargs.get('num_gpus')
        num_gpus = f"CUDA_VISIBLE_DEVICES={','.join([str(i) for i in range(num_gpus)])}" if num_gpus else ""

        # Run command
        cmd = " ".join([num_gpus] + cmd)
        env = f"conda activate {self.conda_fairseq_env_name}" if self.conda_fairseq_env_name else NO_CONDA_MSG
        print(f"\t- [INFO]: Command used: {cmd}")
        subprocess.call(['/bin/bash', '-i', '-c', f"{env} && {cmd}"])

    def _translate(self, src_lang, trg_lang, data_path, output_path, checkpoint_path, src_spm_model_path,
                   trg_spm_model_path, beam_width, max_gen_length, **kwargs):
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
        num_gpus = kwargs.get('num_gpus')
        num_gpus = f"CUDA_VISIBLE_DEVICES={','.join([str(i) for i in range(num_gpus)])}" if num_gpus else ""

        # Run command
        cmd = " ".join([num_gpus] + cmd)
        env = f"conda activate {self.conda_fairseq_env_name}" if self.conda_fairseq_env_name else NO_CONDA_MSG
        print(f"\t- [INFO]: Command used: {cmd}")
        subprocess.call(['/bin/bash', '-i', '-c', f"{env} && {cmd}"])

        # Prepare output files (from fairseq to tokenized form)
        _postprocess_output(output_path=output_path)

