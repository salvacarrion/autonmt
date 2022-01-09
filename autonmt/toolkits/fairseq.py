import os
import shutil
import subprocess

from autonmt.toolkits.base import BaseTranslator
from autonmt.bundle import utils
from autonmt.api import NO_VENV_MSG


def _parse_args(**kwargs):
    cmd = []

    # Set reserved args
    reserved_args = {"fairseq-preprocess", "fairseq-train", "fairseq-generate",
                     "--save-dir", "--tensorboard-logdir", "--wandb-project", "--skip-invalid-size-inputs-valid-test"}
    # reserved_args.update(autonmt2fairseq.keys())

    # Check: autonmt args (proposal)
    autonmt2fairseq = {
        'learning_rate': "--lr",  # Default: 0.25
        'criterion': "--criterion",  # Default: "cross_entropy". (cross_entropy, ctc, hubert, ...)
        'optimizer': "--optimizer",  # No default. (adadelta, adafactor, adagrad, adam, adamax, nag, sgd,...)
        'gradient_clip_val': "--clip-norm",  # Default: 0.0
        'accumulate_grad_batches': "--update-freq",  # Default: 1
        'max_epochs': "--max-epoch",  # No default
        'max_tokens': "--max-tokens",  # No default
        'batch_size': "--batch-size",  # No default
        'patience': "--patience",  # Default: -1
        'seed': "--seed",  # Default: 1
        'monitor': "--best-checkpoint-metric",  # Default: "loss"
        'num_workers': "--num-workers",  # Default: 1
    }
    autonmt_fix_values = {
        "patience": lambda x: -1 if x <= 0 else x,
        "num_workers": lambda x: 1 if x <= 0 else x
    }

    proposed_args = []  # From AutoNMTBase
    for autonmt_arg_name, autonmt_arg_value in kwargs.items():
        fairseq_arg_name = autonmt2fairseq.get(autonmt_arg_name)
        if autonmt_arg_value is not None and fairseq_arg_name:  # Has value and exists translation
            # Is the value valid?
            if autonmt_arg_name in autonmt_fix_values:
                new_value = autonmt_fix_values[autonmt_arg_name](autonmt_arg_value)
            else:
                new_value = autonmt_arg_value
            proposed_args.append(f"{fairseq_arg_name} {str(new_value)}")

    # Check: fairseq args
    fairseq_args = kwargs.get("fairseq_args", [])
    if fairseq_args is not None and isinstance(fairseq_args, (list, set, dict)):
        if any([x.split(' ')[0] in reserved_args for x in fairseq_args]):  # Check for reserved params
            raise ValueError(f"A reserved fairseq arg was used. List of reserved args: {str(reserved_args)}")
    else:
        raise ValueError("No valid fairseq args were provided.\n"
                         "You can add them with 'model.fit(fairseq_args=FARSEQ_ARGS)', where 'FAIRSEQ_ARGS' is a "
                         "list with the fairseq parameters (['--arch transformer', '--lr 0.001',...])")

    # Add proposed args if they were not explicitly set fairseq_args
    fairseq_args_keys = set([arg.split(' ')[0] for arg in fairseq_args])
    proposed_fairseq_keys = [arg.split(' ')[0] for arg in proposed_args]
    for autonmt_arg, autonmt_key in zip(proposed_args, proposed_fairseq_keys):
        if autonmt_key not in fairseq_args_keys:
            cmd += [autonmt_arg]
    cmd += fairseq_args
    return cmd


def _postprocess_output(output_path):
    """
    Important: src and ref will be overwritten with the original preprocessed files to avoid problems with unknowns
    """
    # Extract sentences from generate-test.txt
    gen_test_path = os.path.join(output_path, "generate-test.txt")
    src_tok_path = os.path.join(output_path, "src.tok")
    ref_tok_path = os.path.join(output_path, "ref.tok")
    hyp_tok_path = os.path.join(output_path, "hyp.tok")
    subprocess.call(['/bin/bash', '-c', f"grep ^S {gen_test_path} | LC_ALL=C sort -V | cut -f2- > {src_tok_path}"])
    subprocess.call(['/bin/bash', '-c', f"grep ^T {gen_test_path} | LC_ALL=C sort -V | cut -f2- > {ref_tok_path}"])
    subprocess.call(['/bin/bash', '-c', f"grep ^H {gen_test_path} | LC_ALL=C sort -V | cut -f3- > {hyp_tok_path}"])

    # Replace "<<unk>>" with "<unk>" in ref.tok
    utils.replace_in_file(search_string="<<unk>>", replace_string="<unk>", filename=ref_tok_path)


def vocab_spm2fairseq(filename):
    # Read file
    lines = utils.read_file_lines(filename, strip=False, remove_break_lines=False)

    # Drop headers
    lines = lines[4:]  # <unk>, <s>, </s>, <pad>

    # Clean lines
    lines = [line.split('\t')[0] + f" {1}" for line in lines]

    # Write file
    utils.write_file_lines(lines, filename)


class FairseqTranslator(BaseTranslator):

    def __init__(self, fairseq_venv_path=None, wandb_params=None, **kwargs):
        super().__init__(engine="fairseq", **kwargs)

        # Vars
        self.fairseq_venv_path = fairseq_venv_path
        self.wandb_params = wandb_params

        # Check conda environment
        if fairseq_venv_path is None:
            print("=> [INFO] 'FairseqTranslator' needs the fairseq commands to be accessible from the '/bin/bash'\n"
                  "   - You can also install it in a virtual environment, and set 'venv_path=MYVIRTUALENV'")

    def _preprocess(self, src_lang, trg_lang, output_path, train_path, val_path, test_path, src_vocab_path,
                    trg_vocab_path, subword_model, **kwargs):
        # Reformat vocab files for fairseq
        new_src_vocab_path = ""
        new_trg_vocab_path = ""
        if src_vocab_path:
            new_src_vocab_path = os.path.join(output_path, f"dict.{src_lang}.txt")
            shutil.copyfile(f"{src_vocab_path}.vocab", new_src_vocab_path)
            vocab_spm2fairseq(filename=new_src_vocab_path)
        if trg_vocab_path:
            new_trg_vocab_path = os.path.join(output_path, f"dict.{trg_lang}.txt")
            shutil.copyfile(f"{trg_vocab_path}.vocab", new_trg_vocab_path)
            vocab_spm2fairseq(filename=new_trg_vocab_path)

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
        cmd += [f"--validpref '{val_path}'"] if val_path else []
        cmd += [f"--srcdict '{new_src_vocab_path}'"] if new_src_vocab_path else []
        cmd += [f"--tgtdict '{new_trg_vocab_path}'"] if new_trg_vocab_path else []

        # Parse fairseq args (will throw "error: unrecognized arguments")
        # cmd += _parse_args(**kwargs)

        # Run command
        cmd = " ".join([] + cmd)
        env = f"{self.fairseq_venv_path}" if self.fairseq_venv_path else NO_VENV_MSG
        print(f"\t- [INFO]: Command used: {cmd}")
        subprocess.call(['/bin/bash', '-c', f"{env} && {cmd}"])

    def _train(self, data_bin_path, checkpoints_path, logs_path, max_tokens, batch_size, run_name, ds_alias, **kwargs):
        # Write command
        cmd = [f"fairseq-train '{data_bin_path}'"]
        cmd += [f"--save-dir '{checkpoints_path}'"] if checkpoints_path else []
        cmd += [f"--tensorboard-logdir '{logs_path}'"] if logs_path else []

        # Parse fairseq args
        cmd += _parse_args(max_tokens=max_tokens, batch_size=batch_size, **kwargs)

        # Add wandb logger (if requested)
        wandb_env = []
        if self.wandb_params:
            wandb_run_name = f"{ds_alias}_{run_name}"
            wandb_env = f"WANDB_NAME='{wandb_run_name}'"
            wandb_project = self.wandb_params['project']
            cmd += [f"--wandb-project '{wandb_project}'"]

        # Parse gpu flag
        num_gpus = kwargs.get('devices')
        num_gpus = None if num_gpus == "auto" else num_gpus
        num_gpus = f"CUDA_VISIBLE_DEVICES={','.join([str(i) for i in range(num_gpus)])}" if isinstance(num_gpus, int) else ""

        # Run command
        cmd_env = [wandb_env, num_gpus]
        cmd_env = " ".join([x for x in cmd_env if x])  # Needs spaces. It doesn't work with ";" or "&&"
        cmd_env = cmd_env + " " if cmd_env else cmd_env
        cmd = " ".join([cmd_env] + cmd)
        env = f"{self.fairseq_venv_path}" if self.fairseq_venv_path else NO_VENV_MSG
        print(f"\t- [INFO]: Command used: {cmd}")
        subprocess.call(['/bin/bash', '-c', f"{env} && {cmd}"])

    def _translate(self, src_lang, trg_lang, beam_width, max_gen_length, batch_size, max_tokens,
                   data_bin_path, output_path, checkpoint_path, model_src_vocab_path, model_trg_vocab_path, **kwargs):
        # Write command
        cmd = [f"fairseq-generate {data_bin_path}"]

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
            f"--scoring sacrebleu",
            # f"--skip-invalid-size-inputs-valid-test",  #DISABLE!!! (else, the ref and hyp might not match)
        ]

        # Parse fairseq args
        cmd += _parse_args(max_tokens=max_tokens, batch_size=batch_size, **kwargs)

        # Parse gpu flag
        num_gpus = kwargs.get('devices')
        num_gpus = None if num_gpus == "auto" else num_gpus
        num_gpus = f"CUDA_VISIBLE_DEVICES={','.join([str(i) for i in range(num_gpus)])}" if isinstance(num_gpus, int) else ""

        # Run command
        cmd = " ".join([num_gpus] + cmd)
        env = f"{self.fairseq_venv_path}" if self.fairseq_venv_path else NO_VENV_MSG
        print(f"\t- [INFO]: Command used: {cmd}")
        subprocess.call(['/bin/bash', '-c', f"{env} && {cmd}"])

        # Prepare output files (from fairseq to tokenized form)
        _postprocess_output(output_path=output_path)
