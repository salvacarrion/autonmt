import os
import shutil
import subprocess
import time

from autonmt.bundle import utils
from autonmt.toolkits.base import BaseTranslator

try:
    import fairseq_cli
    from fairseq import options
    from fairseq_cli import preprocess, train, generate
    from fairseq.distributed import utils as distributed_utils
    from fairseq.dataclass.utils import convert_namespace_to_omegaconf
except ImportError as e:
    print("WARNING: Fairseq toolkit could not be loaded. FairseqTranslator will not be available.")
except Exception as e:
    print("WARNING: Fairseq toolkit could not be loaded. FairseqTranslator will not be available.")
    print(e)



def _parse_args(**kwargs):
    cmd = []

    # Set reserved args
    reserved_args = {"fairseq-preprocess", "fairseq-train", "fairseq-generate",
                     "--save-dir", "--tensorboard-logdir", "--wandb-project", "--skip-invalid-size-inputs-valid-test",
                     "--bpe", "--remove-bpe"}
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
    hyp_tok_path = os.path.join(output_path, "hyp.tok")
    subprocess.call(['/bin/bash', '-c', f"grep ^H {gen_test_path} | LC_ALL=C sort -V | cut -f3- > {hyp_tok_path}"])

    # Do not decode src/ref from fairseq as the <unk> would biases the score
    # src_tok_path = os.path.join(output_path, "src.tok")
    # ref_tok_path = os.path.join(output_path, "ref.tok")
    # subprocess.call(['/bin/bash', '-c', f"grep ^S {gen_test_path} | LC_ALL=C sort -V | cut -f2- > {src_tok_path}"])
    # subprocess.call(['/bin/bash', '-c', f"grep ^T {gen_test_path} | LC_ALL=C sort -V | cut -f2- > {ref_tok_path}"])

    # Replace "<<unk>>" with "<unk>" in ref.tok
    # utils.replace_in_file(search_string="<<unk>>", replace_string="<unk>", filename=ref_tok_path)


def vocab_spm2fairseq(filename):
    # Read file
    lines = utils.read_file_lines(filename, autoclean=False)

    # Drop headers
    lines = lines[4:]  # <unk>, <s>, </s>, <pad>

    # Clean lines
    lines = [line.split('\t')[0] + f" {1}" for line in lines]

    # Write file
    utils.write_file_lines(lines=lines, filename=filename, insert_break_line=True)


class FairseqTranslator(BaseTranslator):

    def __init__(self,  wandb_params=None, **kwargs):
        super().__init__(engine="fairseq", **kwargs)

        # Vars
        self.wandb_params = wandb_params

        # Custom
        self.data_bin_name = "data-bin"

    def _preprocess(self, ds, output_path, src_lang, trg_lang, train_path, val_path, test_path, src_vocab_path,
                    trg_vocab_path, apply2train, apply2val, apply2test, force_overwrite, **kwargs):

        # Create data-bin directory if it doesn't exist
        if not output_path:  # Train
            output_path = ds.get_bin_data(self.engine, self.data_bin_name)
        else:  # Test
            output_path = os.path.join(output_path, ds.data_path, self.data_bin_name)
        utils.make_dir([output_path])

        # Check if the output directory is empty and take action
        if not utils.is_dir_empty(output_path):
            if force_overwrite:  # Empty dir
                print(f"\t- [Preprocess]: Deleting directory: {output_path}")
                utils.empty_dir(output_path, safe_seconds=self.safe_seconds)
            else:
                print("\t- [Preprocess]: Skipped. The output directory is not empty")
                return

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
        if apply2test:
            train_path = test_path

        # Write command
        input_args = []  #f"fairseq-preprocess"

        # Base params
        input_args += [
            "--source-lang", src_lang,
            "--target-lang", trg_lang,
            "--trainpref", train_path,
            "--testpref", test_path,
            "--destdir", output_path,
        ]

        # Optional params
        input_args += ["--validpref", val_path] if val_path else []
        input_args += ["--srcdict", new_src_vocab_path] if new_src_vocab_path else []
        input_args += ["--tgtdict", new_trg_vocab_path] if new_trg_vocab_path else []

        # Parse args and execute command
        # From: https://github.com/pytorch/fairseq/blob/main/fairseq_cli/preprocess.py
        input_args = sum([str(c).split(' ', 1) for c in input_args], [])  # Split key/val (str) and flat list

        # Command
        print("COMMAND:")
        print("fairseq-preprocess " + ' '.join(input_args))

        # Run command
        parser = options.get_preprocessing_parser(default_task="translation")
        args = parser.parse_args(args=input_args)
        preprocess.main(args)

    def _train(self, train_ds, checkpoints_dir, logs_path, max_tokens, batch_size, force_overwrite, **kwargs):
        wandb_params = kwargs.get("wandb_params")

        # Get data-bin path
        data_bin_path = train_ds.get_bin_data(self.engine, self.data_bin_name)

        # Check if the directory is empty and take action
        if not utils.is_dir_empty(checkpoints_dir):
            if force_overwrite:  # Empty dir
                print(f"\t- [Train]: Renaming previous checkpoints to avoid overwriting...")
                utils.rename_file(checkpoints_dir, "checkpoint_best.pt", "checkpoint_best.pt.bak")
                utils.rename_file(checkpoints_dir, "checkpoint_last.pt", "checkpoint_last.pt.bak")
            else:
                print("\t- [Train]: Skipped. The checkpoint directory is not empty")
                return

        # if self.wandb_params:
        #     print("\t\t- [WARNING]: 'wandb_params' will be ignored when using Fairseq due to some known bugs")

        # Write command
        input_args = [data_bin_path]
        input_args += ["--save-dir", checkpoints_dir] if checkpoints_dir else []
        input_args += ["--tensorboard-logdir", logs_path] if logs_path else []
        if wandb_params:
            # raise ValueError("WandB monitoring is disabled for FairSeq due to a bug related to parallelization.")
            print("\t\t- [WARNING]: 'wandb_params' will produce to some known bugs")

            # Set vars
            input_args += ["--wandb-project", wandb_params["project"]]
            os.environ["WANDB_NAME"] = self.run_name

        # Parse fairseq args
        input_args += _parse_args(max_tokens=max_tokens, batch_size=batch_size, **kwargs)
        input_args = sum([str(c).split(' ', 1) for c in input_args], [])  # Split key/val (str) and flat list

        # Parse gpu flag
        num_gpus = kwargs.get('devices')
        num_gpus = None if num_gpus == "auto" else num_gpus
        if num_gpus and isinstance(num_gpus, int):
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in range(num_gpus)])

        # Command
        print("COMMAND:")
        print("fairseq-train " + ' '.join(input_args))

        # Run command
        # From: https://github.com/pytorch/fairseq/blob/main/fairseq_cli/train.py
        parser = options.get_training_parser()
        args = options.parse_args_and_arch(parser, input_args=input_args)
        cfg = convert_namespace_to_omegaconf(args)
        distributed_utils.call_main(cfg, train.main)

    def _translate(self, model_ds, data_path, output_path, src_lang, trg_lang, beam_width, max_len_a, max_len_b, batch_size, max_tokens,
                   checkpoints_dir, model_src_vocab_path, model_trg_vocab_path,
                   force_overwrite, **kwargs):
        # Set warnings
        if kwargs.get('devices'):
            print("\t\t- [WARNING]: 'devices' will be ignored when using Fairseq")

        # Write command
        data_bin_path = os.path.join(data_path, model_ds.data_path, self.data_bin_name)
        input_args = [data_bin_path]

        # Add stuff
        input_args += [
            "--source-lang", src_lang,
            "--target-lang", trg_lang,
            "--path", os.path.join(checkpoints_dir, "checkpoint_best.pt"),
            "--results-path", output_path,
            "--beam", beam_width,
            "--max-len-a", max_len_a,  # max_len = ax+b
            "--max-len-b", max_len_b,
            "--nbest", 1,
            "--scoring", "sacrebleu",
            #"--skip-invalid-size-inputs-valid-test",  # DISABLE. (Else, 'ref' and 'hyp' might not match)
            #"--remove-bpe",  # DISABLE. I'd rather decode it using sentencepiece (not available for training)
        ]

        # Parse fairseq args
        input_args += _parse_args(max_tokens=max_tokens, batch_size=batch_size, **kwargs)

        # Parse args and execute command
        # From: https://github.com/pytorch/fairseq/blob/main/fairseq_cli/generate.py
        input_args = sum([str(c).split(' ', 1) for c in input_args], [])  # Split key/val (str) and flat list

        # Command
        print("COMMAND:")
        print("fairseq-generate " + ' '.join(input_args))

        # Run command
        # From: https://github.com/facebookresearch/fairseq/blob/main/fairseq_cli/generate.py
        parser = options.get_generation_parser(default_task="translation")
        args = options.parse_args_and_arch(parser, input_args=input_args)
        generate.main(args)

        # Prepare output files (from fairseq to tokenized form)
        _postprocess_output(output_path=output_path)


