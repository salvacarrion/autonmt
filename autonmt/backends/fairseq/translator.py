"""Translator backend that shells out to the Fairseq CLI.

.. deprecated::
    Fairseq was archived by Meta on 2026-03-20 and is no longer maintained.
    New code should prefer :class:`~autonmt.backends.autonmt.translator.AutonmtTranslator`.
"""
import os
import shutil
import warnings

from autonmt.utils import fileio as utils
from autonmt.utils.logger import get_logger
from autonmt.backends.base.translator import BaseTranslator

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Deprecation banner. Re-emitted at instantiation time too so users see *where*
# in their code the dependency lives, not just at import.
# ---------------------------------------------------------------------------
_DEPRECATION_MSG = (
    "FairseqTranslator is deprecated: fairseq was archived by its maintainers on "
    "2026-03-20 and is no longer receiving updates. Existing installations still "
    "work, but new code should use AutonmtTranslator. See "
    "https://github.com/facebookresearch/fairseq for context."
)
warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)

try:
    import fairseq_cli  # noqa: F401
    from fairseq import options  # noqa: F401
    from fairseq_cli import preprocess, train, generate  # noqa: F401
    from fairseq.distributed import utils as distributed_utils  # noqa: F401
    from fairseq.dataclass.utils import convert_namespace_to_omegaconf  # noqa: F401
    _FAIRSEQ_AVAILABLE = True
    _FAIRSEQ_IMPORT_ERROR = None
except Exception as e:  # ImportError or anything fairseq raises internally
    _FAIRSEQ_AVAILABLE = False
    _FAIRSEQ_IMPORT_ERROR = e
    log.info(
        "Fairseq is not installed or failed to import; FairseqTranslator will "
        "raise ImportError on instantiation. Install with: pip install fairseq"
    )


# ---------------------------------------------------------------------------
# AutoNMT ↔ fairseq argument bridge (module-level so it's not rebuilt per call).
# ---------------------------------------------------------------------------
_RESERVED_FAIRSEQ_ARGS = frozenset({
    "fairseq-preprocess", "fairseq-train", "fairseq-generate",
    "--save-dir", "--tensorboard-logdir", "--wandb-project",
    "--skip-invalid-size-inputs-valid-test", "--bpe", "--remove-bpe",
})

# How autonmt kwargs map onto fairseq flags. Order doesn't matter; collision
# with user-supplied ``fairseq_args`` is detected separately.
_AUTONMT_TO_FAIRSEQ = {
    'learning_rate': "--lr",
    'criterion': "--criterion",
    'optimizer': "--optimizer",
    'gradient_clip_val': "--clip-norm",
    'accumulate_grad_batches': "--update-freq",
    'max_epochs': "--max-epoch",
    'max_tokens': "--max-tokens",
    'batch_size': "--batch-size",
    'patience': "--patience",
    'seed': "--seed",
    'monitor': "--best-checkpoint-metric",
    'num_workers': "--num-workers",
}

# Per-kwarg coercion (fairseq has stricter expectations than autonmt for some).
_AUTONMT_FIX_VALUES = {
    "patience": lambda x: -1 if x <= 0 else x,
    "num_workers": lambda x: 1 if x <= 0 else x,
}


def _translate_autonmt_args(kwargs) -> list:
    """Render autonmt kwargs as fairseq CLI flags (one element per ``--flag value`` pair)."""
    out = []
    for name, value in kwargs.items():
        flag = _AUTONMT_TO_FAIRSEQ.get(name)
        if value is None or flag is None:
            continue
        fixer = _AUTONMT_FIX_VALUES.get(name)
        out.append(f"{flag} {fixer(value) if fixer else value}")
    return out


def _check_no_reserved(fairseq_args) -> None:
    if not isinstance(fairseq_args, (list, set, dict)):
        raise ValueError(
            "No valid fairseq args were provided.\n"
            "You can add them with 'model.fit(fairseq_args=FARSEQ_ARGS)', where 'FAIRSEQ_ARGS' is a "
            "list with the fairseq parameters (['--arch transformer', '--lr 0.001',...])"
        )
    for arg in fairseq_args:
        if arg.split(' ')[0] in _RESERVED_FAIRSEQ_ARGS:
            raise ValueError(
                f"A reserved fairseq arg was used. List of reserved args: {list(_RESERVED_FAIRSEQ_ARGS)}"
            )


def _parse_args(**kwargs) -> list:
    """Build the fairseq CLI args list: user-supplied wins, autonmt fills the gaps."""
    fairseq_args = kwargs.get("fairseq_args", [])
    _check_no_reserved(fairseq_args)

    user_keys = {arg.split(' ')[0] for arg in fairseq_args}
    proposed = _translate_autonmt_args(kwargs)
    cmd = [arg for arg in proposed if arg.split(' ')[0] not in user_keys]
    cmd += list(fairseq_args)
    return cmd


def _extract_hypotheses(output_path: str) -> None:
    """Pull model hypotheses (``^H`` lines) out of fairseq's ``generate-test.txt``.

    Previously this shelled out to ``grep ... | LC_ALL=C sort -V | cut -f3-``; now
    it's pure Python so it works on Windows and doesn't depend on ``/bin/bash``.

    We deliberately *don't* extract src/ref from fairseq's output — its ``<unk>``
    replacements would bias the score; the base translator copies them from the
    original raw files instead.
    """
    gen_test_path = os.path.join(output_path, "generate-test.txt")
    hyp_tok_path = os.path.join(output_path, "hyp.tok")

    hyps = []
    with open(gen_test_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.startswith("H-"):
                continue
            # Format: H-<sent_id>\t<log_prob>\t<hypothesis>
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3:
                continue
            sent_id = int(parts[0][2:])  # strip "H-"
            hyps.append((sent_id, parts[2]))

    hyps.sort(key=lambda h: h[0])
    with open(hyp_tok_path, 'w', encoding='utf-8') as f:
        for _, text in hyps:
            f.write(text + '\n')


def vocab_spm2fairseq(filename):
    """Rewrite an SPM ``.vocab`` file in fairseq's dict format (token + frequency)."""
    lines = utils.read_file_lines(filename, autoclean=False)
    lines = lines[4:]  # Drop unk/sos/eos/pad — fairseq inserts its own.
    lines = [line.split('\t')[0] + " 1" for line in lines]
    utils.write_file_lines(lines=lines, filename=filename, insert_break_line=True)


class FairseqTranslator(BaseTranslator):
    """Translator backend that shells out to the Fairseq CLI.

    .. deprecated::
        Fairseq was archived on 2026-03-20 and is no longer maintained.
        New code should prefer :class:`~autonmt.backends.autonmt.translator.AutonmtTranslator`.
    """

    def __init__(self, wandb_params=None, **kwargs):
        if not _FAIRSEQ_AVAILABLE:
            raise ImportError(
                "Fairseq is not installed. FairseqTranslator requires the (now-archived) "
                "'fairseq' package.\n"
                "  Install:  pip install fairseq\n"
                "  Note:     fairseq was archived on 2026-03-20 and is unmaintained — "
                "prefer AutonmtTranslator for new projects."
            ) from _FAIRSEQ_IMPORT_ERROR

        warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)

        super().__init__(engine="fairseq", **kwargs)
        self.wandb_params = wandb_params
        self.data_bin_name = "data-bin"

    # --- preprocess -----------------------------------------------------

    def _preprocess(self, ds, output_path, src_lang, trg_lang, train_path, val_path, test_path,
                    src_vocab_path, trg_vocab_path,
                    apply2train, apply2val, apply2test, force_overwrite, **kwargs):
        # Build data-bin output path.
        if not output_path:  # Train
            output_path = ds.get_bin_data(self.engine, self.data_bin_name)
        else:  # Test
            output_path = os.path.join(output_path, ds.data_path, self.data_bin_name)
        utils.make_dir([output_path])

        if not utils.is_dir_empty(output_path):
            if force_overwrite:
                log.info(f"\t- [Preprocess]: Deleting directory: {output_path}")
                utils.empty_dir(output_path, safe_seconds=self.safe_seconds)
            else:
                log.info("\t- [Preprocess]: Skipped. The output directory is not empty")
                return

        new_src_vocab_path = self._reformat_vocab(src_vocab_path, src_lang, output_path)
        new_trg_vocab_path = self._reformat_vocab(trg_vocab_path, trg_lang, output_path)

        # Fairseq always wants a trainpref, even at eval time.
        if apply2test:
            train_path = test_path

        input_args = [
            "--source-lang", src_lang,
            "--target-lang", trg_lang,
            "--trainpref", train_path,
            "--testpref", test_path,
            "--destdir", output_path,
        ]
        if val_path:
            input_args += ["--validpref", val_path]
        if new_src_vocab_path:
            input_args += ["--srcdict", new_src_vocab_path]
        if new_trg_vocab_path:
            input_args += ["--tgtdict", new_trg_vocab_path]

        input_args = self._flatten_cli(input_args)

        log.info("COMMAND:")
        log.info("fairseq-preprocess " + ' '.join(input_args))

        parser = options.get_preprocessing_parser(default_task="translation")
        args = parser.parse_args(args=input_args)
        preprocess.main(args)

    @staticmethod
    def _reformat_vocab(vocab_path: str, lang: str, output_path: str) -> str:
        if not vocab_path:
            return ""
        new_path = os.path.join(output_path, f"dict.{lang}.txt")
        shutil.copyfile(f"{vocab_path}.vocab", new_path)
        vocab_spm2fairseq(filename=new_path)
        return new_path

    # --- train ----------------------------------------------------------

    def _train(self, train_ds, checkpoints_dir, logs_path, max_tokens, batch_size,
               force_overwrite, **kwargs):
        wandb_params = kwargs.get("wandb_params")
        data_bin_path = train_ds.get_bin_data(self.engine, self.data_bin_name)

        # Gate on live ``.pt`` files only — ``.pt.bak`` from previous runs is ignored
        # so cumulative backups don't block re-training. Matches AutonmtTranslator.
        existing_ckpts = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pt")] \
            if os.path.isdir(checkpoints_dir) else []
        if existing_ckpts:
            if force_overwrite:
                log.info(f"\t- [Train]: Renaming {len(existing_ckpts)} previous checkpoint(s) to '.pt.bak' to avoid overwriting...")
                for fname in existing_ckpts:
                    utils.rename_file(checkpoints_dir, fname, fname + ".bak")
            else:
                log.info("\t- [Train]: Skipped. The checkpoint directory already contains checkpoints "
                         "(pass force_overwrite=True to back them up and retrain).")
                return

        input_args = [data_bin_path]
        if checkpoints_dir:
            input_args += ["--save-dir", checkpoints_dir]
        if logs_path:
            input_args += ["--tensorboard-logdir", logs_path]
        if wandb_params:
            log.warning("\t\t- 'wandb_params' has known parallelisation bugs in fairseq")
            input_args += ["--wandb-project", wandb_params["project"]]
            os.environ["WANDB_NAME"] = self.run_name

        input_args += _parse_args(max_tokens=max_tokens, batch_size=batch_size, **kwargs)
        input_args = self._flatten_cli(input_args)

        num_gpus = kwargs.get('devices')
        num_gpus = None if num_gpus == "auto" else num_gpus
        if num_gpus and isinstance(num_gpus, int):
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in range(num_gpus))

        log.info("COMMAND:")
        log.info("fairseq-train " + ' '.join(input_args))

        parser = options.get_training_parser()
        args = options.parse_args_and_arch(parser, input_args=input_args)
        cfg = convert_namespace_to_omegaconf(args)
        distributed_utils.call_main(cfg, train.main)

    # --- translate ------------------------------------------------------

    def _translate(self, model_ds, data_path, output_path, src_lang, trg_lang, beam_width,
                   max_len_a, max_len_b, batch_size, max_tokens,
                   checkpoints_dir, model_src_vocab_path, model_trg_vocab_path,
                   **kwargs):
        if kwargs.get('devices'):
            log.warning("\t\t- 'devices' will be ignored when using Fairseq")
        if kwargs.get('decoder') is not None:
            log.warning("\t\t- 'decoder' will be ignored when using Fairseq "
                        "(custom decoders are only supported by AutonmtTranslator)")

        data_bin_path = os.path.join(data_path, model_ds.data_path, self.data_bin_name)
        input_args = [data_bin_path]
        input_args += [
            "--source-lang", src_lang,
            "--target-lang", trg_lang,
            "--path", os.path.join(checkpoints_dir, "checkpoint_best.pt"),
            "--results-path", output_path,
            "--beam", beam_width,
            "--max-len-a", max_len_a,
            "--max-len-b", max_len_b,
            "--nbest", 1,
            "--scoring", "sacrebleu",
        ]
        input_args += _parse_args(max_tokens=max_tokens, batch_size=batch_size, **kwargs)
        input_args = self._flatten_cli(input_args)

        log.info("COMMAND:")
        log.info("fairseq-generate " + ' '.join(input_args))

        parser = options.get_generation_parser(default_task="translation")
        args = options.parse_args_and_arch(parser, input_args=input_args)
        generate.main(args)

        _extract_hypotheses(output_path=output_path)

    @staticmethod
    def _flatten_cli(args):
        """Turn ['--flag value', ...] into ['--flag', 'value', ...] so argparse is happy."""
        return sum([str(c).split(' ', 1) for c in args], [])
