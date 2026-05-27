"""Pytorch-Lightning backed translator.

Concrete :class:`~autonmt.backends.base.translator.BaseTranslator` that owns the
DataLoaders, callbacks, loggers and checkpoint management for a
:class:`~autonmt.core.seq2seq.LitSeq2Seq` model.
"""
import inspect
import os.path

# IMPORTANT: comet_ml must be imported before torch/pytorch_lightning so its auto-logging
# patches take effect. Do not reorder. It is an optional dependency — guarded so the
# module still imports cleanly when the user doesn't need Comet logging.
try:
    import comet_ml  # noqa: F401
    _COMET_AVAILABLE = True
except ImportError:
    _COMET_AVAILABLE = False

import torch
import pytorch_lightning as pl

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader

from autonmt.utils.logger import get_logger
from autonmt.utils.fileio import write_file_lines
from autonmt.core.dataset import TranslationDataset
from autonmt.core.samplers import BucketSampler, RandomSampler, SequentialSampler
from autonmt.core.decoding import BeamSearch, GreedySearch
from autonmt.backends.base.translator import BaseTranslator

log = get_logger(__name__)


def set_model_device(model, accelerator: str = "auto"):
    """Resolve the user's accelerator preference against what's available."""
    if torch.cuda.is_available():
        default_device = "cuda"
    elif torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"

    if accelerator == "auto":
        device = default_device
    elif accelerator in {"cuda", "gpu"}:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif accelerator == "mps":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = "cpu"

    if model.device.type != device:
        log.info(f"\t- Setting '{device}' as the model's device")
        model = model.to(device)
    else:
        log.info(f"\t- Model is already on '{device}' device")
    return model


class AutonmtTranslator(BaseTranslator):

    def __init__(self, model, **kwargs):
        super().__init__(engine="autonmt", **kwargs)
        self.model = model

        # Translation preprocessing tensors (set in _preprocess).
        self.train_tds = None
        self.val_tds = None
        self.test_tds = None

    # --- preprocess -----------------------------------------------------

    def _preprocess(self, train_path, val_path, test_path,
                    apply2train, apply2val, apply2test,
                    src_lang, trg_lang, src_vocab_path, trg_vocab_path,
                    output_path, force_overwrite, **kwargs):
        params = dict(src_lang=src_lang, trg_lang=trg_lang,
                      src_vocab=self.src_vocab, trg_vocab=self.trg_vocab)

        if apply2train:
            _, filter_fn = self.train_subset
            self.train_tds = TranslationDataset(file_prefix=train_path, filter_fn=filter_fn, **params)

        if apply2val:
            self.val_tds = [TranslationDataset(file_prefix=val_path, filter_fn=fn, **params)
                            for _, fn in self.val_subsets]

        if apply2test:
            self.test_tds = [TranslationDataset(file_prefix=test_path, filter_fn=fn, **params)
                             for _, fn in self.test_subsets]

    # --- train ----------------------------------------------------------

    def _train(self, train_ds, checkpoints_dir, logs_path, force_overwrite, **kwargs):
        monitor = kwargs.get("monitor")
        mode_str = "min" if "loss" in monitor.lower() else "max"
        pin_memory = torch.cuda.is_available() and kwargs.get('devices') != "cpu"
        use_bucketing = kwargs.get("use_bucketing")

        if not use_bucketing and self.model.packed_sequence:
            raise ValueError("Packed sequence is only compatible with bucketing")

        self._configure_model(kwargs)

        train_loader = self._build_loader(
            tds=self.train_tds, kwargs=kwargs, pin_memory=pin_memory,
            shuffle_default=True, label="training", index=(1, 1),
        )
        val_loaders = [
            self._build_loader(
                tds=val_tds_i, kwargs=kwargs, pin_memory=pin_memory,
                shuffle_default=False, label="validation",
                index=(i + 1, len(self.val_tds)),
            )
            for i, val_tds_i in enumerate(self.val_tds)
        ]

        callbacks = self._build_callbacks(monitor, mode_str, checkpoints_dir, kwargs)
        loggers, comet_logger = self._build_loggers(logs_path, kwargs)

        pl_whitelist = set(inspect.signature(pl.Trainer.__init__).parameters)
        pl_params = {k: v for k, v in kwargs.items() if k in pl_whitelist}
        trainer = pl.Trainer(logger=loggers, callbacks=callbacks, **pl_params)
        trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loaders)

        closed = []
        if kwargs.get("wandb_params"):
            wandb.finish()
            closed.append("wandb")
        if comet_logger is not None:
            comet_logger.experiment.end()
            closed.append("comet")
        if closed:
            log.info(f"\t- Closed external loggers: {', '.join(closed)}")

    def _configure_model(self, kwargs):
        self.model.strategy = kwargs.get("strategy")
        self.model.optimizer = kwargs.get("optimizer")
        self.model.learning_rate = kwargs.get("learning_rate")
        self.model.weight_decay = kwargs.get("weight_decay")
        self.model.configure_criterion(kwargs.get("criterion"))

        # Per-step metric context that the LightningModule introspects.
        self.model._src_vocab = self.train_tds.src_vocab
        self.model._trg_vocab = self.train_tds.trg_vocab
        self.model._filter_train = self.train_subset
        self.model._filter_eval = self.val_subsets
        self.model._print_samples = kwargs.get("print_samples")

    def _build_loader(self, tds, kwargs, pin_memory, shuffle_default, label, index):
        i, total = index
        log.info(f"\t- Preparing {label} dataloader... ({i}/{total})")
        batch_size = kwargs.get("batch_size")
        max_tokens = kwargs.get("max_tokens")
        num_workers = kwargs.get("num_workers")
        use_bucketing = kwargs.get("use_bucketing")
        seed = kwargs.get("seed") or 0

        common = dict(
            collate_fn=tds.get_collate_fn(max_tokens),
            num_workers=num_workers,
            persistent_workers=bool(num_workers),
            pin_memory=pin_memory,
        )

        if use_bucketing:
            mode = "max_tokens" if max_tokens else "batch_size"
            log.info(f"\t\t- Preparing bucketing iterator (mode={mode})...")
            batch_sampler = BucketSampler(
                tds,
                sort_key=lambda x, y: (
                    len(self.model._src_vocab.encode(x))
                    + len(self.model._trg_vocab.encode(y))
                ),
                batch_size=None if max_tokens else batch_size,
                max_tokens=max_tokens,
                shuffle=shuffle_default,
                sort_within_batch=self.model.packed_sequence,
                seed=seed,
            )
            return DataLoader(tds, batch_sampler=batch_sampler, **common)
        elif shuffle_default:
            return DataLoader(tds, sampler=RandomSampler(tds, seed=seed),
                              batch_size=batch_size, **common)
        else:
            return DataLoader(tds, sampler=SequentialSampler(tds),
                              batch_size=batch_size, **common)

    @staticmethod
    def _build_callbacks(monitor, mode_str, checkpoints_dir, kwargs):
        save_best = kwargs.get("save_best")
        save_last = kwargs.get("save_last")
        patience = kwargs.get("patience")
        # tqdm refreshes via \r so it works in PyCharm/notebook consoles where
        # Lightning's default RichProgressBar buffers until the run ends.
        callbacks = [TQDMProgressBar()]

        ckpt_filename = "{epoch:03d}-{" + monitor.replace('/', '-') + ":.3f}"
        ckpt_p = {}
        if save_best:
            ckpt_p.update({"monitor": monitor, "mode": mode_str,
                           "filename": ckpt_filename + "__best"})
        if save_last:
            ckpt_p.update({"save_last": save_last})
        if ckpt_p:
            checkpoint_callback = ModelCheckpoint(dirpath=checkpoints_dir, save_top_k=1, **ckpt_p)
            checkpoint_callback.FILE_EXTENSION = ".pt"
            if save_last:
                checkpoint_callback.CHECKPOINT_NAME_LAST = ckpt_filename + "__last"
            callbacks.append(checkpoint_callback)

        if patience:
            callbacks.append(EarlyStopping(monitor=monitor, patience=patience, mode=mode_str))
        return callbacks

    def _build_loggers(self, logs_path, kwargs):
        loggers = []
        comet_logger = None

        if logs_path:
            loggers.append(TensorBoardLogger(save_dir=logs_path))

        wandb_params = kwargs.get("wandb_params")
        if wandb_params:
            if not _WANDB_AVAILABLE:
                raise ImportError(
                    "wandb_params was provided but the 'wandb' package is not installed. "
                    "Install with: pip install 'autonmt[wandb]'  (or: pip install wandb)"
                )
            loggers.append(WandbLogger(save_dir=logs_path, name=self.run_name, **wandb_params))

        comet_params = kwargs.get("comet_params")
        if comet_params:
            if not _COMET_AVAILABLE:
                raise ImportError(
                    "comet_params was provided but the 'comet_ml' package is not installed. "
                    "Install with: pip install 'autonmt[comet]'  (or: pip install comet_ml)"
                )
            comet_logger = CometLogger(save_dir=logs_path, experiment_name=self.run_name, **comet_params)
            loggers.append(comet_logger)

        return loggers, comet_logger

    # --- translate ------------------------------------------------------

    def _translate(self, data_path, output_path, src_lang, trg_lang, beam_width,
                   max_len_a, max_len_b, batch_size, max_tokens,
                   checkpoint, num_workers, devices, accelerator,
                   force_overwrite, checkpoints_dir=None, filter_idx=0, **kwargs):
        if checkpoint:  # "best", "last", filename, or absolute path
            self.from_checkpoint = self.load_checkpoint(checkpoint)

        self.model = set_model_device(self.model, accelerator=accelerator)

        search_algorithm = BeamSearch() if beam_width > 1 else GreedySearch()
        predictions, _ = search_algorithm.decode(
            model=self.model, dataset=self.test_tds[filter_idx],
            sos_id=self.trg_vocab.sos_id,
            eos_id=self.trg_vocab.eos_id,
            pad_id=self.trg_vocab.pad_id,
            batch_size=batch_size, max_tokens=max_tokens,
            beam_width=beam_width, max_len_a=max_len_a, max_len_b=max_len_b,
            num_workers=num_workers,
        )
        self._write_hypothesis(predictions=predictions, output_path=output_path)

    def _write_hypothesis(self, predictions, output_path):
        """Decode token ids to text and dump to ``hyp.tok``.

        ``src`` / ``ref`` are filled in by the base translator from the original
        preprocessed files (avoids biasing scores with model-emitted ``<unk>``).
        """
        hyp_tok = [self.trg_vocab.decode(tokens) for tokens in predictions]
        write_file_lines(lines=hyp_tok, filename=os.path.join(output_path, "hyp.tok"),
                         insert_break_line=True)

    # --- checkpoints ----------------------------------------------------

    @staticmethod
    def _get_checkpoints(checkpoints_dir, mode):
        if not os.path.isdir(checkpoints_dir):
            raise ValueError(f"[WARNING] Checkpoint directory does not exist: {checkpoints_dir}")

        # Sorted by ctime so most-recent always wins when multiple match.
        paths = sorted(
            [os.path.join(checkpoints_dir, f) for f in os.listdir(checkpoints_dir)
             if f.endswith(f'__{mode}.pt')],
            key=os.path.getctime, reverse=True,
        )
        if not paths:
            raise ValueError(f"[WARNING] No ({mode}) checkpoints were found in {checkpoints_dir}")
        if len(paths) > 1:
            log.warning(f"Multiple checkpoints were found. Using more recent '{mode}': {paths[0]}")
        else:
            log.info(f"Checkpoint found: {paths[0]}")
        return paths[0]

    def load_checkpoint(self, checkpoint):
        if os.path.isfile(checkpoint):
            checkpoint_path = checkpoint
        elif checkpoint.endswith((".pt", ".pth")):
            checkpoint_path = os.path.join(self.get_model_checkpoints_path(), checkpoint)
        elif checkpoint in {"best", "last"}:
            checkpoint_path = self._get_checkpoints(self.get_model_checkpoints_path(), mode=checkpoint)
        else:
            raise ValueError("'checkpoint' must be a filename or 'best' or 'last'")

        # We persist full Lightning checkpoints (optimiser state, hyperparams…),
        # so weights_only=False is intentional.
        _model = torch.load(checkpoint_path, map_location=self.model.device, weights_only=False)
        self.model.load_state_dict(_model.get("state_dict", _model))
        return checkpoint_path

    def get_checkpoint_path(self, mode="best"):
        return self._get_checkpoints(self.get_model_checkpoints_path(), mode=mode)
