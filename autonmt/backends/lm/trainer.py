"""PyTorch-Lightning trainer for decoder-only language models.

Deliberately *not* a :class:`~autonmt.backends._base.translation_engine.BaseTranslator`
subclass: that class's ``predict()`` is translate-and-score (eval datasets,
beams, BLEU, src/ref/hyp), none of which maps onto an LM. Instead this is a
small, self-contained trainer that composes :class:`RunLayout` for paths and
:func:`manual_seed` for reproducibility, and exposes three verbs:

  * :meth:`fit`      — train on an :class:`LMCorpus` (packed blocks).
  * :meth:`evaluate` — perplexity on a split.
  * :meth:`generate` — sample a continuation for a prompt.

It reuses :class:`~autonmt.backends._base.config.FitConfig` for the training
hyperparameters (the LM-relevant subset: epochs, batch size, optimizer,
scheduler, warmup, devices, …); ``block_size`` is the one extra knob, passed to
:meth:`fit` because it ties the dataset to the model's context length.
"""
import datetime
import inspect
import os
import time

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from autonmt.backends._base.config import FitConfig, merge_config
from autonmt.backends._base.run_layout import RunLayout
from autonmt.core.data.lm_dataset import LMDataset
from autonmt.core.decoding.lm_generate import lm_generate
from autonmt.utils.fileio import make_dir, rename_file
from autonmt.utils.logger import get_logger
from autonmt.utils.seed import manual_seed

log = get_logger(__name__)


class LMTrainer:
    """Train / evaluate / sample from a decoder-only LM on an :class:`LMCorpus`."""

    ENGINE = "lm"

    def __init__(self, model, runs_dir="runs", run_name=None, corpus=None, safe_seconds=3):
        self.model = model
        self.runs_dir = runs_dir or "runs/"
        self.run_name = run_name or f"{datetime.datetime.now().strftime('%Y.%m.%d_%H.%M.%S')}"
        self.corpus = corpus
        self.safe_seconds = safe_seconds
        self._layout = RunLayout(runs_dir=self.runs_dir, run_name=self.run_name)
        self.from_checkpoint = None

    @classmethod
    def from_corpus(cls, corpus, *, run_prefix, model, **kwargs):
        """Bind a trainer to ``corpus``'s runs path (mirrors ``BaseTranslator.from_dataset``)."""
        return cls(
            model=model,
            runs_dir=corpus.get_runs_path(),
            run_name=corpus.get_run_name(run_prefix=run_prefix),
            corpus=corpus,
            **kwargs,
        )

    # --- Paths ----------------------------------------------------------

    def get_model_checkpoints_path(self, fname=""):
        return self._layout.checkpoints_path(fname)

    def get_model_logs_path(self, fname=""):
        return self._layout.logs_path(fname)

    # --- fit ------------------------------------------------------------

    def fit(self, corpus=None, config: FitConfig = None, block_size=256, **kwargs):
        corpus = corpus or self.corpus
        if corpus is None:
            raise ValueError("fit() needs a corpus (pass it here or to the constructor).")
        self.corpus = corpus

        cfg, extra = merge_config(config, FitConfig, kwargs)
        block_size = extra.pop("block_size", block_size)
        if block_size > self.model.max_seq_len:
            raise ValueError(
                f"block_size ({block_size}) exceeds the model's max_seq_len "
                f"({self.model.max_seq_len}); increase max_seq_len or lower block_size."
            )

        log.info("=" * 70)
        log.info(f"=> [LM Fit]: {corpus}  (run={self.run_name}, block_size={block_size})")
        log.info("=" * 70)

        checkpoints_dir = self.get_model_checkpoints_path()
        logs_path = self.get_model_logs_path()
        make_dir([checkpoints_dir, logs_path])

        # Never silently clobber a trained checkpoint (mirrors AutonmtTranslator).
        existing = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pt")] \
            if os.path.isdir(checkpoints_dir) else []
        if existing:
            if cfg["force_overwrite"]:
                log.info(f"\t- [Fit]: backing up {len(existing)} previous checkpoint(s) to '.pt.bak'...")
                for fname in existing:
                    rename_file(checkpoints_dir, fname, fname + ".bak")
            else:
                log.info("\t- [Fit]: Skipped. Checkpoints already exist "
                         "(pass force_overwrite=True to back them up and retrain).")
                return

        manual_seed(seed=cfg["seed"])
        self._configure_model(cfg)

        train_ds, val_ds = self._make_datasets(corpus, block_size)
        log.info(f"\t- Blocks: train={len(train_ds)}, val={len(val_ds)} (block_size={block_size})")

        common = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
                      persistent_workers=bool(cfg["num_workers"]),
                      pin_memory=torch.cuda.is_available())
        train_loader = DataLoader(train_ds, shuffle=True, **common)
        val_loader = DataLoader(val_ds, shuffle=False, **common)

        callbacks = self._build_callbacks(cfg, checkpoints_dir)
        loggers = [TensorBoardLogger(save_dir=logs_path)] if logs_path else []

        pl_whitelist = set(inspect.signature(pl.Trainer.__init__).parameters)
        pl_params = {k: v for k, v in cfg.items() if k in pl_whitelist}
        trainer = pl.Trainer(logger=loggers, callbacks=callbacks, **pl_params)

        start = time.time()
        trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        log.info(f"\t- Training time: {datetime.timedelta(seconds=time.time() - start)}")

    def _make_datasets(self, corpus, block_size):
        """Build the ``(train, val)`` torch datasets for :meth:`fit`.

        Overridden by :class:`~autonmt.backends.lm.mlm_trainer.MLMTrainer` to
        build masked-LM datasets instead.
        """
        return (
            LMDataset(corpus.tokens_file(corpus.train_name), block_size=block_size,
                      supervise_file=self._sup_file(corpus, corpus.train_name)),
            LMDataset(corpus.tokens_file(corpus.val_name), block_size=block_size,
                      supervise_file=self._sup_file(corpus, corpus.val_name)),
        )

    @staticmethod
    def _sup_file(corpus, split):
        """Supervise-mask file for instruct corpora; None for pure-LM (text)."""
        path = corpus.supervise_file(split)
        return path if os.path.exists(path) else None

    def _configure_model(self, cfg):
        self.model.strategy = cfg.get("strategy")
        self.model.optimizer = cfg.get("optimizer")
        self.model.learning_rate = cfg.get("learning_rate")
        self.model.weight_decay = cfg.get("weight_decay")
        self.model.scheduler = cfg.get("scheduler")
        self.model.warmup_steps = cfg.get("warmup_steps")
        self.model.configure_criterion(cfg.get("criterion"))

    @staticmethod
    def _build_callbacks(cfg, checkpoints_dir):
        monitor = cfg.get("monitor") or "val_loss"
        mode = "min" if "loss" in monitor.lower() else "max"
        callbacks = [TQDMProgressBar()]

        ckpt_filename = "{epoch:03d}-{" + monitor.replace('/', '-') + ":.3f}"
        ckpt_p = {}
        if cfg.get("save_best"):
            ckpt_p.update({"monitor": monitor, "mode": mode, "filename": ckpt_filename + "__best"})
        if cfg.get("save_last"):
            ckpt_p.update({"save_last": cfg.get("save_last")})
        if ckpt_p:
            checkpoint_callback = ModelCheckpoint(dirpath=checkpoints_dir, save_top_k=1, **ckpt_p)
            checkpoint_callback.FILE_EXTENSION = ".pt"
            if cfg.get("save_last"):
                checkpoint_callback.CHECKPOINT_NAME_LAST = ckpt_filename + "__last"
            callbacks.append(checkpoint_callback)

        if cfg.get("patience"):
            callbacks.append(EarlyStopping(monitor=monitor, patience=cfg["patience"], mode=mode))
        return callbacks

    # --- evaluate (perplexity) ------------------------------------------

    @torch.no_grad()
    def evaluate(self, corpus=None, split=None, block_size=256, batch_size=16,
                 accelerator="auto"):
        """Token-level perplexity on a split (defaults to the corpus val split)."""
        corpus = corpus or self.corpus
        split = split or corpus.val_name
        device = self._resolve_device(accelerator)
        self.model = self.model.to(device).eval()

        ds = LMDataset(corpus.tokens_file(split), block_size=block_size,
                       supervise_file=self._sup_file(corpus, split))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.model.padding_idx, reduction="sum")

        total_loss, total_tokens = 0.0, 0
        for x, y, supervise in loader:
            x, y, supervise = x.to(device), y.to(device), supervise.to(device)
            logits = self.model(x)
            y = y.masked_fill(supervise == 0, self.model.padding_idx)
            total_loss += criterion(logits.transpose(1, 2), y).item()
            total_tokens += int((y != self.model.padding_idx).sum().item())

        total_tokens = max(total_tokens, 1)
        mean_nll = total_loss / total_tokens
        ppl = float(torch.exp(torch.tensor(mean_nll)))
        log.info(f"=> [Evaluate]: split={split!r} loss={mean_nll:.4f} ppl={ppl:.2f} "
                 f"({total_tokens} tokens)")
        return {"loss": mean_nll, "ppl": ppl, "tokens": total_tokens}

    # --- generate -------------------------------------------------------

    def generate(self, prompt, sampler=None, max_new_tokens=64, corpus=None,
                 add_sos=True, stop_at_eos=True, accelerator="auto", return_ids=False):
        """Sample a continuation for ``prompt`` (a string) and decode it to text."""
        corpus = corpus or self.corpus
        if corpus is None:
            raise ValueError("generate() needs a corpus to (de)tokenize text.")
        device = self._resolve_device(accelerator)
        self.model = self.model.to(device).eval()

        prompt_ids = corpus.encode(prompt, add_sos=add_sos, add_eos=False)
        out_ids = lm_generate(
            self.model, prompt_ids, sampler=sampler, max_new_tokens=max_new_tokens,
            eos_id=corpus.eos_id if stop_at_eos else None, device=device,
        )
        if return_ids:
            return out_ids
        return corpus.decode(out_ids)

    # --- checkpoints ----------------------------------------------------

    def load_checkpoint(self, checkpoint="best"):
        ckpt_dir = self.get_model_checkpoints_path()
        if os.path.isfile(checkpoint):
            path = checkpoint
        elif checkpoint.endswith((".pt", ".pth", ".ckpt")):
            path = os.path.join(ckpt_dir, checkpoint)
        elif checkpoint in {"best", "last"}:
            path = self._find_checkpoint(ckpt_dir, mode=checkpoint)
        else:
            raise ValueError("'checkpoint' must be a filename or 'best' or 'last'")

        ckpt = torch.load(path, map_location=self.model.device, weights_only=False)
        self.model.load_state_dict(ckpt.get("state_dict", ckpt))
        self.from_checkpoint = path
        return path

    @staticmethod
    def _find_checkpoint(checkpoints_dir, mode):
        if not os.path.isdir(checkpoints_dir):
            raise ValueError(f"Checkpoint directory does not exist: {checkpoints_dir}")
        paths = sorted(
            [os.path.join(checkpoints_dir, f) for f in os.listdir(checkpoints_dir)
             if f.endswith(f'__{mode}.pt')],
            key=os.path.getctime, reverse=True,
        )
        if not paths:
            raise ValueError(f"No ({mode}) checkpoints found in {checkpoints_dir}")
        return paths[0]

    @staticmethod
    def _resolve_device(accelerator):
        if accelerator in {"cuda", "gpu"} or (accelerator == "auto" and torch.cuda.is_available()):
            return "cuda" if torch.cuda.is_available() else "cpu"
        if accelerator == "mps" or (accelerator == "auto" and torch.backends.mps.is_available()):
            return "mps" if torch.backends.mps.is_available() else "cpu"
        return "cpu"
