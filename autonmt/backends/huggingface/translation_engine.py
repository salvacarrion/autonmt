"""HuggingFace backend — load any pretrained seq2seq model and evaluate it.

Wraps :class:`transformers.AutoModelForSeq2SeqLM` and
:class:`transformers.AutoTokenizer` so the user can point at any model id
(or local checkpoint) and reuse the existing ``predict()`` pipeline.
Fine-tuning via ``fit()`` is supported through :class:`Seq2SeqTrainer`.

The backend leaves :attr:`_spm` as ``None`` (no SPM round-trip), so
:meth:`BaseTranslator.translate` runs in *direct mode* — looping over
(subset, beam) and calling :meth:`_translate`, which writes the standard
``src.txt`` / ``ref.txt`` / ``hyp.txt`` artifacts straight from the HF
tokenizer/model. :meth:`score_translations` and :meth:`parse_metrics` then
consume those artifacts unchanged.
"""
import os
import time
from typing import Optional

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
    _TRANSFORMERS_IMPORT_ERROR = None
except ImportError as e:
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None
    _TRANSFORMERS_AVAILABLE = False
    _TRANSFORMERS_IMPORT_ERROR = e

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False

from autonmt.utils.fileio import read_file_lines, write_file_lines
from autonmt.utils.logger import get_logger
from autonmt.backends._base.translation_engine import BaseTranslator
from autonmt.reporting.report import RunMetadata

log = get_logger(__name__)


class HuggingFaceTranslator(BaseTranslator):
    """Inference + fine-tuning HuggingFace backend.

    Parameters
    ----------
    model_id : str
        HuggingFace Hub id (e.g. ``"Helsinki-NLP/opus-mt-de-en"``) or a path to
        a local checkpoint directory loadable with
        :meth:`AutoModelForSeq2SeqLM.from_pretrained`.
    tokenizer_id : str, optional
        Defaults to ``model_id``. Override when the tokenizer lives elsewhere.
    src_lang, trg_lang : str, optional
        Source / target language codes. Auto-filled by :meth:`from_dataset` from
        the dataset. Used by :meth:`filter_eval_datasets` (via
        ``_get_lang_pair``) and metric backends that need a target language.
    device : str
        ``"auto"`` (default), ``"cuda"``, ``"mps"``, or ``"cpu"``.
    generation_kwargs : dict, optional
        Extra kwargs forwarded to :meth:`model.generate` on every call. The
        ``num_beams`` / ``max_new_tokens`` arguments are managed by AutoNMT
        from :class:`PredictConfig` and override any value here.
    """

    ENGINE = "huggingface"

    def __init__(self, model_id: str, tokenizer_id: Optional[str] = None,
                 src_lang: Optional[str] = None, trg_lang: Optional[str] = None,
                 device: str = "auto",
                 generation_kwargs: Optional[dict] = None,
                 **kwargs):
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "HuggingFaceTranslator requires the 'transformers' package. "
                "Install it with:\n"
                "  pip install -e '.[hf-models]'   (or: pip install transformers)"
            ) from _TRANSFORMERS_IMPORT_ERROR
        if not _TORCH_AVAILABLE:
            raise ImportError("HuggingFaceTranslator requires PyTorch.")

        super().__init__(engine=self.ENGINE, **kwargs)
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id or model_id
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.device = device
        self.generation_kwargs = dict(generation_kwargs or {})

        # Lazy: don't load on construction so users can iterate (e.g. inspect
        # paths) without paying the download/load cost up-front.
        self._model = None
        self._tokenizer = None
        self._resolved_device: Optional[str] = None

        # No SPM round-trip — HF owns its own tokenizer. BaseTranslator.translate
        # falls into direct mode and calls _translate() with eval_ds + filter.

    # --- from_dataset: auto-fill src_lang/trg_lang ---------------------------

    @classmethod
    def from_dataset(cls, train_ds, *, run_prefix: str, **kwargs) -> "HuggingFaceTranslator":
        kwargs.setdefault("src_lang", train_ds.src_lang)
        kwargs.setdefault("trg_lang", train_ds.trg_lang)
        return super().from_dataset(train_ds, run_prefix=run_prefix, **kwargs)

    # --- Backend hooks (replace the old vocab-based overrides) -------------

    def _get_lang_pair(self):
        if self.src_lang is None or self.trg_lang is None:
            raise ValueError(
                "HuggingFaceTranslator needs src_lang / trg_lang. Pass them "
                "to the constructor or use HuggingFaceTranslator.from_dataset(...)."
            )
        return self.src_lang, self.trg_lang

    def _get_run_metadata(self) -> RunMetadata:
        """HF analogue of the AutoNMT run metadata, sourced from the HF
        model + tokenizer when loaded, falling back to ``model_id`` when not.

        Robust to ``_model`` / ``_tokenizer`` being ``None`` (e.g. predict-time
        report before any translate call has triggered :meth:`_ensure_loaded`).
        """
        model, tokenizer = self._model, self._tokenizer

        if model is not None:
            params = list(model.parameters())
            total_params = sum(p.numel() for p in params)
            trainable_params = sum(p.numel() for p in params if p.requires_grad)
            no_trainable_params = total_params - trainable_params
            dtype = str(next(iter(params)).dtype) if params else "unknown"
        else:
            total_params = trainable_params = no_trainable_params = 0
            dtype = "unknown"

        vocab_size = getattr(tokenizer, "vocab_size", None) if tokenizer is not None else None
        if vocab_size is None and tokenizer is not None:
            vocab_size = len(tokenizer)

        src_lang = self.src_lang
        trg_lang = self.trg_lang
        lang_pair = f"{src_lang}-{trg_lang}" if src_lang and trg_lang else None

        return RunMetadata(
            model__architecture=f"huggingface:{self.model_id}",
            model__total_params=total_params,
            model__trainable_params=trainable_params,
            model__no_trainable_params=no_trainable_params,
            model__dtype=dtype,
            vocab__subword_model="hf_tokenizer",
            vocab__size=vocab_size,
            # HF tokenizers are inherently shared by source & target.
            vocab__merged=True,
            vocab__lang_pair=lang_pair,
        )

    # --- Lazy model / tokenizer loading --------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        log.info(f"=> [HF]: Loading tokenizer from {self.tokenizer_id!r}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)

        log.info(f"=> [HF]: Loading model from {self.model_id!r}")
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)

        self._resolved_device = self._resolve_device(self.device)
        self._model = self._model.to(self._resolved_device)
        self._model.eval()
        log.info(f"\t- Loaded on device: {self._resolved_device}")

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # --- _train ---------------------------------------------------------

    def _train(self, train_ds, checkpoints_dir, logs_path, force_overwrite, **kwargs):
        """Fine-tune via :class:`transformers.Seq2SeqTrainer`.

        Maps :class:`FitConfig` fields onto :class:`Seq2SeqTrainingArguments`
        (see :meth:`_build_training_args`). Extra HF-specific knobs go through
        ``hf_training_args=dict(...)`` and win on collision with the mapping.

        After training, the best model and tokenizer are saved to
        ``checkpoints_dir`` and :attr:`model_id` / :attr:`tokenizer_id` are
        repointed there, so the next :meth:`predict` loads the fine-tuned
        weights instead of the original pretrained model.
        """
        from transformers import (
            DataCollatorForSeq2Seq, EarlyStoppingCallback, Seq2SeqTrainer,
        )

        # Fast path: skip training entirely if a previous checkpoint exists and
        # the caller hasn't asked to overwrite. Runs before the accelerate check
        # so users without accelerate can still load + predict from a finetuned
        # checkpoint produced elsewhere.
        if self._has_finetuned_checkpoint(checkpoints_dir):
            if force_overwrite:
                log.info(f"\t- [Train]: force_overwrite=True; overwriting existing "
                         f"checkpoint at {checkpoints_dir!r}")
            else:
                log.info(f"\t- [Train]: Skipped. A fine-tuned checkpoint already "
                         f"exists at {checkpoints_dir!r} (pass force_overwrite=True "
                         f"to retrain).")
                self.model_id = checkpoints_dir
                self.tokenizer_id = checkpoints_dir
                return

        # transformers.Trainer requires accelerate>=1.1 for any PyTorch flow.
        # Check up-front so the failure points at the right pip install hint,
        # not at HF's internal device-setup path.
        import importlib.util
        if importlib.util.find_spec("accelerate") is None:
            raise ImportError(
                "Fine-tuning with HuggingFaceTranslator requires the 'accelerate' "
                "package (>=1.1). Install with:\n"
                "  pip install -e '.[hf-models]'   (or: pip install 'accelerate>=1.1')"
            )

        # Load original pretrained weights + tokenizer (idempotent if already loaded).
        self._ensure_loaded()

        log.info(f"\t- Building fine-tune datasets from "
                 f"{train_ds.get_splits_auto_path()!r}")
        train_dataset = self._build_finetune_dataset(train_ds, train_ds.train_name)
        val_dataset = self._build_finetune_dataset(train_ds, train_ds.val_name)

        training_args = self._build_training_args(
            fit_kwargs=kwargs, output_dir=checkpoints_dir, logs_dir=logs_path,
            force_overwrite=force_overwrite,
        )

        callbacks = []
        patience = kwargs.get("patience")
        if patience:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=int(patience),
            ))

        collator = DataCollatorForSeq2Seq(tokenizer=self._tokenizer, model=self._model)
        # Seq2SeqTrainer renamed `tokenizer` to `processing_class` in transformers
        # 4.46. Pick whichever the installed version accepts.
        import inspect
        trainer_sig = inspect.signature(Seq2SeqTrainer.__init__).parameters
        tokenizer_kw = ("processing_class" if "processing_class" in trainer_sig
                        else "tokenizer")
        hf_trainer = Seq2SeqTrainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
            callbacks=callbacks,
            **{tokenizer_kw: self._tokenizer},
        )

        hf_trainer.train()

        # Save best model + tokenizer canonically; subsequent predict() reloads.
        hf_trainer.save_model(str(checkpoints_dir))
        self._tokenizer.save_pretrained(str(checkpoints_dir))
        log.info(f"\t- Saved fine-tuned model to {checkpoints_dir!r}")

        self.model_id = checkpoints_dir
        self.tokenizer_id = checkpoints_dir

    def _log_train_summary(self, train_ds, kwargs):
        """HF analogue of the parent summary; sourced from model_id + tokenizer."""
        def _kv(k, default="-"):
            v = kwargs.get(k)
            return default if v is None else v

        tok_size = getattr(self._tokenizer, "vocab_size", None) if self._tokenizer else None
        log.info("\t- Config:")
        log.info(f"\t\t- model: {self.model_id!r} (tokenizer_vocab={tok_size})")
        log.info(f"\t\t- training: epochs={_kv('max_epochs')}, "
                 f"batch_size={_kv('batch_size')}, lr={_kv('learning_rate')}, "
                 f"weight_decay={_kv('weight_decay')}")
        log.info(f"\t\t- monitor: {_kv('monitor')} "
                 f"(patience={_kv('patience')}, save_best={_kv('save_best')})")
        log.info(f"\t\t- device: {self._resolve_device(self.device)}, "
                 f"num_workers={_kv('num_workers')}, seed={_kv('seed')}")

    # --- Fine-tune helpers --------------------------------------------------

    @staticmethod
    def _has_finetuned_checkpoint(checkpoints_dir) -> bool:
        """A previously saved HF model lives as ``config.json`` + weight file."""
        if not os.path.isdir(checkpoints_dir):
            return False
        return os.path.exists(os.path.join(checkpoints_dir, "config.json"))

    def _build_finetune_dataset(self, ds, split_name):
        src_path = ds.get_splits_auto_path(f"{split_name}.{ds.src_lang}")
        trg_path = ds.get_splits_auto_path(f"{split_name}.{ds.trg_lang}")
        src_lines = read_file_lines(filename=src_path, autoclean=True)
        trg_lines = read_file_lines(filename=trg_path, autoclean=True)
        assert len(src_lines) == len(trg_lines), (
            f"src/trg length mismatch for split {split_name!r}: "
            f"{len(src_lines)} vs {len(trg_lines)}"
        )
        return _Seq2SeqTextDataset(
            src_lines=src_lines, trg_lines=trg_lines,
            tokenizer=self._tokenizer,
            max_source_length=self._max_source_length(),
            max_target_length=self._max_target_length(),
        )

    def _max_source_length(self) -> int:
        # Fall back to the tokenizer's reported limit if available.
        return int(getattr(self._tokenizer, "model_max_length", 1024) or 1024)

    def _max_target_length(self) -> int:
        return int(getattr(self._tokenizer, "model_max_length", 1024) or 1024)

    def _build_training_args_dict(self, fit_kwargs, output_dir, logs_dir, force_overwrite):
        """Pure kwargs dict for :class:`Seq2SeqTrainingArguments`.

        Split out from :meth:`_build_training_args` so unit tests can verify
        the FitConfig → HF mapping without instantiating the args object
        (which triggers HF's device-setup path and requires ``accelerate``).
        """
        import inspect
        from transformers import Seq2SeqTrainingArguments

        sig_params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
        # eval_strategy / evaluation_strategy renaming across transformers versions.
        eval_strategy_kw = ("eval_strategy" if "eval_strategy" in sig_params
                            else "evaluation_strategy")

        save_best = fit_kwargs.get("save_best", True)
        # load_best_model_at_end requires save_strategy == eval_strategy.
        save_strategy = "epoch" if save_best else "no"

        mapped = {
            "output_dir": str(output_dir),
            "overwrite_output_dir": bool(force_overwrite),
            "num_train_epochs": fit_kwargs.get("max_epochs", 1),
            "per_device_train_batch_size": fit_kwargs.get("batch_size", 8),
            "per_device_eval_batch_size": fit_kwargs.get("batch_size", 8),
            "learning_rate": fit_kwargs.get("learning_rate", 5e-5),
            "weight_decay": fit_kwargs.get("weight_decay") or 0.0,
            "max_grad_norm": fit_kwargs.get("gradient_clip_val") or 1.0,
            "gradient_accumulation_steps": fit_kwargs.get("accumulate_grad_batches", 1),
            eval_strategy_kw: "epoch",
            "save_strategy": save_strategy,
            "load_best_model_at_end": bool(save_best),
            "metric_for_best_model": fit_kwargs.get("monitor", "eval_loss"),
            "greater_is_better": False,
            "seed": fit_kwargs.get("seed", 42),
            "dataloader_num_workers": fit_kwargs.get("num_workers", 0),
            "logging_dir": str(logs_dir),
            "report_to": ["tensorboard"],
            "save_total_limit": 2,
            "predict_with_generate": False,
        }

        # User-supplied hf_training_args win on collision (mirrors fairseq_args).
        user_overrides = fit_kwargs.get("hf_training_args") or {}
        for k in set(mapped) & set(user_overrides):
            log.warning(
                f"\t- hf_training_args override: {k}={mapped[k]!r} → "
                f"{user_overrides[k]!r}"
            )
        mapped.update(user_overrides)

        # Drop any kwargs the installed transformers version doesn't accept.
        for k in [k for k in mapped if k not in sig_params]:
            log.warning(f"\t- Dropping unsupported Seq2SeqTrainingArguments kwarg: {k!r}")
            mapped.pop(k)

        return mapped

    def _build_training_args(self, fit_kwargs, output_dir, logs_dir, force_overwrite):
        from transformers import Seq2SeqTrainingArguments
        mapped = self._build_training_args_dict(
            fit_kwargs=fit_kwargs, output_dir=output_dir, logs_dir=logs_dir,
            force_overwrite=force_overwrite,
        )
        return Seq2SeqTrainingArguments(**mapped)

    # --- _translate (direct mode: writes src/ref/hyp itself) ------------

    def _translate(self, *, eval_ds, output_path, beam_width,
                   filter_idx, fn_name, filter_fn, preprocess_fn,
                   force_overwrite, batch_size=None,
                   max_len_a=None, max_len_b=None, **kwargs):
        """Tokenize → generate → decode for one (subset, beam) pass.

        Writes ``src.txt`` / ``ref.txt`` / ``hyp.txt`` directly — the parent's
        SPM encode/decode round-trip doesn't apply to HF models.
        """
        del kwargs  # consume the rest of the PredictConfig kwargs we don't use
        self._ensure_loaded()
        batch_size = batch_size or 8

        src_output_file = os.path.join(output_path, "src.txt")
        ref_output_file = os.path.join(output_path, "ref.txt")
        hyp_output_file = os.path.join(output_path, "hyp.txt")

        # 1. Load source / reference text (post user preprocess_splits_fn, pre-SPM).
        src_lines, ref_lines = self._load_eval_text(eval_ds, filter_fn, fn_name)

        # 2. Apply user's predict-time preprocess_fn (normalization etc).
        if preprocess_fn:
            src_lines = _apply_preprocess(preprocess_fn, src_lines, eval_ds,
                                           input_lang=eval_ds.src_lang,
                                           vocab_lang=self.src_lang or eval_ds.src_lang)
            ref_lines = _apply_preprocess(preprocess_fn, ref_lines, eval_ds,
                                           input_lang=eval_ds.trg_lang,
                                           vocab_lang=self.trg_lang or eval_ds.trg_lang)

        # 3. Tokenize → generate → decode.
        start = time.time()
        hyp_lines = self._generate(src_lines, beam=beam_width, batch_size=batch_size,
                                    max_len_a=max_len_a, max_len_b=max_len_b)
        log.info(f"\t- HF generate ({len(src_lines)} lines): "
                 f"{time.time() - start:.2f}s")

        # 4. Write artifacts. score_translations() reads src/ref/hyp from here.
        write_file_lines(filename=src_output_file, lines=src_lines,
                         autoclean=True, insert_break_line=True)
        write_file_lines(filename=ref_output_file, lines=ref_lines,
                         autoclean=True, insert_break_line=True)
        write_file_lines(filename=hyp_output_file, lines=hyp_lines,
                         autoclean=True, insert_break_line=True)

        assert len(ref_lines) == len(hyp_lines), (
            f"ref/hyp line count mismatch ({len(ref_lines)} vs {len(hyp_lines)}). "
            f"This usually means generate() truncated or skipped some inputs."
        )

    @staticmethod
    def _load_eval_text(eval_ds, filter_fn, fn_name):
        """Read source / target text from the dataset's preprocessed-split stage."""
        src_path = eval_ds.get_splits_auto_path(f"{eval_ds.test_name}.{eval_ds.src_lang}")
        trg_path = eval_ds.get_splits_auto_path(f"{eval_ds.test_name}.{eval_ds.trg_lang}")
        src_lines = read_file_lines(filename=src_path, autoclean=True)
        trg_lines = read_file_lines(filename=trg_path, autoclean=True)
        if filter_fn:
            log.info(f"Filtering src/ref (split='{fn_name}')...")
            src_lines, trg_lines = filter_fn(src_lines, trg_lines, from_fn="translate")
        return src_lines, trg_lines

    def _generate(self, src_lines, beam, batch_size, max_len_a, max_len_b):
        device = self._resolved_device
        gen_kwargs = dict(self.generation_kwargs)
        gen_kwargs["num_beams"] = beam
        # Prevent the "Both `max_new_tokens` and `max_length` seem to have been
        # set" warning from transformers — we manage length via max_new_tokens.
        gen_kwargs.setdefault("max_length", None)

        # Sort inputs by length so each batch pads to a tight bound instead of
        # to the longest sentence in the dataset. Generation cost is dominated
        # by ``num_beams * batch * padded_len`` decoder steps, so eliminating
        # mixed-length batches typically saves 2–3× FLOPs on natural test
        # distributions. We restore the original order before returning so
        # callers see hyp[i] aligned with src[i].
        order = sorted(range(len(src_lines)), key=lambda j: len(src_lines[j]))
        sorted_src = [src_lines[j] for j in order]

        sorted_hyps: list = [None] * len(src_lines)
        for i in range(0, len(sorted_src), batch_size):
            batch = sorted_src[i:i + batch_size]
            inputs = self._tokenizer(batch, return_tensors="pt",
                                      padding=True, truncation=True).to(device)
            if max_len_a is not None and max_len_b is not None:
                input_len = inputs["input_ids"].shape[1]
                gen_kwargs["max_new_tokens"] = int(max_len_a * input_len + max_len_b)
            with torch.no_grad():
                out_ids = self._model.generate(**inputs, **gen_kwargs)
            decoded = self._tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            for k, hyp in enumerate(decoded):
                sorted_hyps[order[i + k]] = hyp
        return sorted_hyps


class _Seq2SeqTextDataset:
    """Lazy parallel-text torch Dataset that tokenizes on access.

    Holds the raw source / target strings in memory and tokenizes per item so
    we don't materialize a huge tensor up-front. The HF
    :class:`DataCollatorForSeq2Seq` pads dynamically per batch and replaces
    target pad ids with ``-100`` for the loss.
    """

    def __init__(self, src_lines, trg_lines, tokenizer,
                 max_source_length=1024, max_target_length=1024):
        assert len(src_lines) == len(trg_lines)
        self.src_lines = src_lines
        self.trg_lines = trg_lines
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src = self.src_lines[idx]
        trg = self.trg_lines[idx]
        enc = self.tokenizer(
            src, text_target=trg,
            truncation=True,
            max_length=self.max_source_length,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": enc["labels"],
        }


def _apply_preprocess(preprocess_fn, lines, ds, input_lang, vocab_lang):
    """Call the user's predict-time preprocess_fn against an in-memory list.

    The function signature matches the existing examples: ``fn(data, ds)`` where
    ``data["lines"]`` is the input. We accept either a list-of-strings return
    value (the common case in the examples) or a dict with a ``"lines"`` key.
    """
    data = {"lines": lines, "input_lang": input_lang, "vocab_lang": vocab_lang}
    out = preprocess_fn(data, ds)
    if isinstance(out, dict):
        return list(out.get("lines", lines))
    return list(out)
