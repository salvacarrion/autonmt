"""HuggingFace backends for language models (decoder-only & encoder-only).

Siblings of :class:`~autonmt.backends.huggingface.translation_engine.HuggingFaceTranslator`
(which is encoder–decoder / ``AutoModelForSeq2SeqLM``). These wrap the *other*
two model families so you can fine-tune real pretrained checkpoints from the Hub
inside AutoNMT's pipeline:

  * :class:`HuggingFaceCausalLM` — ``AutoModelForCausalLM`` (GPT-2, Llama, …).
    Counterpart of the native :class:`~autonmt.backends.lm.trainer.LMTrainer`.
  * :class:`HuggingFaceMaskedLM` — ``AutoModelForMaskedLM`` (BERT, RoBERTa, …).
    Counterpart of the native :class:`~autonmt.backends.lm.mlm_trainer.MLMTrainer`.

They consume an :class:`~autonmt.datasets.lm_corpus.LMCorpus` for *data* (reading
its split text), but tokenize with the model's **own** HF tokenizer — the
SentencePiece vocab AutoNMT trains is ignored, exactly as in the translation
backend. Verbs mirror the native LM trainers: ``fit`` / ``evaluate`` /
``generate`` (causal) or ``fill_mask`` (masked).

References
----------
Wolf et al. (2020). *Transformers: State-of-the-Art Natural Language Processing.*
[arXiv:1910.03771](https://arxiv.org/abs/1910.03771)
"""
import datetime
import importlib.util
import inspect
import os
import time
from typing import Optional

try:
    import transformers  # noqa: F401
    _TRANSFORMERS_AVAILABLE = True
    _TRANSFORMERS_IMPORT_ERROR = None
except ImportError as e:
    _TRANSFORMERS_AVAILABLE = False
    _TRANSFORMERS_IMPORT_ERROR = e

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False

from autonmt.backends._base.config import FitConfig, merge_config
from autonmt.backends._base.run_layout import RunLayout
from autonmt.utils.fileio import make_dir, read_file_lines
from autonmt.utils.logger import get_logger
from autonmt.utils.seed import manual_seed

log = get_logger(__name__)


class _ListDataset:
    """Minimal indexable dataset wrapping a list of feature dicts (for HF ``Trainer``)."""

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class _HuggingFaceLMBase:
    """Shared plumbing for the HF causal / masked LM backends."""

    ENGINE = "huggingface"

    def __init__(self, model_id: str, tokenizer_id: Optional[str] = None,
                 device: str = "auto", run_name: Optional[str] = None,
                 runs_dir: str = "runs", corpus=None,
                 generation_kwargs: Optional[dict] = None):
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "HuggingFace LM backends require the 'transformers' package. "
                "Install with:\n  pip install -e '.[hf-models]'   (or: pip install transformers)"
            ) from _TRANSFORMERS_IMPORT_ERROR
        if not _TORCH_AVAILABLE:
            raise ImportError("HuggingFace LM backends require PyTorch.")

        self.model_id = model_id
        self.tokenizer_id = tokenizer_id or model_id
        self.device = device
        self.corpus = corpus
        self.runs_dir = runs_dir or "runs/"
        self.run_name = run_name or datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        self._layout = RunLayout(runs_dir=self.runs_dir, run_name=self.run_name)
        self.generation_kwargs = dict(generation_kwargs or {})

        self._model = None
        self._tokenizer = None
        self._resolved_device: Optional[str] = None

    @classmethod
    def from_corpus(cls, corpus, *, run_prefix: str, model_id: str, **kwargs):
        """Bind to ``corpus``'s runs path (mirrors ``LMTrainer.from_corpus``)."""
        return cls(model_id=model_id, corpus=corpus,
                   runs_dir=corpus.get_runs_path(),
                   run_name=corpus.get_run_name(run_prefix), **kwargs)

    # --- hooks subclasses implement -------------------------------------

    def _auto_model_cls(self):
        raise NotImplementedError

    def _build_examples(self, corpus, split, block_size):
        raise NotImplementedError

    def _data_collator(self):
        raise NotImplementedError

    # --- paths ----------------------------------------------------------

    def get_model_checkpoints_path(self, fname: str = ""):
        return self._layout.checkpoints_path(fname)

    def get_model_logs_path(self, fname: str = ""):
        return self._layout.logs_path(fname)

    # --- loading --------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        from transformers import AutoTokenizer
        log.info(f"=> [HF-LM]: Loading tokenizer from {self.tokenizer_id!r}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)
        # Many causal tokenizers (e.g. GPT-2) ship without a pad token; reuse EOS.
        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        log.info(f"=> [HF-LM]: Loading model from {self.model_id!r}")
        self._model = self._auto_model_cls().from_pretrained(self.model_id)
        self._resolved_device = self._resolve_device(self.device)
        self._model = self._model.to(self._resolved_device)
        self._model.eval()
        log.info(f"\t- Loaded on device: {self._resolved_device}")

    # --- data helpers ---------------------------------------------------

    def _read_lines(self, corpus, fname):
        return read_file_lines(corpus.get_splits_path(fname), autoclean=True)

    def _encode_lines(self, lines):
        """Tokenize raw text lines to id lists (no padding, no tensors)."""
        return self._tokenizer(lines, add_special_tokens=False)["input_ids"]

    @staticmethod
    def _group_blocks(id_lists, block_size):
        """nanoGPT-style packing: concatenate and slice into fixed blocks."""
        stream = [tok for ids in id_lists for tok in ids]
        n_blocks = len(stream) // block_size
        if n_blocks < 1:
            raise ValueError(
                f"Corpus too small for block_size={block_size}: only {len(stream)} tokens."
            )
        return [stream[k * block_size:(k + 1) * block_size] for k in range(n_blocks)]

    # --- fit ------------------------------------------------------------

    def fit(self, corpus=None, config: FitConfig = None, block_size: int = 128, **kwargs):
        corpus = corpus or self.corpus
        if corpus is None:
            raise ValueError("fit() needs a corpus (pass it here or to the constructor).")
        self.corpus = corpus

        cfg, extra = merge_config(config, FitConfig, kwargs)
        block_size = extra.pop("block_size", block_size)

        checkpoints_dir = self.get_model_checkpoints_path()
        logs_path = self.get_model_logs_path()
        make_dir([checkpoints_dir, logs_path])

        if self._has_finetuned_checkpoint(checkpoints_dir) and not cfg["force_overwrite"]:
            log.info(f"\t- [Fit]: Skipped. A fine-tuned checkpoint already exists at "
                     f"{checkpoints_dir!r} (pass force_overwrite=True to retrain).")
            self.model_id = checkpoints_dir
            self.tokenizer_id = checkpoints_dir
            return

        if importlib.util.find_spec("accelerate") is None:
            raise ImportError(
                "Fine-tuning with the HuggingFace backends requires the 'accelerate' "
                "package (>=1.1). Install with:\n  pip install -e '.[hf-models]'"
            )

        self._ensure_loaded()
        manual_seed(seed=cfg["seed"])

        from transformers import Trainer
        train_ds = _ListDataset(self._build_examples(corpus, corpus.train_name, block_size))
        val_ds = _ListDataset(self._build_examples(corpus, corpus.val_name, block_size))
        log.info(f"\t- Examples: train={len(train_ds)}, val={len(val_ds)} (block_size={block_size})")

        training_args = self._build_training_args(cfg, checkpoints_dir, logs_path, cfg["force_overwrite"])
        # `tokenizer` was renamed to `processing_class` in transformers 4.46.
        tok_kw = ("processing_class" if "processing_class" in inspect.signature(Trainer.__init__).parameters
                  else "tokenizer")
        trainer = Trainer(
            model=self._model, args=training_args,
            train_dataset=train_ds, eval_dataset=val_ds,
            data_collator=self._data_collator(),
            **{tok_kw: self._tokenizer},
        )

        start = time.time()
        trainer.train()
        log.info(f"\t- Training time: {datetime.timedelta(seconds=time.time() - start)}")

        trainer.save_model(str(checkpoints_dir))
        self._tokenizer.save_pretrained(str(checkpoints_dir))
        self.model_id = checkpoints_dir
        self.tokenizer_id = checkpoints_dir
        log.info(f"\t- Saved fine-tuned model to {checkpoints_dir!r}")

    @staticmethod
    def _has_finetuned_checkpoint(checkpoints_dir) -> bool:
        return os.path.isdir(checkpoints_dir) and os.path.exists(os.path.join(checkpoints_dir, "config.json"))

    def _build_training_args_dict(self, fit_kwargs, output_dir, logs_dir, force_overwrite):
        """FitConfig → ``transformers.TrainingArguments`` kwargs (pure; testable)."""
        from transformers import TrainingArguments
        sig_params = inspect.signature(TrainingArguments.__init__).parameters
        eval_strategy_kw = ("eval_strategy" if "eval_strategy" in sig_params else "evaluation_strategy")

        save_best = fit_kwargs.get("save_best", True)
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
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "seed": fit_kwargs.get("seed", 42),
            "dataloader_num_workers": fit_kwargs.get("num_workers", 0),
            "logging_dir": str(logs_dir),
            "report_to": ["tensorboard"],
            "save_total_limit": 2,
        }
        user_overrides = fit_kwargs.get("hf_training_args") or {}
        for k in set(mapped) & set(user_overrides):
            log.warning(f"\t- hf_training_args override: {k}={mapped[k]!r} → {user_overrides[k]!r}")
        mapped.update(user_overrides)
        for k in [k for k in mapped if k not in sig_params]:
            log.warning(f"\t- Dropping unsupported TrainingArguments kwarg: {k!r}")
            mapped.pop(k)
        return mapped

    def _build_training_args(self, fit_kwargs, output_dir, logs_dir, force_overwrite):
        from transformers import TrainingArguments
        return TrainingArguments(**self._build_training_args_dict(
            fit_kwargs, output_dir, logs_dir, force_overwrite))

    # --- evaluate (manual loop → no Trainer / accelerate needed) --------

    def evaluate(self, corpus=None, split=None, block_size: int = 128,
                 batch_size: int = 8, accelerator: str = "auto"):
        """Perplexity on a split (+ masked accuracy for the masked backend)."""
        corpus = corpus or self.corpus
        split = split or corpus.val_name
        self.device = accelerator
        self._ensure_loaded()
        device = self._resolved_device

        examples = self._build_examples(corpus, split, block_size)
        collator = self._data_collator()

        total_loss, total_tokens = 0.0, 0
        extra_acc, extra_tokens = 0, 0
        with torch.no_grad():
            for i in range(0, len(examples), batch_size):
                batch = collator(examples[i:i + batch_size])
                batch = {k: v.to(device) for k, v in batch.items()}
                out = self._model(**batch)
                labels = batch["labels"]
                counted = int((labels != -100).sum().item())
                # out.loss is the mean over counted tokens; un-mean to aggregate.
                total_loss += float(out.loss.item()) * counted
                total_tokens += counted
                preds = out.logits.argmax(-1)
                extra_acc += int(((preds == labels) & (labels != -100)).sum().item())
                extra_tokens += counted

        total_tokens = max(total_tokens, 1)
        mean_nll = total_loss / total_tokens
        ppl = float(torch.exp(torch.tensor(mean_nll)))
        metrics = {"loss": mean_nll, "ppl": ppl, "tokens": total_tokens}
        if self.TASK == "masked":
            metrics["masked_acc"] = extra_acc / max(extra_tokens, 1)
        log.info(f"=> [Evaluate HF-{self.TASK}]: split={split!r} loss={mean_nll:.4f} ppl={ppl:.2f} "
                 f"({total_tokens} tokens)")
        return metrics


class HuggingFaceCausalLM(_HuggingFaceLMBase):
    """Decoder-only HF backend (``AutoModelForCausalLM``).

    Trains on a ``mode="text"`` corpus (packed next-token) or a ``mode="instruct"``
    corpus (prompt masked out of the loss). ``generate`` continues a prompt.

    Parameters
    ----------
    model_id : str
        Hub id or local path (e.g. ``"gpt2"``, ``"meta-llama/Llama-3.2-1B"``).
    """

    TASK = "causal"

    def _auto_model_cls(self):
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM

    def _build_examples(self, corpus, split, block_size):
        # Instruct: per-example, with the prompt span excluded from the loss.
        if getattr(corpus, "mode", "text") == "instruct":
            prompts = self._read_lines(corpus, f"{split}.prompt")
            completions = self._read_lines(corpus, f"{split}.completion")
            p_ids = self._encode_lines(prompts)
            c_ids = self._encode_lines(completions)
            eos = self._tokenizer.eos_token_id
            examples = []
            for p, c in zip(p_ids, c_ids):
                c = c + ([eos] if eos is not None else [])
                input_ids = p + c
                labels = [-100] * len(p) + list(c)        # supervise only the completion
                examples.append({"input_ids": input_ids, "labels": labels})
            return examples
        # Text: packed blocks, every position supervised.
        lines = self._read_lines(corpus, f"{split}.txt")
        blocks = self._group_blocks(self._encode_lines(lines), block_size)
        return [{"input_ids": b, "labels": list(b)} for b in blocks]

    def _data_collator(self):
        return _CausalCollator(self._tokenizer.pad_token_id or 0)

    def generate(self, prompt, max_new_tokens=64, corpus=None, accelerator="auto", **gen_kwargs):
        """Continue ``prompt`` with the model's own tokenizer + ``model.generate``."""
        self.device = accelerator
        self._ensure_loaded()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._resolved_device)
        gen = dict(self.generation_kwargs)
        gen.update(gen_kwargs)
        gen["max_new_tokens"] = max_new_tokens
        with torch.no_grad():
            out = self._model.generate(**inputs, **gen)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)


class HuggingFaceMaskedLM(_HuggingFaceLMBase):
    """Encoder-only HF backend (``AutoModelForMaskedLM``).

    Trains on a ``mode="mlm"`` (or ``"text"``) corpus; HF's
    ``DataCollatorForLanguageModeling`` applies the BERT 80/10/10 masking, so
    masking is handled for you. ``fill_mask`` predicts the masked positions.
    ``model_id`` is a Hub id or local path (e.g. ``"bert-base-uncased"``,
    ``"roberta-base"``).

    Parameters
    ----------
    mlm_probability : float, default 0.15
        Fraction of tokens masked per batch.
    """

    TASK = "masked"

    def __init__(self, *args, mlm_probability: float = 0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlm_probability = mlm_probability

    def _auto_model_cls(self):
        from transformers import AutoModelForMaskedLM
        return AutoModelForMaskedLM

    def _build_examples(self, corpus, split, block_size):
        lines = self._read_lines(corpus, f"{split}.txt")
        blocks = self._group_blocks(self._encode_lines(lines), block_size)
        # The collator adds labels by masking; just provide input_ids + attention.
        return [{"input_ids": b, "attention_mask": [1] * len(b)} for b in blocks]

    def _data_collator(self):
        from transformers import DataCollatorForLanguageModeling
        if self._tokenizer.mask_token is None:
            raise ValueError(
                f"Tokenizer {self.tokenizer_id!r} has no mask token; masked-LM training "
                f"needs one (use a BERT/RoBERTa-style tokenizer)."
            )
        return DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer, mlm=True, mlm_probability=self.mlm_probability)

    def fill_mask(self, text, top_k=1, accelerator="auto"):
        """Predict the token(s) at each ``tokenizer.mask_token`` in ``text``.

        Write the model's mask token in the text, e.g. ``"the [MASK] fox jumps"``
        for BERT.
        """
        self.device = accelerator
        self._ensure_loaded()
        inputs = self._tokenizer(text, return_tensors="pt").to(self._resolved_device)
        with torch.no_grad():
            logits = self._model(**inputs).logits[0]
        mask_positions = (inputs["input_ids"][0] == self._tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if mask_positions.numel() == 0:
            log.warning(f"fill_mask: no {self._tokenizer.mask_token!r} found in the input.")
            return []
        return [
            [self._tokenizer.decode([t]).strip() for t in logits[p].topk(top_k).indices.tolist()]
            for p in mask_positions.tolist()
        ]


class _CausalCollator:
    """Right-pads ``input_ids`` / ``labels`` for causal LM (labels padded with -100)."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention = [], [], []
        for f in features:
            ids = list(f["input_ids"])
            lab = list(f.get("labels", ids))
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            labels.append(lab + [-100] * pad)
            attention.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }
