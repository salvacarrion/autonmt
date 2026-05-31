"""Abstract translator pipeline shared by every backend.

The public surface is ``fit()`` / ``predict()``. Each method:

  * resolves the effective config (defaults < ``config=`` < explicit kwargs),
  * persists it to ``logs/config_*.json`` so runs are reproducible,
  * dispatches to abstract ``_train`` / ``_translate`` hooks.

Path computation lives in :class:`~autonmt.backends._base.run_layout.RunLayout`,
the seed helper in :mod:`autonmt.utils.seed`, the SPM encode/decode
round-trip in :class:`~autonmt.backends._base.spm_pipeline.SPMTranslatePipeline`,
and the report schema in :mod:`autonmt.reporting.report` — keeping this class
focused on orchestration.

Backends supply two hooks instead of touching ``self.src_vocab`` / ``self.tgt_vocab``
directly:

  * ``_get_lang_pair() -> (src_lang, tgt_lang)`` — drives evaluation filtering
    and the target-language argument handed to metric backends.
  * ``_get_run_metadata() -> RunMetadata`` — backend-specific keys for the
    per-run report (model arch, param counts, vocab info).

Backends with file-based subword tokenization (autonmt, fairseq) declare
``self._spm = SPMTranslatePipeline(...)`` in their constructor;
``translate()`` then delegates the encode/decode bookkeeping to it. Backends
with their own tokenizer (huggingface) leave ``self._spm = None`` and write
``src.txt`` / ``ref.txt`` / ``hyp.txt`` directly from ``_translate``.
"""
import datetime
import importlib
import os.path
import platform
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from autonmt.utils.enums import EvalMode
from autonmt.utils.fileio import make_dir, save_json
from autonmt.utils.logger import get_logger
from autonmt.evaluation.metrics import METRIC_BACKENDS, resolve_backends
from autonmt.reporting.schema import RunMetadata, build_run_report
from autonmt.utils.seed import manual_seed
from autonmt.datasets.dataset import Dataset
from autonmt.backends._base.config import FitConfig, PredictConfig, UNSET, merge_config
from autonmt.backends._base.run_layout import RunLayout
from autonmt.backends._base.spm_pipeline import SPMTranslatePipeline

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers (free functions kept at module level so they're easy to test)
# ---------------------------------------------------------------------------

def _is_debug_enabled() -> bool:
    """True when running under a debugger (sys.settrace is set)."""
    gettrace = getattr(sys, 'gettrace', None)
    return bool(gettrace and gettrace())


def _all_supported_metric_names() -> Set[str]:
    return {m for b in METRIC_BACKENDS.values() for m in b.metrics}


def _check_datasets(train_ds: Optional[Dataset] = None, eval_ds: Optional[Dataset] = None) -> None:
    if train_ds and not isinstance(train_ds, Dataset):
        raise TypeError("'train_ds' must be an instance of 'Dataset' so that we can know the layout of the trained "
                        "model (e.g. checkpoints available, subword model, vocabularies, etc")
    if eval_ds and not isinstance(eval_ds, Dataset):
        raise TypeError("'eval_ds' must be an instance of 'Dataset' so that we can know the layout of the dataset "
                        "and get the corresponding data (e.g. splits, pretokenized, encoded, stc)")

    if train_ds and eval_ds and ((train_ds.src_lang != eval_ds.src_lang)
                                  or (train_ds.tgt_lang != eval_ds.tgt_lang)):
        raise ValueError(f"The languages from the train and test datasets are not compatible:\n"
                         f"\t- train_lang_pair=({train_ds.dataset_lang_pair})\n"
                         f"\t- test_lang_pair=({eval_ds.dataset_lang_pair})\n")


def _collect_environment() -> Dict[str, Any]:
    """Snapshot what a future researcher would need to reproduce this run:
    python/package versions, working dir, argv, and git SHA + dirty flag of
    the cwd when it's a repo. Failures are silent — env capture is best-effort
    and must never break a training run."""
    env: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
    }
    for mod_name in ("torch", "pytorch_lightning", "transformers", "sentencepiece",
                     "sacrebleu", "autonmt"):
        try:
            env[mod_name] = getattr(importlib.import_module(mod_name), "__version__", "unknown")
        except Exception:
            pass
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL, timeout=2).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"],
                                        stderr=subprocess.DEVNULL, timeout=2).decode().strip()
        env["git_sha"] = sha
        env["git_dirty"] = bool(dirty)
    except Exception:
        pass
    return env


def _check_supported_metrics(metrics: Iterable[str], metrics_supported: Iterable[str]) -> Set[str]:
    metrics = set(metrics)
    supported = set(metrics_supported)

    valid = list(metrics.intersection(supported))
    valid += [m for m in metrics if m.startswith("hg_")]  # HuggingFace wildcard
    valid = set(valid)

    non_valid = metrics.difference(valid)
    if non_valid:
        log.warning(f"=> These metrics are not supported: {str(non_valid)}")
        if metrics == non_valid:
            log.info("\t- [Score]: Skipped. No valid metrics were found.")
    return valid


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------

class BaseTranslator(ABC):

    total_runs = 0
    # Toolkit identifier used to compute the on-disk runs path (e.g. when the
    # caller uses :meth:`from_dataset`). Subclasses override.
    ENGINE: str = "base"

    @classmethod
    def from_dataset(cls, train_ds: Dataset, *, run_prefix: str, **kwargs) -> "BaseTranslator":
        """Build a translator bound to ``train_ds``'s runs path.

        Resolves ``runs_dir`` and ``run_name`` from the dataset variant and
        forwards everything else (``model=...``, ``src_vocab=...``, …) to the
        normal constructor. Equivalent to::

            cls(
                runs_dir=train_ds.get_runs_path(toolkit=cls.ENGINE),
                run_name=train_ds.get_run_name(run_prefix=run_prefix),
                **kwargs,
            )
        """
        return cls(
            runs_dir=train_ds.get_runs_path(toolkit=cls.ENGINE),
            run_name=train_ds.get_run_name(run_prefix=run_prefix),
            **kwargs,
        )

    def __init__(self, engine=None, runs_dir="runs", run_name=None, src_vocab=None, tgt_vocab=None,
                 train_subset=None, val_subsets=None, test_subsets=None,
                 safe_seconds=3, **kwargs):
        # Seed config with an environment snapshot so config_*.json captures
        # the python/package/git state at construction time — without this
        # every saved config is silently ambiguous about what code produced it.
        self.config: Dict[str, Dict] = {"environment": _collect_environment()}
        # ``engine`` defaults to the subclass's ENGINE classvar (used for the
        # on-disk runs path); only override when constructing a bare base.
        self.engine = engine or type(self).ENGINE
        self.runs_dir = runs_dir or "runs/"
        self.run_name = run_name or f"{datetime.datetime.now().strftime('%Y.%m.%d_%H.%M.%S')}"
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.from_checkpoint = None
        self.safe_seconds = safe_seconds
        self.trained_ds: List[Dataset] = []

        # Subsets: a list of (name, callable) entries. Each entry produces one
        # translate/score pass, useful when the user wants to slice the test set
        # by some property (e.g. per language pair in multilingual training).
        self.train_subset: Tuple[str, Optional[Any]] = train_subset or ('', None)
        self.val_subsets: List[Tuple[str, Optional[Any]]] = list(val_subsets or [('', None)])
        self.test_subsets: List[Tuple[str, Optional[Any]]] = list(test_subsets or [('', None)])

        # Run-side path engine.
        self._layout = RunLayout(runs_dir=self.runs_dir, run_name=self.run_name)

        # Set by subclasses that need the SPM encode/decode round-trip
        # (autonmt, fairseq). Backends with their own tokenizer (huggingface)
        # leave it None and write src.txt/ref.txt/hyp.txt directly from
        # _translate. See translate() for the branching logic.
        self._spm: Optional[SPMTranslatePipeline] = None

    # --- Config persistence ---------------------------------------------

    def _add_config(self, key: str, values: dict, reset: bool = False) -> None:
        # Previously this silently dropped anything non-primitive (callables,
        # objects), which hid reproducibility-relevant info like which
        # preprocess_fn or decoder a run used. Now we keep everything and
        # render callables as ``module.qualname`` so the JSON stays readable.
        def parse_value(x):
            if isinstance(x, (list, set, tuple)):
                return [parse_value(_x) for _x in x]
            if callable(x):
                mod = getattr(x, "__module__", "?")
                name = getattr(x, "__qualname__", None) or getattr(x, "__name__", None) or repr(x)
                return f"{mod}.{name}"
            return str(x)

        if reset or key not in self.config:
            self.config[key] = {}
        self.config[key].update(
            {k: parse_value(v) for k, v in values.items()
             if not (k.startswith("_") or k == "kwargs")})

    def _save_config(self, fname: str = "config.json") -> None:
        # Config files mirror *this* run's params; always overwrite so the
        # persisted JSON can't disagree with the artifacts produced in the
        # same call. (The user-facing ``force_overwrite`` flag gates expensive
        # outputs like checkpoints / translations, not run metadata.)
        logs_path = self.get_model_logs_path()
        make_dir(logs_path)
        save_json(self.config, savepath=os.path.join(logs_path, fname),
                  allow_overwrite=True)

    # --- Backend hooks --------------------------------------------------

    @abstractmethod
    def _get_lang_pair(self) -> Tuple[str, str]:
        """Return ``(src_lang, tgt_lang)``.

        Drives :meth:`filter_eval_datasets`, the ``tgt_lang`` argument handed
        to metric backends in :meth:`score_translations`, and the language
        pair recorded in the report.
        """

    def _get_run_metadata(self) -> RunMetadata:
        """Backend-specific metadata for the per-run report.

        Default returns an empty :class:`RunMetadata` — a backend without
        introspectable model/vocab still gets a valid (if minimal) report.
        """
        return RunMetadata()

    # --- fit ------------------------------------------------------------

    def fit(self, train_ds: Dataset, config: Optional[FitConfig] = None, **kwargs) -> None:
        """Train the model on a dataset variant.

        Parameters
        ----------
        train_ds : Dataset
            The dataset variant to train on (one cell of the builder grid).
        config : FitConfig, optional
            Training configuration. Any field may instead be passed directly as
            a keyword argument; explicit ``**kwargs`` win on collision. See
            :class:`~autonmt.backends._base.config.FitConfig`.
        **kwargs : Any
            Per-field overrides plus backend-specific options forwarded to the
            underlying toolkit.
        """
        log.info("=" * 70)
        log.info(f"=> [Fit]: {train_ds.variant_id(as_path=True)}  (run={self.run_name})")
        log.info("=" * 70)
        cfg, extra = merge_config(config, FitConfig, kwargs)

        self._add_config(key="fit", values=cfg, reset=False)
        self._add_config(key="fit", values=extra, reset=False)
        self._save_config(fname="config_train.json")

        self.train(train_ds, **cfg, **extra)

    # --- predict --------------------------------------------------------

    def predict(self, eval_datasets: Iterable[Dataset],
                config: Optional[PredictConfig] = None, **kwargs) -> List[dict]:
        """Translate and score one or more evaluation datasets.

        Runs translation (one pass per beam width), scores the hypotheses with
        the requested metrics, and returns the parsed scores.

        Parameters
        ----------
        eval_datasets : Iterable[Dataset]
            Candidate test sets; filtered by ``eval_mode`` to those compatible
            with the trained model.
        config : PredictConfig, optional
            Prediction configuration. Fields may instead be passed as keyword
            arguments; explicit ``**kwargs`` win on collision. See
            :class:`~autonmt.backends._base.config.PredictConfig`.
        **kwargs : Any
            Per-field overrides plus backend-specific options.

        Returns
        -------
        list of dict
            One nested score dict per evaluated dataset, with flattened keys of
            the form ``translations.beam<N>.<tool>_<metric>_<field>``.
        """
        log.info("=" * 70)
        log.info(f"=> [Predict]: (run={self.run_name})")
        log.info("=" * 70)
        cfg, extra = merge_config(config, PredictConfig, kwargs)

        # Normalize defaults that need post-processing.
        cfg["beams"] = [1] if cfg["beams"] is None else list(sorted(set(cfg["beams"]), reverse=True))
        cfg["metrics"] = {"bleu"} if cfg["metrics"] is None else set(cfg["metrics"])

        self._add_config(key="predict", values=cfg, reset=False)
        self._add_config(key="predict", values=extra, reset=False)
        self._save_config(fname="config_predict.json")

        scores = []
        eval_datasets = self.filter_eval_datasets(eval_datasets, eval_mode=cfg["eval_mode"])
        if not eval_datasets:
            log.info(f"=> [Predict]: Skipped. No valid test datasets were found.")

        for eval_ds in eval_datasets:
            self.translate(eval_ds, beams=cfg["beams"],
                           max_len_a=cfg["max_len_a"], max_len_b=cfg["max_len_b"],
                           batch_size=cfg["batch_size"], max_tokens=cfg["max_tokens"],
                           devices=cfg["devices"], accelerator=cfg["accelerator"],
                           num_workers=cfg["num_workers"],
                           checkpoint=cfg["load_checkpoint"],
                           preprocess_fn=cfg["preprocess_fn"],
                           decoder=cfg["decoder"],
                           force_overwrite=cfg["force_overwrite"], **extra)
            self.score_translations(eval_ds, beams=cfg["beams"], metrics=cfg["metrics"],
                                    force_overwrite=cfg["force_overwrite"], **extra)
            run_scores = self.parse_metrics(eval_ds, beams=cfg["beams"], metrics=cfg["metrics"],
                                            **extra)
            scores.append(run_scores)
        return scores

    # --- Stages: train / translate (abstract hooks) ---------------------

    @abstractmethod
    def _train(self, *args, **kwargs):
        pass

    def _log_train_summary(self, train_ds, kwargs):
        """Generic training-config summary. Backends extend to add their own
        model/vocab line — the default deliberately avoids vocab access so
        backends without an AutoNMT :class:`Vocabulary` (e.g. huggingface)
        still get a useful summary out of the box."""
        def _kv(k, default="-"):
            v = kwargs.get(k)
            return default if v is None else v

        log.info("\t- Config:")
        log.info(f"\t\t- training: epochs={_kv('max_epochs')}, "
                 f"batch_size={_kv('batch_size')}, max_tokens={_kv('max_tokens')}, "
                 f"lr={_kv('learning_rate')}, optimizer={_kv('optimizer')}")
        log.info(f"\t\t- monitor: {_kv('monitor')} "
                 f"(patience={_kv('patience')}, save_best={_kv('save_best')}, save_last={_kv('save_last')})")
        log.info(f"\t\t- device: accelerator={_kv('accelerator')}, devices={_kv('devices')}, "
                 f"num_workers={_kv('num_workers')}, seed={_kv('seed')}")

    def train(self, train_ds, force_overwrite, **kwargs):
        log.info(f"=> [Train]: Started. ({train_ds.variant_id(as_path=True)})")
        _check_datasets(train_ds=train_ds)

        if _is_debug_enabled():
            log.warning("\t=> Debug is enabled. This could lead to critical problems when using a data parallel strategy.")

        self._log_train_summary(train_ds, kwargs)

        self.trained_ds.append(train_ds)
        checkpoints_dir = self.get_model_checkpoints_path()
        logs_path = self.get_model_logs_path()
        make_dir([checkpoints_dir, logs_path])

        manual_seed(seed=kwargs.get("seed"))

        start = time.time()
        self._train(train_ds=train_ds, checkpoints_dir=checkpoints_dir, logs_path=logs_path,
                    force_overwrite=force_overwrite, **kwargs)
        log.info(f"\t- Training time: {datetime.timedelta(seconds=time.time() - start)}")

    @abstractmethod
    def _translate(self, *args, **kwargs):
        """Toolkit-specific translate hook. Contract depends on whether the
        backend wired an SPMTranslatePipeline:

        - With SPM (autonmt, fairseq): produce ``hyp.tok`` in ``output_path``;
          the pipeline handles decoding it into ``hyp.txt`` and materializing
          ``src.txt`` / ``ref.txt``.
        - Without SPM (huggingface): produce ``src.txt`` / ``ref.txt`` /
          ``hyp.txt`` directly in ``output_path``. The hook receives
          ``eval_ds`` and the per-subset filter so it can slice the raw test
          files itself.
        """

    # --- translate ------------------------------------------------------

    def translate(self, eval_ds, beams, preprocess_fn, force_overwrite, **kwargs):
        """Public orchestrator. Two execution modes:

        - SPM-pipeline mode (autonmt, fairseq): the pipeline runs
          encode → ``_translate`` → decode → materialize src/ref.
          ``_translate`` only writes ``hyp.tok``.
        - Direct mode (huggingface): loop over (subset, beam) and call
          ``_translate``, which writes src.txt/ref.txt/hyp.txt itself.
        """
        log.info(f"=> [Translate]: Started. (Model: {self.run_name} | Test: {str(eval_ds)})")
        _check_datasets(eval_ds=eval_ds)

        if self._spm is not None:
            self._spm.run(
                eval_ds=eval_ds, beams=beams,
                preprocess_fn=preprocess_fn, force_overwrite=force_overwrite,
                translate_fn=self._translate,
                preprocess_eval_fn=getattr(self, "_prepare_eval_data", None),
                **kwargs,
            )
            return

        for filter_idx, (fn_name, filter_fn) in enumerate(self.test_subsets):
            for beam in beams:
                output_path = self.get_model_eval_translations_beam_path(
                    eval_name=str(eval_ds), split_name=fn_name, beam=beam)
                make_dir(output_path)
                if (not force_overwrite
                        and os.path.exists(os.path.join(output_path, "hyp.txt"))):
                    log.info(f"\t- [Translate]: Skipped (cached) beam={beam}"
                             + (f" | split='{fn_name}'" if fn_name else ""))
                    continue
                start = time.time()
                self._translate(
                    eval_ds=eval_ds, output_path=output_path,
                    beam_width=beam, filter_idx=filter_idx,
                    fn_name=fn_name, filter_fn=filter_fn,
                    preprocess_fn=preprocess_fn,
                    force_overwrite=force_overwrite, **kwargs,
                )
                extra_str = f" | split='{fn_name}'" if fn_name else ""
                log.info(f"\t- Translating time (beam={beam}{extra_str}): "
                         f"{datetime.timedelta(seconds=time.time() - start)}")

    # --- score_translations + parse_metrics -----------------------------

    def score_translations(self, eval_ds: Dataset, beams: List[int], metrics: Set[str],
                           force_overwrite, **kwargs):
        log.info(f"=> [Scoring translations]: Started. (Model: {self.run_name} | Test: {str(eval_ds)})")
        _check_datasets(eval_ds=eval_ds)

        valid = _check_supported_metrics(metrics, _all_supported_metric_names())
        if not valid:
            return

        grouped = resolve_backends(metrics)
        if not grouped:
            return

        _, tgt_lang = self._get_lang_pair()

        for fn_name, _ in self.test_subsets:
            extra_str = f" | split='{fn_name}'" if fn_name else ""
            for beam in beams:
                start = time.time()
                beam_path = self.get_model_eval_translations_beam_path(
                    eval_name=str(eval_ds), split_name=fn_name, beam=beam)
                scores_path = self.get_model_eval_translations_beam_scores_path(
                    eval_name=str(eval_ds), split_name=fn_name, beam=beam)
                make_dir([scores_path])

                files = {
                    "src_file": os.path.join(beam_path, "src.txt"),
                    "ref_file": os.path.join(beam_path, "ref.txt"),
                    "hyp_file": os.path.join(beam_path, "hyp.txt"),
                }
                # ref and hyp are always required; src only if a backend uses it
                # (currently COMET).
                required = {"ref_file": files["ref_file"], "hyp_file": files["hyp_file"]}
                if any(b.needs_src for b in grouped):
                    required["src_file"] = files["src_file"]
                missing = [p for p in required.values() if not os.path.exists(p)]
                if missing:
                    raise IOError(f"Missing files required to compute scores: {missing}")

                for backend, backend_metrics in grouped.items():
                    output_file = os.path.join(scores_path, backend.output_filename)
                    if not force_overwrite and os.path.exists(output_file):
                        continue
                    backend.compute_fn(
                        **files, output_file=output_file, metrics=backend_metrics,
                        tgt_lang=tgt_lang,
                    )

                log.info(f"\t- Scoring time (beam={beam}{extra_str}): "
                         f"{datetime.timedelta(seconds=time.time() - start)}")

    def parse_metrics(self, eval_ds, beams, metrics, **kwargs):
        log.info(f"=> [Parsing]: Started. ({str(eval_ds)})")
        _check_datasets(eval_ds=eval_ds)

        valid = _check_supported_metrics(metrics, _all_supported_metric_names())
        if not valid:
            return

        backends = list(resolve_backends(metrics).keys())
        translations: Dict[str, Dict] = {}

        for fn_name, _ in self.test_subsets:
            extra_str = f" | split='{fn_name}'" if fn_name else ""
            for beam in beams:
                start = time.time()
                scores_path = self.get_model_eval_translations_beam_scores_path(
                    eval_name=str(eval_ds), split_name=fn_name, beam=beam)

                beam_scores = self._parse_beam_scores(scores_path, backends)

                entry = {f"beam{beam}": beam_scores}
                entry = {fn_name: entry} if fn_name else entry
                translations.update(entry)
                log.info(f"\t- Parsed time (beam={beam}{extra_str}): "
                         f"{datetime.timedelta(seconds=time.time() - start)}")

        return self._build_run_report(eval_ds=eval_ds, translations=translations)

    def _build_run_report(self, eval_ds, translations) -> Dict:
        """Assemble the per-(run, eval_ds) report dict via the reporting layer.

        Backends customise the model/vocab keys by overriding
        :meth:`_get_run_metadata`. The dataset/config/translations bits come
        straight from ``self`` / ``eval_ds`` and don't need a hook.
        """
        return build_run_report(
            engine=self.engine, run_name=self.run_name,
            eval_ds=eval_ds,
            config=self.config, translations=translations,
            metadata=self._get_run_metadata(),
            train_ds=self.trained_ds[-1] if self.trained_ds else None,
        )

    @staticmethod
    def _parse_beam_scores(scores_path, backends) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for backend in backends:
            filename = os.path.join(scores_path, backend.output_filename)
            if not os.path.exists(filename):
                log.warning(f"\t- There are no metrics from '{backend.name}'")
                continue
            try:
                with open(filename, 'r') as f:
                    m_scores = backend.parse_fn(f.readlines())
                for m_name, m_values in m_scores.items():
                    for score_name, score_value in m_values.items():
                        key = f"{backend.name}_{m_name}_{score_name}".lower().strip()
                        scores[key] = score_value
            except Exception as e:
                log.warning(f"\t- [PARSING ERROR]: ({filename}) {e}")
        return scores

    # --- Eval-dataset filtering -----------------------------------------

    def filter_eval_datasets(self, ts_datasets, eval_mode):
        eval_mode = EvalMode.coerce(eval_mode)
        if eval_mode is EvalMode.ALL:
            return list(ts_datasets)
        if eval_mode is EvalMode.SAME:
            trained_ds = {str(ds.base_id()) for ds in self.trained_ds}
            return [ds for ds in ts_datasets if str(ds.base_id()) in trained_ds]
        if eval_mode is EvalMode.COMPATIBLE:
            src_lang, tgt_lang = self._get_lang_pair()
            langs = {src_lang, tgt_lang}
            return [ds for ds in ts_datasets if set(ds.langs).issubset(langs)]
        raise ValueError(f"Unknown 'eval_mode' ({str(eval_mode)})")

    # --- Path accessors (delegate to RunLayout) -------------------------

    def get_model_eval_translations_beam_path(self, eval_name, split_name, beam, fname=""):
        return self._layout.beam_path(eval_name, split_name, beam, fname)

    def get_model_eval_translations_beam_scores_path(self, eval_name, split_name, beam, fname=""):
        return self._layout.beam_scores_path(eval_name, split_name, beam, fname)

    def get_model_logs_path(self, fname=""):
        return self._layout.logs_path(fname)

    def get_model_checkpoints_path(self, fname=""):
        return self._layout.checkpoints_path(fname)

    # --- Static utilities used by examples (re-exported for compatibility) ---

    @staticmethod
    def manual_seed(seed=None, use_deterministic_algorithms=False):
        return manual_seed(seed=seed, use_deterministic_algorithms=use_deterministic_algorithms)


# UNSET re-exported here so ``from autonmt.backends._base.translation_engine import UNSET`` works.
__all__ = ["BaseTranslator", "UNSET"]
