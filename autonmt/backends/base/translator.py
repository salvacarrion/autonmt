"""Abstract translator pipeline shared by every backend.

The public surface is ``fit()`` / ``predict()``. Each method:

  * resolves the effective config (defaults < ``config=`` < explicit kwargs),
  * persists it to ``logs/config_*.json`` so runs are reproducible,
  * dispatches to abstract ``_preprocess`` / ``_train`` / ``_translate`` hooks.

Path computation lives in :class:`~autonmt.backends.base.run_layout.RunLayout`,
the per-translate context in :class:`TranslateContext`, the seed helper in
:mod:`autonmt.utils.seed`, and the report dict schema in
:mod:`autonmt.reporting.report` — keeping this class focused on orchestration.
"""
import datetime
import os.path
import shutil
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from autonmt.utils.enums import EvalMode
from autonmt.utils.fileio import count_file_lines, make_dir, read_file_lines, save_json, write_file_lines
from autonmt.utils.logger import get_logger
from autonmt.evaluation.metrics import METRIC_BACKENDS, resolve_backends
from autonmt.reporting.report import build_run_report
from autonmt.utils.seed import manual_seed
from autonmt.datasets.dataset import Dataset
from autonmt.datasets.processors import decode_file, encode_file, preprocess_predict_file
from autonmt.backends.base.config import FitConfig, PredictConfig, UNSET, merge_config
from autonmt.backends.base.run_layout import RunLayout

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
                                  or (train_ds.trg_lang != eval_ds.trg_lang)):
        raise ValueError(f"The languages from the train and test datasets are not compatible:\n"
                         f"\t- train_lang_pair=({train_ds.dataset_lang_pair})\n"
                         f"\t- test_lang_pair=({eval_ds.dataset_lang_pair})\n")


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
# TranslateContext: snapshot of paths + per-language metadata for one eval_ds
# ---------------------------------------------------------------------------

@dataclass
class TranslateContext:
    """Per-eval snapshot consumed by every beam pass of one translate() call.

    Replaces the dict-of-strings pseudo-object the previous translator built —
    now the field names are checked statically and `.x` access is cheap.
    """
    checkpoints_dir: str
    model_eval_path: str
    dst_raw_path: str
    dst_preprocessed_path: str
    dst_encoded_path: str

    model_src_vocab_path: Optional[str]
    model_trg_vocab_path: Optional[str]

    vocab_langs: Tuple[str, str]
    pretok_flags: Dict[str, bool] = field(default_factory=dict)
    vocab_paths: Dict[str, str] = field(default_factory=dict)
    subword_models: Dict[str, Any] = field(default_factory=dict)


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

    def __init__(self, engine, runs_dir="runs", run_name=None, src_vocab=None, trg_vocab=None,
                 train_subset=None, val_subsets=None, test_subsets=None,
                 safe_seconds=3,
                 # Backwards-compat aliases for the old filter_*_data_fn API.
                 filter_tr_data_fn=None, filter_vl_data_fn=None, filter_ts_data_fn=None,
                 **kwargs):
        self.config: Dict[str, Dict] = {}
        self.engine = engine
        self.runs_dir = runs_dir or "runs/"
        self.run_name = run_name or f"{datetime.datetime.now().strftime('%Y.%m.%d_%H.%M.%S')}"
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.from_checkpoint = None
        self.safe_seconds = safe_seconds
        self.trained_ds: List[Dataset] = []

        # Subsets: a list of (name, callable) entries. Each entry produces one
        # translate/score pass, useful when the user wants to slice the test set
        # by some property (e.g. per language pair in multilingual training).
        # Accepts either the new ``*_subsets`` names or the legacy ``filter_*_data_fn``
        # names; they are exactly the same shape.
        self.train_subset: Tuple[str, Optional[Any]] = (
            (train_subset or filter_tr_data_fn) or ('', None)
        )
        self.val_subsets: List[Tuple[str, Optional[Any]]] = list(
            val_subsets or filter_vl_data_fn or [('', None)]
        )
        self.test_subsets: List[Tuple[str, Optional[Any]]] = list(
            test_subsets or filter_ts_data_fn or [('', None)]
        )

        # Run-side path engine.
        self._layout = RunLayout(runs_dir=self.runs_dir, run_name=self.run_name)

    # --- Aliases (preserve the original attribute names for examples) ----

    @property
    def filter_tr_data_fn(self):
        return self.train_subset

    @property
    def filter_vl_data_fn(self):
        return self.val_subsets

    @property
    def filter_ts_data_fn(self):
        return self.test_subsets

    # --- Config persistence ---------------------------------------------

    def _add_config(self, key: str, values: dict, reset: bool = False) -> None:
        primitive_types = (str, bool, int, float, dict, set, list)

        def is_valid(k, v):
            return (not (k.startswith("_") or k == "kwargs")
                    and (isinstance(v, primitive_types) or v is None))

        def parse_value(x):
            if isinstance(x, (list, set)):
                return [str(_x) for _x in x]
            return str(x)

        if reset or key not in self.config:
            self.config[key] = {}
        self.config[key].update(
            {k: parse_value(v) for k, v in values.items() if is_valid(k, v)})

    def _save_config(self, fname: str = "config.json") -> None:
        # Config files mirror *this* run's params; always overwrite so the
        # persisted JSON can't disagree with the artifacts produced in the
        # same call. (The user-facing ``force_overwrite`` flag gates expensive
        # outputs like checkpoints / translations, not run metadata.)
        logs_path = self.get_model_logs_path()
        make_dir(logs_path)
        save_json(self.config, savepath=os.path.join(logs_path, fname),
                  allow_overwrite=True)

    # --- fit ------------------------------------------------------------

    def fit(self, train_ds: Dataset, config: Optional[FitConfig] = None, **kwargs) -> None:
        log.info("=" * 70)
        log.info(f"=> [Fit]: {train_ds.id2(as_path=True)}  (run={self.run_name})")
        log.info("=" * 70)
        cfg, extra = merge_config(config, FitConfig, kwargs)

        self._add_config(key="fit", values=cfg, reset=False)
        self._add_config(key="fit", values=extra, reset=False)
        self._save_config(fname="config_train.json")

        self.preprocess(train_ds, apply2train=True, apply2val=True, apply2test=False,
                        force_overwrite=cfg["force_overwrite"], **extra)
        self.train(train_ds, **cfg, **extra)

    # --- predict --------------------------------------------------------

    def predict(self, eval_datasets: Iterable[Dataset],
                config: Optional[PredictConfig] = None, **kwargs) -> List[dict]:
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

    # --- Stages: preprocess / train / translate (abstract hooks) ---------

    @abstractmethod
    def _preprocess(self, *args, **kwargs):
        pass

    def preprocess(self, ds: Dataset, apply2train, apply2val, apply2test, force_overwrite, **kwargs):
        log.info(f"=> [Preprocess]: Started. ({ds.id2(as_path=True)})")

        start = time.time()
        self._preprocess(
            ds=ds, output_path=None,
            src_lang=ds.src_lang, trg_lang=ds.trg_lang,
            src_vocab_path=ds.get_vocab_file(lang=ds.src_lang),
            trg_vocab_path=ds.get_vocab_file(lang=ds.trg_lang),
            train_path=ds.get_encoded_path(fname=ds.train_name),
            val_path=ds.get_encoded_path(fname=ds.val_name),
            test_path=ds.get_encoded_path(fname=ds.test_name),
            apply2train=apply2train, apply2val=apply2val, apply2test=apply2test,
            force_overwrite=force_overwrite, **kwargs,
        )
        log.info(f"\t- Preprocess time: {datetime.timedelta(seconds=time.time() - start)}")

    @abstractmethod
    def _train(self, *args, **kwargs):
        pass

    def _log_train_summary(self, train_ds, kwargs):
        sw = self.src_vocab.subword_model
        v_src, v_trg = len(self.src_vocab), len(self.trg_vocab)
        vocab_line = f"src={v_src}, trg={v_trg}, subword={sw}"
        if getattr(train_ds, "merge_vocabs", False):
            vocab_line += ", merged=True"

        def _kv(k, default="-"):
            v = kwargs.get(k)
            return default if v is None else v

        log.info("\t- Config:")
        log.info(f"\t\t- vocab: {vocab_line}")
        log.info(f"\t\t- training: epochs={_kv('max_epochs')}, "
                 f"batch_size={_kv('batch_size')}, max_tokens={_kv('max_tokens')}, "
                 f"lr={_kv('learning_rate')}, optimizer={_kv('optimizer')}")
        log.info(f"\t\t- monitor: {_kv('monitor')} "
                 f"(patience={_kv('patience')}, save_best={_kv('save_best')}, save_last={_kv('save_last')})")
        log.info(f"\t\t- device: accelerator={_kv('accelerator')}, devices={_kv('devices')}, "
                 f"num_workers={_kv('num_workers')}, seed={_kv('seed')}")

    def train(self, train_ds, force_overwrite, **kwargs):
        log.info(f"=> [Train]: Started. ({train_ds.id2(as_path=True)})")
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
        pass

    # --- translate ------------------------------------------------------

    def translate(self, eval_ds, beams, preprocess_fn, force_overwrite, **kwargs):
        log.info(f"=> [Translate]: Started. (Model: {self.run_name} | Test: {str(eval_ds)})")
        _check_datasets(eval_ds=eval_ds)

        ctx = self._build_translate_context(eval_ds)
        self._encode_eval_test_data(eval_ds, ctx, preprocess_fn=preprocess_fn,
                                    force_overwrite=force_overwrite)
        self._preprocess_eval_test_data(eval_ds, ctx, force_overwrite=force_overwrite, **kwargs)

        for filter_idx, (fn_name, filter_fn) in enumerate(self.test_subsets):
            for beam in beams:
                self._translate_one_beam(eval_ds, ctx, beam=beam, filter_idx=filter_idx,
                                         filter_fn=filter_fn, fn_name=fn_name,
                                         preprocess_fn=preprocess_fn,
                                         force_overwrite=force_overwrite, **kwargs)

    def _build_translate_context(self, eval_ds) -> TranslateContext:
        model_eval_path = self.get_model_eval_path(eval_name=str(eval_ds))
        make_dir([model_eval_path])

        dst_raw_path = self._layout.eval_raw_path(str(eval_ds))
        dst_preprocessed_path = self._layout.eval_preprocessed_path(str(eval_ds))
        dst_encoded_path = self._layout.eval_encoded_path(str(eval_ds))
        make_dir([dst_raw_path, dst_preprocessed_path, dst_encoded_path])

        src_lang, trg_lang = self.src_vocab.lang, self.trg_vocab.lang
        return TranslateContext(
            checkpoints_dir=self.get_model_checkpoints_path(),
            model_eval_path=model_eval_path,
            dst_raw_path=dst_raw_path,
            dst_preprocessed_path=dst_preprocessed_path,
            dst_encoded_path=dst_encoded_path,
            model_src_vocab_path=self.src_vocab.vocab_path,
            model_trg_vocab_path=self.trg_vocab.vocab_path,
            vocab_langs=(src_lang, trg_lang),
            pretok_flags={src_lang: self.src_vocab.pretok_flag,
                          trg_lang: self.trg_vocab.pretok_flag},
            vocab_paths={src_lang: self.src_vocab.model_path,
                         trg_lang: self.trg_vocab.model_path},
            subword_models={src_lang: self.src_vocab.subword_model,
                            trg_lang: self.trg_vocab.subword_model},
        )

    def _encode_eval_test_data(self, eval_ds, ctx: TranslateContext, preprocess_fn, force_overwrite):
        """Copy raw test files, run preprocess_fn (+ pretokenization), then subword-encode."""
        test_fnames = [f"{eval_ds.test_name}.{eval_ds.src_lang}",
                       f"{eval_ds.test_name}.{eval_ds.trg_lang}"]
        for i, ts_fname in enumerate(test_fnames):
            input_file = eval_ds.get_split_path(ts_fname)
            input_lang = ts_fname.split(".")[-1]
            vocab_lang = ctx.vocab_langs[i]

            # 1 - copy raw
            source_file = os.path.join(ctx.dst_raw_path, ts_fname)
            if force_overwrite or not os.path.exists(source_file):
                shutil.copyfile(input_file, source_file)
                assert os.path.exists(source_file)
                input_file = source_file

            # 2 - preprocess (+ pretokenize)
            preprocessed_file = os.path.join(ctx.dst_preprocessed_path, ts_fname)
            preprocess_predict_file(
                input_file=input_file, output_file=preprocessed_file,
                preprocess_fn=preprocess_fn,
                pretokenize=ctx.pretok_flags[vocab_lang],
                input_lang=input_lang, vocab_lang=vocab_lang,
                ds=eval_ds, force_overwrite=force_overwrite,
            )

            # 3 - subword encode
            enc_file = os.path.join(ctx.dst_encoded_path, ts_fname)
            encode_file(
                input_file=preprocessed_file, output_file=enc_file,
                model_vocab_path=ctx.vocab_paths[vocab_lang],
                subword_model=ctx.subword_models[vocab_lang],
                force_overwrite=force_overwrite,
            )

    def _preprocess_eval_test_data(self, eval_ds, ctx: TranslateContext, force_overwrite, **kwargs):
        test_path = os.path.join(ctx.dst_encoded_path, eval_ds.test_name)
        self._preprocess(
            train_path=None, val_path=None, test_path=test_path,
            src_lang=eval_ds.src_lang, trg_lang=eval_ds.trg_lang,
            src_vocab_path=ctx.model_src_vocab_path,
            trg_vocab_path=ctx.model_trg_vocab_path,
            apply2train=False, apply2val=False, apply2test=True,
            output_path=ctx.model_eval_path,
            force_overwrite=force_overwrite, **kwargs,
        )

    def _translate_one_beam(self, eval_ds, ctx: TranslateContext, beam, filter_idx, filter_fn,
                            fn_name, preprocess_fn, force_overwrite, **kwargs):
        extra_str = f" | split='{fn_name}'" if fn_name else ""
        output_path = self.get_model_eval_translations_beam_path(
            eval_name=str(eval_ds), split_name=fn_name, beam=beam)
        make_dir(output_path)

        if not force_overwrite and os.path.exists(os.path.join(output_path, "hyp.tok")):
            return

        start = time.time()
        # NOTE: ``force_overwrite`` is *not* forwarded to ``_translate`` — the
        # cache gate above (hyp.tok existence) is the single source of truth. The
        # backend's job is just to produce the artifacts.
        self._translate(
            data_path=ctx.model_eval_path, output_path=output_path,
            src_lang=eval_ds.src_lang, trg_lang=eval_ds.trg_lang,
            beam_width=beam, checkpoints_dir=ctx.checkpoints_dir,
            model_src_vocab_path=ctx.model_src_vocab_path,
            model_trg_vocab_path=ctx.model_trg_vocab_path,
            filter_idx=filter_idx, **kwargs,
        )

        src_output_file = os.path.join(output_path, "src.txt")
        ref_output_file = os.path.join(output_path, "ref.txt")
        hyp_output_file = os.path.join(output_path, "hyp.txt")

        self._decode_hypothesis(ctx, output_path, hyp_output_file, force_overwrite)
        self._materialize_src_ref(eval_ds, ctx, src_output_file, ref_output_file,
                                  filter_fn=filter_fn, fn_name=fn_name)
        if preprocess_fn:
            self._postprocess_eval_files(eval_ds, ctx, preprocess_fn,
                                         src_output_file, ref_output_file, hyp_output_file)
        self._assert_ref_hyp_line_count(output_path)

        log.info(f"\t- Translating time (beam={beam}{extra_str}): "
                 f"{datetime.timedelta(seconds=time.time() - start)}")

    def _decode_hypothesis(self, ctx: TranslateContext, output_path, hyp_output_file, force_overwrite):
        model_lang = self.trg_vocab.lang
        hyp_input_file = os.path.join(output_path, "hyp.tok")
        decode_file(
            input_file=hyp_input_file, output_file=hyp_output_file, lang=model_lang,
            subword_model=ctx.subword_models[model_lang],
            pretok_flag=ctx.pretok_flags[model_lang],
            model_vocab_path=ctx.vocab_paths[model_lang],
            remove_unk_hyphen=True, force_overwrite=force_overwrite,
        )

    def _materialize_src_ref(self, eval_ds, ctx: TranslateContext, src_output_file, ref_output_file,
                             filter_fn, fn_name):
        src_input_file = os.path.join(ctx.dst_raw_path,
                                      f"{eval_ds.test_name}.{eval_ds.src_lang}")
        ref_input_file = os.path.join(ctx.dst_raw_path,
                                      f"{eval_ds.test_name}.{eval_ds.trg_lang}")
        if not filter_fn:
            shutil.copyfile(src_input_file, src_output_file)
            shutil.copyfile(ref_input_file, ref_output_file)
            return

        log.info(f"Filtering src/ref raw files (split='{fn_name}')...")
        src_lines = read_file_lines(filename=src_input_file, autoclean=True)
        trg_lines = read_file_lines(filename=ref_input_file, autoclean=True)
        src_lines, trg_lines = filter_fn(src_lines, trg_lines, from_fn="translate")
        write_file_lines(filename=src_output_file, lines=src_lines,
                         autoclean=True, insert_break_line=True)
        write_file_lines(filename=ref_output_file, lines=trg_lines,
                         autoclean=True, insert_break_line=True)

    def _postprocess_eval_files(self, eval_ds, ctx: TranslateContext, preprocess_fn,
                                src_output_file, ref_output_file, hyp_output_file):
        # force_overwrite must be True here to rewrite the src/ref/hyp files in-place.
        for path, vocab_lang, lang in (
            (src_output_file, self.src_vocab.lang, eval_ds.src_lang),
            (ref_output_file, self.trg_vocab.lang, eval_ds.trg_lang),
            (hyp_output_file, self.trg_vocab.lang, eval_ds.trg_lang),
        ):
            preprocess_predict_file(
                input_file=path, output_file=path,
                preprocess_fn=preprocess_fn,
                pretokenize=ctx.pretok_flags[vocab_lang],
                input_lang=lang, vocab_lang=vocab_lang,
                ds=eval_ds, force_overwrite=True,
            )

    @staticmethod
    def _assert_ref_hyp_line_count(output_path):
        n_ref = count_file_lines(os.path.join(output_path, "ref.txt"))
        n_hyp = count_file_lines(os.path.join(output_path, "hyp.txt"))
        if n_ref != n_hyp:
            raise ValueError(
                f"The number of lines in 'ref.txt' ({n_ref}) and 'hyp.txt' "
                f"({n_hyp}) does not match. If you see a 'CUDA out of memory' "
                f"message, try again with smaller batch.")

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
                        trg_lang=self.trg_vocab.lang,
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

        return build_run_report(
            engine=self.engine, run_name=self.run_name, model=self.model,
            eval_ds=eval_ds, src_vocab=self.src_vocab, trg_vocab=self.trg_vocab,
            config=self.config, translations=translations,
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
        langs = {self.src_vocab.lang, self.trg_vocab.lang}
        if eval_mode is EvalMode.ALL:
            return ts_datasets
        if eval_mode is EvalMode.COMPATIBLE:
            return [ds for ds in ts_datasets if set(ds.langs).issubset(langs)]
        if eval_mode is EvalMode.SAME:
            trained_ds = {str(ds.id()) for ds in self.trained_ds}
            return [ds for ds in ts_datasets if str(ds.id()) in trained_ds]
        raise ValueError(f"Unknown 'eval_mode' ({str(eval_mode)})")

    # --- Path accessors (delegate to RunLayout) -------------------------

    def get_model_eval_path(self, eval_name, fname=""):
        return self._layout.eval_path(eval_name, fname)

    def get_model_eval_data_bin_path(self, eval_name, data_bin_name, fname=""):
        return self._layout.eval_data_bin_path(eval_name, data_bin_name, fname)

    def get_model_eval_translations_path(self, eval_name, split_name=""):
        return self._layout.translations_path(eval_name, split_name)

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


# UNSET re-exported here so legacy ``from autonmt.backends.base.translator import UNSET`` keeps working.
__all__ = ["BaseTranslator", "TranslateContext", "UNSET"]
