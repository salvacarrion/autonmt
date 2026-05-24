import datetime
import os.path
import shutil
import time
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Set

from autonmt.bundle.metrics import (
    compute_sacrebleu,
    compute_bertscore,
    compute_comet,
    compute_fairseq,
    compute_huggingface,
)
from autonmt.bundle.utils import (
    count_file_lines,
    is_debug_enabled,
    make_dir,
    parse_beer_json,
    parse_bertscore_json,
    parse_comet_json,
    parse_fairseq_txt,
    parse_huggingface_json,
    parse_sacrebleu_json,
    read_file_lines,
    save_json,
    write_file_lines,
)
from autonmt.bundle.enums import EvalMode
from autonmt.bundle.logger import get_logger
from autonmt.preprocessing.dataset import Dataset
from autonmt.preprocessing.processors import preprocess_predict_file, encode_file, decode_file
from autonmt.toolkits.config import FitConfig, PredictConfig, UNSET, merge_config

log = get_logger(__name__)


def _check_datasets(train_ds: Optional[Dataset] = None, eval_ds: Optional[Dataset] = None) -> None:
    # Check that train_ds is a Dataset
    if train_ds and not isinstance(train_ds, Dataset):
        raise TypeError("'train_ds' must be an instance of 'Dataset' so that we can know the layout of the trained "
                        "model (e.g. checkpoints available, subword model, vocabularies, etc")

    # Check that train_ds is a Dataset
    if eval_ds and not isinstance(eval_ds, Dataset):
        raise TypeError("'eval_ds' must be an instance of 'Dataset' so that we can know the layout of the dataset "
                        "and get the corresponding data (e.g. splits, pretokenized, encoded, stc)")

    # Check that the preprocessing are compatible
    if train_ds and eval_ds and ((train_ds.src_lang != eval_ds.src_lang) or (train_ds.trg_lang != eval_ds.trg_lang)):
        raise ValueError(f"The languages from the train and test datasets are not compatible:\n"
                         f"\t- train_lang_pair=({train_ds.dataset_lang_pair})\n"
                         f"\t- test_lang_pair=({eval_ds.dataset_lang_pair})\n")


def _check_supported_metrics(metrics: Iterable[str], metrics_supported: Iterable[str]) -> Set[str]:
    # Check
    metrics = set(metrics)
    metrics_supported = set(metrics_supported)

    # Get valid metrics
    metrics_valid = list(metrics.intersection(metrics_supported))
    metrics_valid += [x for x in metrics if x.startswith("hg_")]  # Ignore huggingface metrics
    metrics_valid = set(metrics_valid)
    metrics_non_valid = metrics.difference(metrics_valid)

    if metrics_non_valid:
        log.warning(f"=> [WARNING] These metrics are not supported: {str(metrics_non_valid)}")
        if metrics == metrics_non_valid:
            log.info("\t- [Score]: Skipped. No valid metrics were found.")

    return metrics_valid


class BaseTranslator(ABC):

    # Global variables
    total_runs = 0
    TOOL_PARSERS = {"sacrebleu": {"filename": "sacrebleu_scores", "py": (parse_sacrebleu_json, "json")},
                    "bertscore": {"filename": "bertscore_scores", "py": (parse_bertscore_json, "json")},
                    "comet": {"filename": "comet_scores", "py": (parse_comet_json, "json")},
                    "beer": {"filename": "beer_scores", "py": (parse_beer_json, "json")},
                    "huggingface": {"filename": "huggingface_scores", "py": (parse_huggingface_json, "json")},
                    "fairseq": {"filename": "fairseq_scores", "py": (parse_fairseq_txt, "txt")},
                    }
    TOOL2METRICS = {"sacrebleu": {"bleu", "chrf", "ter"},
                    "bertscore": {"bertscore"},
                    "comet": {"comet"},
                    "beer": {"beer"},
                    "fairseq": {"fairseq"},
                    # "huggingface": "huggingface",
                    }
    METRICS2TOOL = {m: tool for tool, metrics in TOOL2METRICS.items() for m in metrics}

    def __init__(self, engine, runs_dir="runs", run_name=None, src_vocab=None, trg_vocab=None,
                 filter_tr_data_fn=None, filter_vl_data_fn=None, filter_ts_data_fn=None,
                 safe_seconds=3, **kwargs):
        # Store vars
        self.config = {}
        self.engine = engine
        self.runs_dir = runs_dir if runs_dir else "runs/"
        self.run_name = run_name if run_name else f"{str(datetime.datetime.now().strftime('%Y.%m.%d_%H.%M.%S'))}"
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.from_checkpoint = None
        self.safe_seconds = safe_seconds
        self.trained_ds = []  # Trick to perform evaluate "same"

        # Further split/preprocess each dataset if needed
        self.filter_tr_data_fn = ('', None) if not filter_tr_data_fn else filter_tr_data_fn
        self.filter_vl_data_fn = [('', None)] if not filter_vl_data_fn else filter_vl_data_fn
        self.filter_ts_data_fn = [('', None)] if not filter_ts_data_fn else filter_ts_data_fn

        # Models paths: toolkit/runs/model_name/[checkpoints, logs, eval]
        self.models_checkpoints_path = "checkpoints"
        self.model_logs_path = "logs"
        self.models_eval_path = "eval"
        self.models_eval_translations_name = "translations"
        self.models_eval_beam_path = "beam"
        self.models_eval_beam_scores_path = "scores"

    def _get_metrics_tool(self, metrics):
        tools = set()
        for m in metrics:
            if m.startswith("hg_"):
                m_tool = "huggingface"
            else:
                m_tool = self.METRICS2TOOL.get(m)

            # Add tools
            if m_tool:
                tools.add(m_tool)
        return tools

    def _add_config(self, key: str, values: dict, reset=False):
        def is_valid(k, v):
            primitive_types = (str, bool, int, float, dict, set, list)  # Problems with list of objects
            return not(k.startswith("_") or k in {"kwargs"}) and (isinstance(v, primitive_types) or v is None)

        def parse_value(x):
            if isinstance(x, (list, set)):
                return [str(_x) for _x in x]
            return str(x)

        # Reset value (if needed)
        if reset or key not in self.config:
            self.config[key] = {}

        # Update values
        self.config[key].update({k: parse_value(v) for k, v in values.items() if is_valid(k, v)})

    def _save_config(self, fname="config.json", force_overwrite=False):
        logs_path = self.get_model_logs_path()
        make_dir(logs_path)
        save_json(self.config, savepath=os.path.join(logs_path, fname), allow_overwrite=force_overwrite)

    def fit(self, train_ds: Dataset,
            config: Optional[FitConfig] = None,
            max_tokens=UNSET, batch_size=UNSET,
            max_epochs=UNSET, patience=UNSET,
            optimizer=UNSET, learning_rate=UNSET,
            weight_decay=UNSET, gradient_clip_val=UNSET,
            accumulate_grad_batches=UNSET,
            criterion=UNSET, monitor=UNSET,
            devices=UNSET, accelerator=UNSET, num_workers=UNSET,
            seed=UNSET, force_overwrite=UNSET,
            use_bucketing=UNSET, **kwargs) -> None:
        log.info("=> [Fit]: Started.")

        # Materialise effective config: defaults from FitConfig, overridden by ``config``
        # if supplied, then by any kwargs the caller actually passed (sentinel-aware).
        explicit = dict(
            max_tokens=max_tokens, batch_size=batch_size, max_epochs=max_epochs,
            patience=patience, optimizer=optimizer, learning_rate=learning_rate,
            weight_decay=weight_decay, gradient_clip_val=gradient_clip_val,
            accumulate_grad_batches=accumulate_grad_batches, criterion=criterion,
            monitor=monitor, devices=devices, accelerator=accelerator,
            num_workers=num_workers, seed=seed, force_overwrite=force_overwrite,
            use_bucketing=use_bucketing,
        )
        cfg, extra = merge_config(config, FitConfig, {**explicit, **kwargs})

        # Save training config
        self._add_config(key="fit", values=cfg, reset=False)
        self._add_config(key="fit", values=extra, reset=False)
        self._save_config(fname="config_train.json", force_overwrite=cfg["force_overwrite"])

        # Train and preprocess
        self.preprocess(train_ds, apply2train=True, apply2val=True, apply2test=False,
                        force_overwrite=cfg["force_overwrite"], **extra)
        self.train(train_ds, **cfg, **extra)

    def predict(self, eval_datasets: Iterable[Dataset],
                config: Optional[PredictConfig] = None,
                metrics=UNSET, beams=UNSET,
                max_len_a=UNSET, max_len_b=UNSET,
                max_tokens=UNSET, batch_size=UNSET,
                devices=UNSET, accelerator=UNSET, num_workers=UNSET,
                load_checkpoint=UNSET, preprocess_fn=UNSET,
                eval_mode=UNSET, force_overwrite=UNSET, **kwargs) -> List[dict]:
        log.info("=> [Predict]: Started.")

        explicit = dict(
            metrics=metrics, beams=beams, max_len_a=max_len_a, max_len_b=max_len_b,
            max_tokens=max_tokens, batch_size=batch_size, devices=devices,
            accelerator=accelerator, num_workers=num_workers,
            load_checkpoint=load_checkpoint, preprocess_fn=preprocess_fn,
            eval_mode=eval_mode, force_overwrite=force_overwrite,
        )
        cfg, extra = merge_config(config, PredictConfig, {**explicit, **kwargs})

        # Normalize defaults that need post-processing
        cfg["beams"] = [1] if cfg["beams"] is None else list(sorted(set(cfg["beams"]), reverse=True))
        cfg["metrics"] = {"bleu"} if cfg["metrics"] is None else set(cfg["metrics"])

        # Store config
        self._add_config(key="predict", values=cfg, reset=False)
        self._add_config(key="predict", values=extra, reset=False)
        self._save_config(fname="config_predict.json", force_overwrite=cfg["force_overwrite"])

        # Translate and score
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
                           force_overwrite=cfg["force_overwrite"], **extra)
            self.score_translations(eval_ds, beams=cfg["beams"], metrics=cfg["metrics"],
                                    force_overwrite=cfg["force_overwrite"], **extra)
            run_scores = self.parse_metrics(eval_ds, beams=cfg["beams"], metrics=cfg["metrics"],
                                            engine=self.engine,
                                            force_overwrite=cfg["force_overwrite"], **extra)
            scores.append(run_scores)
        return scores

    @abstractmethod
    def _preprocess(self, *args, **kwargs):
        pass

    def preprocess(self, ds: Dataset, apply2train, apply2val, apply2test, force_overwrite, **kwargs):
        log.info(f"=> [Preprocess]: Started. ({ds.id2(as_path=True)})")

        # Set vocab paths
        model_src_vocab_path = ds.get_vocab_file(lang=ds.src_lang)
        model_trg_vocab_path = ds.get_vocab_file(lang=ds.trg_lang)

        # Get split paths
        train_path = ds.get_encoded_path(fname=ds.train_name)
        val_path = ds.get_encoded_path(fname=ds.val_name)
        test_path = ds.get_encoded_path(fname=ds.test_name)

        start_time = time.time()
        self._preprocess(ds=ds, output_path=None,
                         src_lang=ds.src_lang, trg_lang=ds.trg_lang,
                         src_vocab_path=model_src_vocab_path, trg_vocab_path=model_trg_vocab_path,
                         train_path=train_path, val_path=val_path, test_path=test_path,
                         apply2train=apply2train, apply2val=apply2val, apply2test=apply2test,
                         force_overwrite=force_overwrite, **kwargs)
        log.info(f"\t- [INFO]: Preprocess time: {str(datetime.timedelta(seconds=time.time()-start_time))}")

    @abstractmethod
    def _train(self, *args, **kwargs):
        pass

    def train(self, train_ds, force_overwrite, **kwargs):
        log.info(f"=> [Train]: Started. ({train_ds.id2(as_path=True)})")

        # Check preprocessing
        _check_datasets(train_ds=train_ds)

        # Check debug
        if is_debug_enabled():
            log.warning("\t=> [WARNING]: Debug is enabled. This could lead to critical problems when using a data parallel strategy.")

        # Set stuff
        self.trained_ds.append(train_ds)
        checkpoints_dir = self.get_model_checkpoints_path()
        logs_path = self.get_model_logs_path()
        make_dir([checkpoints_dir, logs_path])

        # Set seed
        self.manual_seed(seed=kwargs.get("seed"))

        # Train
        start_time = time.time()
        self._train(train_ds=train_ds, checkpoints_dir=checkpoints_dir, logs_path=logs_path,
                    force_overwrite=force_overwrite, **kwargs)
        log.info(f"\t- [INFO]: Training time: {str(datetime.timedelta(seconds=time.time()-start_time))}")


    @abstractmethod
    def _translate(self, *args, **kwargs):
        pass

    def translate(self, eval_ds, beams, preprocess_fn, force_overwrite, **kwargs):
        log.info(f"=> [Translate]: Started. (Model: {self.run_name} | Test: {str(eval_ds)})")
        _check_datasets(eval_ds=eval_ds)

        ctx = self._build_translate_context(eval_ds)
        self._encode_eval_test_data(eval_ds, ctx, preprocess_fn=preprocess_fn,
                                    force_overwrite=force_overwrite)
        self._preprocess_eval_test_data(eval_ds, ctx, force_overwrite=force_overwrite, **kwargs)

        for filter_idx, (fn_name, filter_fn) in enumerate(self.filter_ts_data_fn):
            for beam in beams:
                self._translate_one_beam(eval_ds, ctx, beam=beam, filter_idx=filter_idx,
                                         filter_fn=filter_fn, fn_name=fn_name,
                                         preprocess_fn=preprocess_fn,
                                         force_overwrite=force_overwrite, **kwargs)

    def _build_translate_context(self, eval_ds):
        """Snapshot of paths and per-language metadata reused by every beam pass."""
        checkpoints_dir = self.get_model_checkpoints_path()
        model_eval_path = self.get_model_eval_path(eval_name=str(eval_ds))
        make_dir([model_eval_path])

        dst_raw_path = os.path.join(model_eval_path, "data/0_raw")
        dst_preprocessed_path = os.path.join(model_eval_path, "data/1_preprocessed")
        dst_encoded_path = os.path.join(model_eval_path, "data/3_encoded")
        make_dir([dst_raw_path, dst_preprocessed_path, dst_encoded_path])

        src_lang, trg_lang = self.src_vocab.lang, self.trg_vocab.lang
        return {
            "checkpoints_dir": checkpoints_dir,
            "model_eval_path": model_eval_path,
            "model_src_vocab_path": self.src_vocab.vocab_path,
            "model_trg_vocab_path": self.trg_vocab.vocab_path,
            "dst_raw_path": dst_raw_path,
            "dst_preprocessed_path": dst_preprocessed_path,
            "dst_encoded_path": dst_encoded_path,
            "vocab_langs": [src_lang, trg_lang],
            "pretok_flags": {src_lang: self.src_vocab.pretok_flag,
                             trg_lang: self.trg_vocab.pretok_flag},
            "vocab_paths": {src_lang: self.src_vocab.model_path,
                            trg_lang: self.trg_vocab.model_path},
            "subword_models": {src_lang: self.src_vocab.subword_model,
                               trg_lang: self.trg_vocab.subword_model},
        }

    def _encode_eval_test_data(self, eval_ds, ctx, preprocess_fn, force_overwrite):
        """Copy raw test files, run preprocess_fn (+ pretokenization), then subword-encode."""
        test_fnames = [f"{eval_ds.test_name}.{eval_ds.src_lang}",
                       f"{eval_ds.test_name}.{eval_ds.trg_lang}"]  # (0: src, 1: trg)
        for i, ts_fname in enumerate(test_fnames):
            input_file = eval_ds.get_split_path(ts_fname)
            input_lang = ts_fname.split(".")[-1]
            vocab_lang = ctx["vocab_langs"][i]

            # 1 - copy raw
            source_file = os.path.join(ctx["dst_raw_path"], ts_fname)
            if force_overwrite or not os.path.exists(source_file):
                shutil.copyfile(input_file, source_file)
                assert os.path.exists(source_file)
                input_file = source_file

            # 2 - preprocess (+ pretokenize)
            preprocessed_file = os.path.join(ctx["dst_preprocessed_path"], ts_fname)
            preprocess_predict_file(input_file=input_file, output_file=preprocessed_file,
                                    preprocess_fn=preprocess_fn,
                                    pretokenize=ctx["pretok_flags"][vocab_lang],
                                    input_lang=input_lang, vocab_lang=vocab_lang,
                                    ds=eval_ds, force_overwrite=force_overwrite)

            # 3 - subword encode
            enc_file = os.path.join(ctx["dst_encoded_path"], ts_fname)
            encode_file(input_file=preprocessed_file, output_file=enc_file,
                        model_vocab_path=ctx["vocab_paths"][vocab_lang],
                        subword_model=ctx["subword_models"][vocab_lang],
                        force_overwrite=force_overwrite)

    def _preprocess_eval_test_data(self, eval_ds, ctx, force_overwrite, **kwargs):
        """Toolkit-specific preprocessing of the encoded test data (e.g. fairseq binarization)."""
        test_path = os.path.join(ctx["dst_encoded_path"], eval_ds.test_name)
        self._preprocess(train_path=None, val_path=None, test_path=test_path,
                         src_lang=eval_ds.src_lang, trg_lang=eval_ds.trg_lang,
                         src_vocab_path=ctx["model_src_vocab_path"],
                         trg_vocab_path=ctx["model_trg_vocab_path"],
                         apply2train=False, apply2val=False, apply2test=True,
                         output_path=ctx["model_eval_path"],
                         force_overwrite=force_overwrite, **kwargs)

    def _translate_one_beam(self, eval_ds, ctx, beam, filter_idx, filter_fn, fn_name,
                            preprocess_fn, force_overwrite, **kwargs):
        extra_str = f" | split='{fn_name}'" if fn_name else ""
        output_path = self.get_model_eval_translations_beam_path(
            eval_name=str(eval_ds), split_name=fn_name, beam=beam)
        make_dir(output_path)

        if not (force_overwrite or not os.path.exists(os.path.join(output_path, "hyp.tok"))):
            return

        start_time = time.time()
        self._translate(
            data_path=ctx["model_eval_path"], output_path=output_path,
            src_lang=eval_ds.src_lang, trg_lang=eval_ds.trg_lang,
            beam_width=beam, checkpoints_dir=ctx["checkpoints_dir"],
            model_src_vocab_path=ctx["model_src_vocab_path"],
            model_trg_vocab_path=ctx["model_trg_vocab_path"],
            force_overwrite=force_overwrite, filter_idx=filter_idx, **kwargs)

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

        log.info(f"\t- [INFO]: Translating time (beam={beam}{extra_str}): "
                 f"{datetime.timedelta(seconds=time.time() - start_time)}")

    def _decode_hypothesis(self, ctx, output_path, hyp_output_file, force_overwrite):
        model_lang = self.trg_vocab.lang
        hyp_input_file = os.path.join(output_path, "hyp.tok")
        decode_file(input_file=hyp_input_file, output_file=hyp_output_file, lang=model_lang,
                    subword_model=ctx["subword_models"][model_lang],
                    pretok_flag=ctx["pretok_flags"][model_lang],
                    model_vocab_path=ctx["vocab_paths"][model_lang],
                    remove_unk_hyphen=True, force_overwrite=force_overwrite)

    def _materialize_src_ref(self, eval_ds, ctx, src_output_file, ref_output_file,
                             filter_fn, fn_name):
        src_input_file = os.path.join(ctx["dst_raw_path"],
                                      f"{eval_ds.test_name}.{eval_ds.src_lang}")
        ref_input_file = os.path.join(ctx["dst_raw_path"],
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

    def _postprocess_eval_files(self, eval_ds, ctx, preprocess_fn,
                                src_output_file, ref_output_file, hyp_output_file):
        # force_overwrite must be True here to rewrite the src/ref/hyp files in-place.
        for path, vocab_lang, lang in (
            (src_output_file, self.src_vocab.lang, eval_ds.src_lang),
            (ref_output_file, self.trg_vocab.lang, eval_ds.trg_lang),
            (hyp_output_file, self.trg_vocab.lang, eval_ds.trg_lang),
        ):
            preprocess_predict_file(input_file=path, output_file=path,
                                    preprocess_fn=preprocess_fn,
                                    pretokenize=ctx["pretok_flags"][vocab_lang],
                                    input_lang=lang, vocab_lang=vocab_lang,
                                    ds=eval_ds, force_overwrite=True)

    @staticmethod
    def _assert_ref_hyp_line_count(output_path):
        num_lines_ref = count_file_lines(os.path.join(output_path, "ref.txt"))
        num_lines_hyp = count_file_lines(os.path.join(output_path, "hyp.txt"))
        if num_lines_ref != num_lines_hyp:
            raise ValueError(
                f"The number of lines in 'ref.txt' ({num_lines_ref}) and 'hyp.txt' "
                f"({num_lines_hyp}) does not match. If you see a 'CUDA out of memory' "
                f"message, try again with smaller batch.")


    def score_translations(self, eval_ds: Dataset, beams: List[int], metrics: Set[str], force_overwrite, **kwargs):
        log.info(f"=> [Scoring translations]: Started. (Model: {self.run_name} | Test: {str(eval_ds)})")

        # Check preprocessing
        _check_datasets(eval_ds=eval_ds)

        # Check supported metrics
        metrics_valid = _check_supported_metrics(metrics, self.METRICS2TOOL.keys())
        if not metrics_valid:
            return

        # Allow to split ts data (optional)
        for fn_name, _ in self.filter_ts_data_fn:
            extra_str = f" | split='{fn_name}'" if fn_name else ""

            # Iterate over beams
            for beam in beams:
                start_time = time.time()

                # Paths
                beam_path = self.get_model_eval_translations_beam_path(eval_name=str(eval_ds), split_name=fn_name, beam=beam)
                scores_path = self.get_model_eval_translations_beam_scores_path(eval_name=str(eval_ds), split_name=fn_name, beam=beam)
                make_dir([scores_path])

                # Set input files (results)
                src_file_path = os.path.join(beam_path, "src.txt")
                ref_file_path = os.path.join(beam_path, "ref.txt")
                hyp_file_path = os.path.join(beam_path, "hyp.txt")

                # Check that the paths exists
                if not all([os.path.exists(p) for p in [src_file_path, ref_file_path, hyp_file_path]]):
                    raise IOError("Missing files to compute scores")

                # Score: bleu, chrf and ter
                if self.TOOL2METRICS["sacrebleu"].intersection(metrics):
                    output_file = os.path.join(scores_path, f"sacrebleu_scores.json")
                    if force_overwrite or not os.path.exists(output_file):
                        compute_sacrebleu(ref_file=ref_file_path, hyp_file=hyp_file_path, output_file=output_file, metrics=metrics)

                # Score: bertscore
                if self.TOOL2METRICS["bertscore"].intersection(metrics):
                    output_file = os.path.join(scores_path, f"bertscore_scores.json")
                    if force_overwrite or not os.path.exists(output_file):
                        compute_bertscore(ref_file=ref_file_path, hyp_file=hyp_file_path, output_file=output_file, trg_lang=self.trg_vocab.lang)

                # Score: comet
                if self.TOOL2METRICS["comet"].intersection(metrics):
                    output_file = os.path.join(scores_path, f"comet_scores.json")
                    if force_overwrite or not os.path.exists(output_file):
                        compute_comet(src_file=src_file_path, ref_file=ref_file_path, hyp_file=hyp_file_path, output_file=output_file)

                 # Score: fairseq
                if self.TOOL2METRICS["fairseq"].intersection(metrics):
                    output_file = os.path.join(scores_path, f"fairseq_scores.txt")
                    if force_overwrite or not os.path.exists(output_file):
                        compute_fairseq(ref_file=ref_file_path, hyp_file=hyp_file_path, output_file=output_file)

                # Huggingface metrics
                hg_metrics = {x[3:] for x in metrics if x.startswith("hg_")}
                if hg_metrics:
                    output_file = os.path.join(scores_path, f"huggingface_scores.json")
                    if force_overwrite or not os.path.exists(output_file):
                        compute_huggingface(src_file=src_file_path, hyp_file=hyp_file_path, ref_file=ref_file_path,
                                            output_file=output_file, metrics=hg_metrics, trg_lang=self.trg_vocab.lang)

                log.info(f"\t- [INFO]: Scoring time (beam={str(beam)}{extra_str}): {str(datetime.timedelta(seconds=time.time() - start_time))}")


    def parse_metrics(self, eval_ds, beams, metrics, **kwargs):
        log.info(f"=> [Parsing]: Started. ({str(eval_ds)})")

        # Check preprocessing
        _check_datasets(eval_ds=eval_ds)

        # Check supported metrics
        metrics_valid = _check_supported_metrics(metrics, self.METRICS2TOOL.keys())
        if not metrics_valid:
            return

        # Metrics to retrieve
        metric_tools = self._get_metrics_tool(metrics)

        # Walk through beams
        assert self.src_vocab.subword_model == self.trg_vocab.subword_model
        if len(self.src_vocab) != len(self.trg_vocab):
            vocab_size = f"{len(self.src_vocab)}/{len(self.trg_vocab)}"
        else:
            vocab_size = f"{len(self.src_vocab)}"

        # Get model params
        total_params, trainable_params, no_trainable_params = self.model.count_parameters()

        # Report
        report_dict = {
            # Engine
            "engine": self.engine,
            "run_name": self.run_name,
            "eval_datetime": str(datetime.datetime.now()),

            # Model
            "model__architecture": self.model.architecture,
            "model__trainable_params": trainable_params,
            "model__no_trainable_params": no_trainable_params,
            "model__total_params": total_params,
            "model__dtype": str(self.model.dtype),

            # Vocab
            "vocab__subword_model": self.src_vocab.subword_model,
            "vocab__size": vocab_size,
            "vocab__merged": "no-specified",
            "vocab__lang_pair": f"{self.src_vocab.lang}-{self.trg_vocab.lang}",

            # Language pairs
            "train__lang_pair": f"{self.src_vocab.lang}-{self.trg_vocab.lang}",  # Careful with "filter_tr_data_fn"
            "test__lang_pair": f"{eval_ds.src_lang}-{eval_ds.trg_lang}",  # Careful with "filter_ts_data_fn"

            # Datasets
            "train_dataset": "no-specified",
            "test_dataset": eval_ds.dataset_name,
            "test_dataset_full": eval_ds.dataset_name + f"__{eval_ds.src_lang}-{eval_ds.trg_lang}",

            # Scores
            "translations": {},

            # Extra
            "config": self.config,
        }

        # Allow to split ts data (optional)
        for fn_name, _ in self.filter_ts_data_fn:
            extra_str = f" | split='{fn_name}'" if fn_name else ""

            # Iterate over beams
            for beam in beams:
                start_time = time.time()

                # Paths
                scores_path = self.get_model_eval_translations_beam_scores_path(eval_name=str(eval_ds), split_name=fn_name, beam=beam)

                # Walk through metric files
                beam_scores = {}
                for m_tool in metric_tools:
                    values = self.TOOL_PARSERS[m_tool]
                    m_parser, ext = values["py"]
                    m_fname = f"{values['filename']}.{ext}"

                    # Read file
                    filename = os.path.join(scores_path, m_fname)
                    if os.path.exists(filename):
                        try:
                            with open(filename, 'r') as f:
                                m_scores = m_parser(text=f.readlines())
                                for m_name, m_values in m_scores.items():  # [bleu_score, chrf_score, ter_score], [bertscore_precision]
                                    for score_name, score_value in m_values.items():
                                        m_name_full = f"{m_tool}_{m_name}_{score_name}".lower().strip()
                                        beam_scores[m_name_full] = score_value
                        except Exception as e:
                            log.warning(f"\t- [PARSING ERROR]: ({m_fname}) {str(e)}")
                    else:
                        log.warning(f"\t- [WARNING]: There are no metrics from '{m_tool}'")

                # Add beam scores
                # d = {f"beam{str(beam)}": beam_scores, "ts_filter_fn": fn_name}
                d = {f"beam{str(beam)}": beam_scores}
                d = {fn_name: d} if fn_name else d  # Pretty
                report_dict["translations"].update(d)
                log.info(f"\t- [INFO]: Parsed time (beam={str(beam)}{extra_str}): {str(datetime.timedelta(seconds=time.time() - start_time))}")
        return report_dict

    @staticmethod
    def manual_seed(seed, use_deterministic_algorithms=False):
        import torch
        import random
        import numpy as np
        from lightning_fabric.utilities.seed import seed_everything

        # Define seed
        seed = seed if seed is not None else int(time.time()) % 2**32

        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        seed_everything(seed)

        # Tricky: https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
        torch.use_deterministic_algorithms(use_deterministic_algorithms)

        # Test randomness
        log.info(f"\t- [INFO]: Testing random seed ({seed}):")
        log.info(f"\t\t- random: {random.random()}")
        log.info(f"\t\t- numpy: {np.random.rand(1)}")
        log.info(f"\t\t- torch: {torch.rand(1)}")

        return seed

    def filter_eval_datasets(self, ts_datasets, eval_mode):
        eval_mode = EvalMode.coerce(eval_mode)
        langs = {self.src_vocab.lang, self.trg_vocab.lang}
        if eval_mode is EvalMode.ALL:
            return ts_datasets
        elif eval_mode is EvalMode.COMPATIBLE:
            return [ds for ds in ts_datasets if set(ds.langs).issubset(set(langs))]
        elif eval_mode is EvalMode.SAME:
            trained_ds = {str(ds.id()) for ds in self.trained_ds}
            return [ds for ds in ts_datasets if str(ds.id()) in trained_ds]
        else:
            raise ValueError(f"Unknown 'eval_mode' ({str(eval_mode)})")
    def get_model_eval_path(self, eval_name, fname=""):
        return os.path.join(self.runs_dir, self.run_name, self.models_eval_path, eval_name, fname)

    def get_model_eval_data_bin_path(self, eval_name, data_bin_name, fname=""):
        eval_path = self.get_model_eval_path(eval_name)
        return os.path.join(eval_path, data_bin_name, fname)

    def get_model_eval_translations_path(self, eval_name, split_name=""):
        eval_path = self.get_model_eval_path(eval_name)
        return os.path.join(eval_path, self.models_eval_translations_name, split_name)

    def get_model_eval_translations_beam_path(self, eval_name, split_name, beam, fname=""):
        beam_n = f"beam{str(beam)}"
        eval_translations_path = self.get_model_eval_translations_path(eval_name, split_name)
        return os.path.join(eval_translations_path, self.models_eval_beam_path, beam_n, fname)

    def get_model_eval_translations_beam_scores_path(self, eval_name, split_name, beam, fname=""):
        eval_translations_beam_path = self.get_model_eval_translations_beam_path(eval_name, split_name, beam)
        return os.path.join(eval_translations_beam_path, self.models_eval_beam_scores_path, fname)

    def get_model_logs_path(self, fname=""):
        return os.path.join(self.runs_dir, self.run_name, self.model_logs_path, fname)

    def get_model_checkpoints_path(self, fname=""):
        return os.path.join(self.runs_dir, self.run_name, self.models_checkpoints_path, fname)