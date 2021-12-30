import os.path
import shutil

from typing import List, Set, Iterable

from autonmt.utils import *

from abc import ABC, abstractmethod
from autonmt.datasets import Dataset, DatasetBuilder
from autonmt.datasets.builder import encode_file, decode_file, get_compatible_datasets
from autonmt import py_cmd_api


def _check_datasets(train_ds: Dataset = None, eval_ds: Dataset = None):
    # Check that train_ds is a Dataset
    if train_ds and not isinstance(train_ds, Dataset):
        raise TypeError("'train_ds' must be an instance of 'Dataset' so that we can know the layout of the trained "
                        "model (e.g. checkpoints available, subword model, vocabularies, etc")

    # Check that train_ds is a Dataset
    if eval_ds and not isinstance(eval_ds, Dataset):
        raise TypeError("'eval_ds' must be an instance of 'Dataset' so that we can know the layout of the dataset "
                        "and get the corresponding data (e.g. splits, pretokenized, encoded, stc)")

    # Check that the datasets are compatible
    if train_ds and eval_ds and ((train_ds.src_lang != eval_ds.src_lang) or (train_ds.trg_lang != eval_ds.trg_lang)):
        raise ValueError(f"The languages from the train and test datasets are not compatible:\n"
                         f"\t- train_lang_pair=({train_ds.dataset_lang_pair})\n"
                         f"\t- test_lang_pair=({eval_ds.dataset_lang_pair})\n")


def _check_supported_metrics(metrics, metrics_supported):
    # Check
    metrics = set(metrics)
    metrics_supported = set(metrics_supported)

    # Get valid metrics
    metrics_valid = list(metrics.intersection(metrics_supported))
    metrics_valid += [x for x in metrics if x.startswith("hg_")]  # Ignore huggingface metrics
    metrics_valid = set(metrics_valid)
    metrics_non_valid = metrics.difference(metrics_valid)

    if metrics_non_valid:
        print(f"=> [WARNING] These metrics are not supported: {str(metrics_non_valid)}")
        if metrics == metrics_non_valid:
            print("\t- [Score]: Skipped. No valid metrics were found.")

    return metrics_valid


class BaseTranslator(ABC):

    # Global variables
    total_runs = 0
    TOOL_PARSERS = {"sacrebleu": {"filename": "sacrebleu_scores", "py": (parse_sacrebleu_json, "json"), "cmd": (parse_sacrebleu_json, "json")},
                    "bertscore": {"filename": "bertscore_scores", "py": (parse_bertscore_json, "json"), "cmd": (parse_bertscore_txt, "txt")},
                    "comet": {"filename": "comet_scores", "py": (parse_comet_json, "json"), "cmd": (parse_comet_txt, "txt")},
                    "beer": {"filename": "beer_scores", "py": (parse_beer_json, "json"), "cmd": (parse_beer_txt, "txt")},
                    "huggingface": {"filename": "huggingface_scores", "py": (parse_huggingface_json, "json"), "cmd": (parse_huggingface_json, "json")},
                    }
    TOOL2METRICS = {"sacrebleu": {"bleu", "chrf", "ter"},
                    "bertscore": {"bertscore"},
                    "comet": {"comet"},
                    "beer": {"beer"},
                    # "huggingface": "huggingface",
                    }
    METRICS2TOOL = {m: tool for tool, metrics in TOOL2METRICS.items() for m in metrics}

    def __init__(self, engine, run_prefix="model", model_ds=None, safe_seconds=0, force_overwrite=False,
                 interactive=False, use_cmd=False, conda_env_name=None, **kwargs):
        # Store vars
        self.engine = engine
        self.run_prefix = run_prefix
        self.model_ds = model_ds
        self.safe_seconds = safe_seconds
        self.force_overwrite = force_overwrite
        self.interactive = interactive
        self.use_cmd = use_cmd
        self.conda_env_name = conda_env_name
        self.config = {}

        # Check dataset
        _check_datasets(train_ds=self.model_ds) if self.model_ds else None

    def _make_empty_path(self, path, safe_seconds=0):
        # Check if the directory and can be delete it
        is_empty = os.listdir(path) == []
        if self.force_overwrite and os.path.exists(path) and not is_empty:
            print(f"=> [Existing data]: The contents of following directory are going to be deleted: {path}")
            res = ask_yes_or_no(question="Do you want to continue?", interactive=self.interactive)
            if res:
                if safe_seconds:
                    print(f"\t- Deleting files... (waiting {safe_seconds} seconds)")
                    time.sleep(safe_seconds)
                # Delete path
                shutil.rmtree(path)

        # Create path if it doesn't exist
        make_dir(path)
        is_empty = os.listdir(path) == []
        return is_empty

    def _get_run_name(self, ds):
        return f"{self.run_prefix}_{ds.subword_model}_{ds.vocab_size}".lower()

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

    def fit(self, train_ds: Dataset = None, max_epochs=1, learning_rate=0.001, criterion="cross_entropy",
            optimizer="adam", weight_decay=0, clip_norm=0.0, update_freq=1, max_tokens=None, batch_size=64, patience=10,
            seed=None, num_gpus=None, **kwargs):
        print("=> [Fit]: Started.")

        # Get train_ds
        if not train_ds and not self.model_ds:
            raise ValueError("'train_ds' is missing. You can either specify it in the constructor ('model_ds') or "
                             "pass it as an argument to this function")
        elif not train_ds and self.model_ds:
            print("\t- [INFO]: Using the 'model_ds' from the constructor as 'train_ds")
            train_ds = self.model_ds
        else:
            print("\t- [INFO]: Setting the 'train_ds' as the 'model_ds")
            self.model_ds = train_ds

        # Store config (and save file)
        self._add_config(key="fit", values=locals(), reset=False)
        self._add_config(key="fit", values=kwargs, reset=False)
        logs_path = train_ds.get_model_logs_path(toolkit=self.engine, run_name=self._get_run_name(ds=train_ds))
        make_dir(logs_path)
        save_json(self.config, savepath=os.path.join(logs_path, "config.json"))

        # Train and preprocess
        self.preprocess(train_ds, **kwargs)
        self.train(train_ds, max_epochs=max_epochs, learning_rate=learning_rate, criterion=criterion,
                   optimizer=optimizer, weight_decay=weight_decay, clip_norm=clip_norm, update_freq=update_freq,
                   max_tokens=max_tokens, batch_size=batch_size, patience=patience,
                   seed=seed, num_gpus=num_gpus, **kwargs)

    def predict(self, eval_datasets: List[Dataset], model_ds: Dataset = None, beams: List[int] = None,
                metrics: Set[str] = None, batch_size=128, max_tokens=None, max_gen_length=150, **kwargs):
        print("=> [Predict]: Started.")

        # Get model_ds
        if not model_ds and not self.model_ds:
            raise ValueError("'model_ds' is missing. You can either specify it in the constructor ('model_ds') or "
                             "pass it as an argument to this function")
        elif not model_ds and self.model_ds:
            print("\t- [INFO]: Using the 'model_ds' specified in the constructor")
            model_ds = self.model_ds
        else:
            print("\t- [INFO]: Setting the 'model_ds' as the 'model_ds")
            self.model_ds = model_ds

        # Set default values
        if beams is None:
            beams = [5]
        else:
            beams = list(set(beams))
            beams.sort(reverse=True)

        # Default metrics
        if metrics is None:
            metrics = {"bleu"}
        else:
            metrics = set(metrics)

        # Store config
        self._add_config(key="predict", values=locals(), reset=False)
        self._add_config(key="predict", values=kwargs, reset=False)
        logs_path = model_ds.get_model_logs_path(toolkit=self.engine, run_name=self._get_run_name(ds=model_ds))
        make_dir(logs_path)
        save_json(self.config, savepath=os.path.join(logs_path, "config.json"))

        # Translate and score
        eval_scores = []
        eval_datasets = get_compatible_datasets(eval_datasets, model_ds)
        for eval_ds in eval_datasets:
            self.translate(model_ds=model_ds, eval_ds=eval_ds, beams=beams, max_gen_length=max_gen_length,
                           batch_size=batch_size, max_tokens=max_tokens, **kwargs)
            self.score(model_ds=model_ds, eval_ds=eval_ds, beams=beams, metrics=metrics, **kwargs)
            model_scores = self.parse_metrics(model_ds=model_ds, eval_ds=eval_ds, beams=beams, metrics=metrics,
                                              engine=self.engine, **kwargs)
            eval_scores.append(model_scores)
        return eval_scores

    @abstractmethod
    def _preprocess(self, *args, **kwargs):
        pass

    def preprocess(self, ds: Dataset, **kwargs):
        print("=> [Preprocess]: Started.")

        # Set vars
        src_lang = ds.src_lang
        trg_lang = ds.trg_lang
        train_path = ds.get_encoded_path(fname=ds.train_name)
        val_path = ds.get_encoded_path(fname=ds.val_name)
        test_path = ds.get_encoded_path(fname=ds.test_name)
        model_src_vocab_path = ds.get_vocab_file(lang=src_lang)  # Ignore if: none or bytes
        model_trg_vocab_path = ds.get_vocab_file(lang=trg_lang)  # Ignore if: none or bytes
        model_data_bin_path = ds.get_model_data_bin(toolkit=self.engine)

        # Create dirs
        make_dir([model_data_bin_path])

        # Checks: Make sure the directory exist, and it is empty
        is_empty = self._make_empty_path(path=model_data_bin_path, safe_seconds=self.safe_seconds)
        if not is_empty:
            print("\t- [Preprocess]: Skipped. The output directory is not empty")
            return

        start_time = time.time()
        self._preprocess(src_lang=src_lang, trg_lang=trg_lang, output_path=model_data_bin_path,
                         train_path=train_path, val_path=val_path, test_path=test_path,
                         src_vocab_path=model_src_vocab_path, trg_vocab_path=model_trg_vocab_path,
                         subword_model=ds.subword_model, **kwargs)
        print(f"\t- [INFO]: Preprocess time: {str(datetime.timedelta(seconds=time.time()-start_time))}")

    @abstractmethod
    def _train(self, *args, **kwargs):
        pass

    def train(self, train_ds: Dataset, **kwargs):
        print("=> [Train]: Started.")

        # Check datasets
        _check_datasets(train_ds=train_ds)

        # Set run name
        run_name = self._get_run_name(ds=train_ds)

        # Set paths
        data_bin_path = train_ds.get_model_data_bin(toolkit=self.engine)
        checkpoints_path = train_ds.get_model_checkpoints_path(toolkit=self.engine, run_name=run_name)
        logs_path = train_ds.get_model_logs_path(toolkit=self.engine, run_name=run_name)

        # Create dirs
        make_dir([data_bin_path, checkpoints_path, logs_path])

        # Checks: Make sure the directory exist, and it is empty
        # is_empty = os.listdir(checkpoints_path) == []  # Too dangerous to allow overwrite
        is_empty = self._make_empty_path(path=checkpoints_path, safe_seconds=self.safe_seconds)
        if not is_empty:
            print("\t- [Train]: Skipped. The checkpoints directory is not empty")
            return

        # Set seed
        self.manual_seed(seed=kwargs.get("seed"))

        start_time = time.time()
        self._train(data_bin_path=data_bin_path, checkpoints_path=checkpoints_path, logs_path=logs_path, **kwargs)
        print(f"\t- [INFO]: Training time: {str(datetime.timedelta(seconds=time.time()-start_time))}")

    @abstractmethod
    def _translate(self, *args, **kwargs):
        pass

    def translate(self, model_ds: Dataset, eval_ds: Dataset, beams: List[int], max_gen_length,
                  batch_size, max_tokens, **kwargs):
        print("=> [Translate]: Started.")

        # Check datasets
        _check_datasets(train_ds=model_ds, eval_ds=eval_ds)
        assert model_ds.dataset_lang_pair == eval_ds.dataset_lang_pair

        # Set run names
        run_name = self._get_run_name(ds=model_ds)
        eval_name = '_'.join(eval_ds.id())  # Subword model and vocab size don't characterize the dataset!

        # Checkpoint path
        checkpoint_path = model_ds.get_model_checkpoints_path(self.engine, run_name, "checkpoint_best.pt")

        # [Trained model]: Create eval folder
        model_src_vocab_path = model_ds.get_vocab_file(lang=model_ds.src_lang)  # Needed to preprocess
        model_trg_vocab_path = model_ds.get_vocab_file(lang=model_ds.trg_lang)  # Needed to preprocess
        model_eval_data_encoded_path = model_ds.get_model_eval_data_encoded_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name)
        model_eval_data_bin_path = model_ds.get_model_eval_data_bin_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name)

        # Create dirs
        make_dir([model_eval_data_encoded_path, model_eval_data_bin_path])

        # Checks: Make sure the directory exist, and it is empty
        is_empty = self._make_empty_path(path=model_eval_data_bin_path, safe_seconds=self.safe_seconds)
        if not is_empty:
            print("\t- [Translate]: Skipped preprocessing. The output directory for the preprocessing data is not empty")
        else:
            # [Encode extern data]: Encode test data using the subword model of the trained model
            for ts_fname in [fname for fname in eval_ds.split_names_lang if eval_ds.test_name in fname]:
                lang = ts_fname.split('.')[-1]
                input_file = eval_ds.get_split_path(ts_fname)
                output_file = model_ds.get_model_eval_data_encoded_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name, fname=ts_fname)

                # Add pretokenization (if needed)
                output_file_pretok = None
                if model_ds.pretok_flag:
                    output_file_pretok = output_file + ".tok"

                # Encode file
                encode_file(ds=model_ds, input_file=input_file, output_file=output_file,
                            output_file_pretok=output_file_pretok,
                            lang=lang, merge_vocabs=model_ds.merge_vocabs, force_overwrite=self.force_overwrite,
                            use_cmd=self.use_cmd, conda_env_name=self.conda_env_name)

            # Preprocess external data
            test_path = os.path.join(model_eval_data_encoded_path, eval_ds.test_name)
            self._preprocess(src_lang=model_ds.src_lang, trg_lang=model_ds.trg_lang,
                             output_path=model_eval_data_bin_path,
                             train_path=None, val_path=None, test_path=test_path,
                             src_vocab_path=model_src_vocab_path, trg_vocab_path=model_trg_vocab_path,
                             subword_model=model_ds.subword_model,
                             external_data=True,
                             **kwargs)

        # Iterate over beams
        for beam in beams:
            start_time = time.time()
            # Create output path (if needed)
            output_path = model_ds.get_model_beam_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name, beam=beam)
            make_dir(output_path)

            # Translate
            tok_flag = [os.path.exists(os.path.join(output_path, f)) for f in ["src.tok", "ref.tok", "hyp.tok"]]
            if self.force_overwrite or not all(tok_flag):
                self._translate(
                    src_lang=model_ds.src_lang, trg_lang=model_ds.trg_lang,
                    beam_width=beam, max_gen_length=max_gen_length, batch_size=batch_size, max_tokens=max_tokens,
                    data_bin_path=model_eval_data_bin_path, output_path=output_path, checkpoint_path=checkpoint_path,
                    model_src_vocab_path=model_src_vocab_path, model_trg_vocab_path=model_trg_vocab_path, **kwargs)

            # Postprocess tokenized files
            for fname, lang in [("src", model_ds.src_lang), ("ref", model_ds.trg_lang), ("hyp", model_ds.trg_lang)]:
                input_file = os.path.join(output_path, f"{fname}.tok")
                output_file = os.path.join(output_path, f"{fname}.txt")
                model_vocab_path = model_src_vocab_path if lang == model_ds.src_lang else model_trg_vocab_path

                # Post-process files
                decode_file(input_file=input_file, output_file=output_file, lang=lang,
                            subword_model=model_ds.subword_model,
                            model_vocab_path=model_vocab_path, force_overwrite=self.force_overwrite,
                            use_cmd=self.use_cmd, conda_env_name=self.conda_env_name)

            print(f"\t- [INFO]: Translate time (beam={str(beam)}): {str(datetime.timedelta(seconds=time.time() - start_time))}")

    def score(self, model_ds: Dataset, eval_ds: Dataset, beams: List[int], metrics: Set[str], **kwargs):
        print("=> [Score]: Started.")

        # Check datasets
        _check_datasets(train_ds=model_ds, eval_ds=eval_ds)
        assert model_ds.dataset_lang_pair == eval_ds.dataset_lang_pair

        # Check supported metrics
        metrics_valid = _check_supported_metrics(metrics, self.METRICS2TOOL.keys())
        if not metrics_valid:
            return

        # Set run names
        run_name = self._get_run_name(ds=model_ds)
        eval_name = '_'.join(eval_ds.id())  # Subword model and vocab size don't characterize the dataset!

        # Iterate over beams
        for beam in beams:
            start_time = time.time()

            # Paths
            beam_path = model_ds.get_model_beam_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name, beam=beam)
            scores_path = model_ds.get_model_scores_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name, beam=beam)

            # Create dirs
            make_dir([scores_path])

            # Set input files (results)
            src_file_path = os.path.join(beam_path, "src.txt")
            ref_file_path = os.path.join(beam_path, "ref.txt")
            hyp_file_path = os.path.join(beam_path, "hyp.txt")

            # Check that the paths exists
            if not all([os.path.exists(p) for p in [src_file_path, ref_file_path, hyp_file_path]]):
                raise IOError("Missing files to compute scores")

            # Huggingface metrics
            hg_metrics = {x[3:] for x in metrics if x.startswith("hg_")}
            if hg_metrics:
                ext = "json" if self.use_cmd else "json"
                output_file = os.path.join(scores_path, f"huggingface_scores.{ext}")
                if self.force_overwrite or not os.path.exists(output_file):
                    py_cmd_api.compute_huggingface(src_file=src_file_path, hyp_file=hyp_file_path, ref_file=ref_file_path,
                                                   output_file=output_file, metrics=hg_metrics, trg_lang=model_ds.trg_lang,
                                                   use_cmd=self.use_cmd, conda_env_name=self.conda_env_name)

            # [CMD] Score: bleu, chrf and ter
            if self.TOOL2METRICS["sacrebleu"].intersection(metrics):
                ext = "json" if self.use_cmd else "json"
                output_file = os.path.join(scores_path, f"sacrebleu_scores.{ext}")
                if self.force_overwrite or not os.path.exists(output_file):
                    py_cmd_api.compute_sacrebleu(ref_file=ref_file_path, hyp_file=hyp_file_path, output_file=output_file, metrics=metrics, use_cmd=self.use_cmd, conda_env_name=self.conda_env_name)

            # [CMD] Score: bertscore
            if self.TOOL2METRICS["bertscore"].intersection(metrics):
                ext = "txt" if self.use_cmd else "json"
                output_file = os.path.join(scores_path, f"bertscore_scores.{ext}")
                if self.force_overwrite or not os.path.exists(output_file):
                    py_cmd_api.compute_bertscore(ref_file=ref_file_path, hyp_file=hyp_file_path, output_file=output_file, trg_lang=model_ds.trg_lang, use_cmd=self.use_cmd, conda_env_name=self.conda_env_name)

            # [CMD] Score: comet
            if self.TOOL2METRICS["comet"].intersection(metrics):
                ext = "txt" if self.use_cmd else "json"
                output_file = os.path.join(scores_path, f"comet_scores.{ext}")
                if self.force_overwrite or not os.path.exists(output_file):
                    py_cmd_api.compute_comet(src_file=src_file_path, ref_file=ref_file_path, hyp_file=hyp_file_path, output_file=output_file, use_cmd=self.use_cmd, conda_env_name=self.conda_env_name)

            # [CMD] Score: beer
            if self.TOOL2METRICS["beer"].intersection(metrics):
                ext = "txt" if self.use_cmd else "json"
                output_file = os.path.join(scores_path, f"beer_scores.{ext}")
                if self.force_overwrite or not os.path.exists(output_file):
                    py_cmd_api.compute_beer(ref_file=ref_file_path, hyp_file=hyp_file_path, output_file=output_file, use_cmd=self.use_cmd, conda_env_name=self.conda_env_name)
            print(f"\t- [INFO]: Translate time (beam={str(beam)}): {str(datetime.timedelta(seconds=time.time() - start_time))}")

    def parse_metrics(self, model_ds, eval_ds, beams: List[int], metrics: Set[str], **kwargs):
        print("=> [Parsing]: Started.")

        # Check datasets
        _check_datasets(train_ds=model_ds, eval_ds=eval_ds)
        assert model_ds.dataset_lang_pair == eval_ds.dataset_lang_pair

        # Check supported metrics
        metrics_valid = _check_supported_metrics(metrics, self.METRICS2TOOL.keys())
        if not metrics_valid:
            return

        # Metrics to retrieve
        metric_tools = self._get_metrics_tool(metrics)

        # Set run names
        run_name = self._get_run_name(ds=model_ds)
        eval_name = '_'.join(eval_ds.id())  # Subword model and vocab size don't characterize the dataset!

        # Walk through beams
        scores = {
            "engine": kwargs.get("engine"),
            "lang_pair": model_ds.dataset_lang_pair,
            "train_dataset": model_ds.dataset_name,
            "eval_dataset": eval_ds.dataset_name,
            "subword_model": str(model_ds.subword_model).lower(),
            "vocab_size": str(model_ds.vocab_size).lower(),
            "run_name": run_name,
            "train_max_lines": model_ds.dataset_lines,
            "beams": {},
            "config": self.config,
        }

        # Iterate over beams
        for beam in beams:
            # Paths
            scores_path = model_ds.get_model_scores_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name,
                                                         beam=beam)

            # Walk through metric files
            beam_scores = {}
            for m_tool in metric_tools:
                values = self.TOOL_PARSERS[m_tool]
                m_parser, ext = values["cmd"] if self.use_cmd else values["py"]
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
                        print(f"\t- [PARSING ERROR]: ({m_fname}) {str(e)}")
                else:
                    print(f"\t- [WARNING]: There are no metrics from '{m_tool}'")

            # Add beam scores
            scores["beams"].update({f"beam{str(beam)}": beam_scores})
        return scores

    @staticmethod
    def manual_seed(seed, use_deterministic_algorithms=False):
        import torch
        import random
        import numpy as np

        # Define seed
        seed = seed if seed is not None else int(time.time()) % 2**32

        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Tricky: https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
        torch.use_deterministic_algorithms(use_deterministic_algorithms)

        # Test randomness
        print(f"\t- [INFO]: Testing random seed ({seed}):")
        print(f"\t\t- random: {random.random()}")
        print(f"\t\t- numpy: {np.random.rand(1)}")
        print(f"\t\t- torch: {torch.rand(1)}")

        return seed
