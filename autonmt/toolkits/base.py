import os.path
import shutil
from abc import ABC, abstractmethod
from typing import List, Set

from autonmt.api import py_cmd_api
from autonmt.bundle.utils import *
from autonmt.preprocessing.dataset import Dataset
from autonmt.preprocessing.processors import normalize_file, pretokenize_file, encode_file, decode_file


def _check_datasets(train_ds: Dataset = None, eval_ds: Dataset = None):
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
                    "fairseq": {"filename": "fairseq_scores", "py": (parse_fairseq_txt, "txt"), "cmd": (parse_fairseq_txt, "txt")},
                    }
    TOOL2METRICS = {"sacrebleu": {"bleu", "chrf", "ter"},
                    "bertscore": {"bertscore"},
                    "comet": {"comet"},
                    "beer": {"beer"},
                    "fairseq": {"fairseq"},
                    # "huggingface": "huggingface",
                    }
    METRICS2TOOL = {m: tool for tool, metrics in TOOL2METRICS.items() for m in metrics}

    def __init__(self, engine, run_prefix="model", model_ds=None, src_vocab=None, trg_vocab=None,
                 use_cmd=False, venv_path=None, **kwargs):
        # Store vars
        self.engine = engine
        self.run_prefix = run_prefix
        self.model_ds = model_ds
        self.use_cmd = use_cmd
        self.venv_path = venv_path
        self.config = {}
        self.model_ds = None

        # Set vocab (optional)
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        # Check dataset
        _check_datasets(train_ds=self.model_ds) if self.model_ds else None

    # def _make_empty_path(self, path, safe_seconds=0):
    #     # Check if the directory and can be delete it
    #     is_empty = os.listdir(path) == []
    #     if self.force_overwrite and os.path.exists(path) and not is_empty:
    #         print(f"=> [Existing data]: The contents of following directory are going to be deleted: {path}")
    #         res = ask_yes_or_no(question="Do you want to continue?", interactive=self.interactive)
    #         if res:
    #             if safe_seconds:
    #                 print(f"\t- Deleting files... (waiting {safe_seconds} seconds)")
    #                 time.sleep(safe_seconds)
    #             # Delete path
    #             shutil.rmtree(path)
    #
    #     # Create path if it doesn't exist
    #     make_dir(path)
    #     is_empty = os.listdir(path) == []
    #     return is_empty

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

    def fit(self, train_ds, max_tokens=None, batch_size=128, max_epochs=1,
            learning_rate=0.001, optimizer="adam", weight_decay=0, gradient_clip_val=0.0, accumulate_grad_batches=1,
            criterion="cross_entropy", patience=None, seed=None, devices="auto", accelerator="auto", num_workers=0,
            monitor="loss", resume_training=False, force_overwrite=False, **kwargs):
        print("=> [Fit]: Started.")

        # Set model
        self.model_ds = train_ds

        # Store config (and save file)
        self._add_config(key="fit", values=locals(), reset=False)
        self._add_config(key="fit", values=kwargs, reset=False)
        logs_path = train_ds.get_model_logs_path(toolkit=self.engine, run_name=train_ds.get_run_name(self.run_prefix))
        make_dir(logs_path)
        save_json(self.config, savepath=os.path.join(logs_path, "config_train.json"))

        # Train and preprocess
        self.preprocess(train_ds, force_overwrite=force_overwrite, **kwargs)
        self.train(train_ds, max_tokens=max_tokens, batch_size=batch_size, max_epochs=max_epochs,
                   learning_rate=learning_rate, optimizer=optimizer, weight_decay=weight_decay,
                   gradient_clip_val=gradient_clip_val, accumulate_grad_batches=accumulate_grad_batches,
                   criterion=criterion, patience=patience, seed=seed, devices=devices, accelerator=accelerator,
                   num_workers=num_workers, monitor=monitor, resume_training=resume_training,
                   force_overwrite=force_overwrite, **kwargs)

    def predict(self, eval_datasets: List[Dataset], beams: List[int] = None,
                metrics: Set[str] = None, batch_size=64, max_tokens=None, max_len_a=1.2, max_len_b=50, truncate_at=None,
                devices="auto", accelerator="auto", num_workers=0, load_best_checkpoint=False,
                model_ds=None, force_overwrite=False, **kwargs):
        print("=> [Predict]: Started.")

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

        # Get model dataset
        if model_ds:
            self.model_ds = model_ds
        elif self.model_ds:
            pass
        else:
            raise ValueError(f"Missing 'model_ds'. It's needed to get the model's path (training and eval).")

        # Store config
        self._add_config(key="predict", values=locals(), reset=False)
        self._add_config(key="predict", values=kwargs, reset=False)
        logs_path = self.model_ds.get_model_logs_path(toolkit=self.engine, run_name=self.model_ds.get_run_name(self.run_prefix))
        make_dir(logs_path)
        save_json(self.config, savepath=os.path.join(logs_path, "config_predict.json"))

        # Translate and score
        eval_scores = []
        eval_datasets = self.model_ds.get_eval_datasets(eval_datasets)
        for eval_ds in eval_datasets:
            self.translate(model_ds=self.model_ds, eval_ds=eval_ds, beams=beams, max_len_a=max_len_a, max_len_b=max_len_b,
                           truncate_at=truncate_at, batch_size=batch_size, max_tokens=max_tokens,
                           devices=devices, accelerator=accelerator, num_workers=num_workers,
                           load_best_checkpoint=load_best_checkpoint, force_overwrite=force_overwrite, **kwargs)
            self.score(model_ds=self.model_ds, eval_ds=eval_ds, beams=beams, metrics=metrics,
                       force_overwrite=force_overwrite, **kwargs)
            model_scores = self.parse_metrics(model_ds=self.model_ds, eval_ds=eval_ds, beams=beams, metrics=metrics,
                                              engine=self.engine, force_overwrite=force_overwrite, **kwargs)
            eval_scores.append(model_scores)
        return eval_scores

    @abstractmethod
    def _preprocess(self, *args, **kwargs):
        pass

    def preprocess(self, ds: Dataset, force_overwrite, **kwargs):
        print(f"=> [Preprocess]: Started. ({ds.id2(as_path=True)})")

        # Set vars
        src_lang = ds.src_lang
        trg_lang = ds.trg_lang
        train_path = ds.get_encoded_path(fname=ds.train_name)
        val_path = ds.get_encoded_path(fname=ds.val_name)
        test_path = ds.get_encoded_path(fname=ds.test_name)
        model_src_vocab_path = ds.get_vocab_file(lang=src_lang)
        model_trg_vocab_path = ds.get_vocab_file(lang=trg_lang)
        model_data_bin_path = ds.get_model_data_bin(toolkit=self.engine)

        # Create dirs
        make_dir([model_data_bin_path])

        start_time = time.time()
        self._preprocess(src_lang=src_lang, trg_lang=trg_lang, output_path=model_data_bin_path,
                         train_path=train_path, val_path=val_path, test_path=test_path,
                         src_vocab_path=model_src_vocab_path, trg_vocab_path=model_trg_vocab_path,
                         subword_model=ds.subword_model, pretok_flag=ds.pretok_flag,
                         force_overwrite=force_overwrite, **kwargs)
        print(f"\t- [INFO]: Preprocess time: {str(datetime.timedelta(seconds=time.time()-start_time))}")

    @abstractmethod
    def _train(self, *args, **kwargs):
        pass

    def train(self, train_ds: Dataset, resume_training, force_overwrite, **kwargs):
        print(f"=> [Train]: Started. ({train_ds.id2(as_path=True)})")

        # Check preprocessing
        _check_datasets(train_ds=train_ds)

        # Set run name
        run_name = train_ds.get_run_name(self.run_prefix)

        # Set paths
        data_bin_path = train_ds.get_model_data_bin(toolkit=self.engine)
        checkpoints_dir = train_ds.get_model_checkpoints_path(toolkit=self.engine, run_name=run_name)
        logs_path = train_ds.get_model_logs_path(toolkit=self.engine, run_name=run_name)

        # Create dirs
        make_dir([data_bin_path, checkpoints_dir, logs_path])

        # Set seed
        self.manual_seed(seed=kwargs.get("seed"))

        start_time = time.time()
        self._train(data_bin_path=data_bin_path, checkpoints_dir=checkpoints_dir, logs_path=logs_path,
                    run_name=run_name, ds_alias='_'.join(train_ds.id()),
                    resume_training=resume_training, force_overwrite=force_overwrite, **kwargs)
        print(f"\t- [INFO]: Training time: {str(datetime.timedelta(seconds=time.time()-start_time))}")

    @abstractmethod
    def _translate(self, *args, **kwargs):
        pass

    def translate(self, model_ds: Dataset, eval_ds: Dataset, beams: List[int], max_len_a, max_len_b, truncate_at,
                  batch_size, max_tokens, num_workers, force_overwrite, **kwargs):
        print(f"=> [Translate]: Started. ({model_ds.id2(as_path=True)})")

        # Check preprocessing
        _check_datasets(train_ds=model_ds, eval_ds=eval_ds)
        assert model_ds.dataset_lang_pair == eval_ds.dataset_lang_pair

        # Set run names
        run_name = model_ds.get_run_name(self.run_prefix)
        eval_name = '_'.join(eval_ds.id())  # Subword model and vocab size don't characterize the dataset!

        # Checkpoints dir
        checkpoints_dir = model_ds.get_model_checkpoints_path(self.engine, run_name)

        # [Trained model]: Create eval folder
        model_src_vocab_path = model_ds.get_vocab_file(lang=model_ds.src_lang)  # Needed to preprocess
        model_trg_vocab_path = model_ds.get_vocab_file(lang=model_ds.trg_lang)  # Needed to preprocess
        model_eval_data_path = model_ds.get_model_eval_data_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name)
        model_eval_data_bin_path = model_ds.get_model_eval_data_bin_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name)

        # Create dirs
        make_dir([model_eval_data_path, model_eval_data_bin_path])

        # [Encode extern data]: Encode test data using the subword model of the trained model
        for ts_fname in [fname for fname in eval_ds.split_names_lang if eval_ds.test_name in fname]:
            lang = ts_fname.split('.')[-1]
            input_file = eval_ds.get_split_path(ts_fname)  # as raw as possible
            output_file = model_ds.get_model_eval_data_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name)

            # Create directories
            make_dir([
                os.path.join(output_file, "raw"),
                os.path.join(output_file, "normalized"),
                os.path.join(output_file, "tokenized"),
                os.path.join(output_file, "encoded"),
            ])

            # Copy raw
            raw_file = os.path.join(output_file, "raw", ts_fname)
            shutil.copyfile(input_file, raw_file)
            input_file = raw_file

            # Normalize data
            norm_file = os.path.join(output_file, "normalized", ts_fname)
            normalize_file(input_file=input_file, output_file=norm_file,
                           normalizer=model_ds.normalizer, force_overwrite=force_overwrite)
            input_file = norm_file

            # Pretokenize data (if needed)
            if model_ds.pretok_flag:
                pretok_file = os.path.join(output_file, "tokenized", ts_fname)
                pretokenize_file(input_file=input_file, output_file=pretok_file, lang=lang,
                                 force_overwrite=force_overwrite, use_cmd=self.use_cmd,
                                 venv_path=self.venv_path)
                input_file = pretok_file

            # Encode file
            enc_file = os.path.join(output_file, "encoded", ts_fname)
            encode_file(ds=model_ds, input_file=input_file, output_file=enc_file,
                        lang=lang, merge_vocabs=model_ds.merge_vocabs, truncate_at=truncate_at,
                        force_overwrite=force_overwrite,
                        use_cmd=self.use_cmd, venv_path=self.venv_path)

        # Preprocess external data
        test_path = os.path.join(model_eval_data_path, "encoded", eval_ds.test_name)
        self._preprocess(src_lang=model_ds.src_lang, trg_lang=model_ds.trg_lang,
                         output_path=model_eval_data_bin_path,
                         train_path=None, val_path=None, test_path=test_path,
                         src_vocab_path=model_src_vocab_path, trg_vocab_path=model_trg_vocab_path,
                         subword_model=model_ds.subword_model, pretok_flag=model_ds.pretok_flag,
                         external_data=True, force_overwrite=force_overwrite,
                         **kwargs)

        # Iterate over beams
        for beam in beams:
            start_time = time.time()
            # Create output path (if needed)
            output_path = model_ds.get_model_beam_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name, beam=beam)
            make_dir(output_path)

            # Translate
            tok_flag = [os.path.exists(os.path.join(output_path, f)) for f in ["hyp.tok"]]
            if force_overwrite or not all(tok_flag):
                self._translate(
                    src_lang=model_ds.src_lang, trg_lang=model_ds.trg_lang,
                    beam_width=beam, max_len_a=max_len_a, max_len_b=max_len_b, batch_size=batch_size, max_tokens=max_tokens,
                    data_bin_path=model_eval_data_bin_path, output_path=output_path, checkpoints_dir=checkpoints_dir,
                    model_src_vocab_path=model_src_vocab_path, model_trg_vocab_path=model_trg_vocab_path,
                    num_workers=num_workers, model_ds=model_ds, force_overwrite=force_overwrite, **kwargs)

            # Copy src/ref raw
            for fname, lang in [("src", model_ds.src_lang), ("ref", model_ds.trg_lang)]:
                raw_file = model_ds.get_model_eval_data_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name, fname=f"normalized/test.{lang}")
                output_file = os.path.join(output_path, f"{fname}.txt")
                shutil.copyfile(raw_file, output_file)

            # Postprocess tokenized files
            for fname, lang in [("hyp", model_ds.trg_lang)]:
                input_file = os.path.join(output_path, f"{fname}.tok")
                output_file = os.path.join(output_path, f"{fname}.txt")
                model_vocab_path = model_src_vocab_path if lang == model_ds.src_lang else model_trg_vocab_path

                # Post-process files
                decode_file(input_file=input_file, output_file=output_file, lang=lang,
                            subword_model=model_ds.subword_model, pretok_flag=model_ds.pretok_flag,
                            model_vocab_path=model_vocab_path, remove_unk_hyphen=True,
                            force_overwrite=force_overwrite,
                            use_cmd=self.use_cmd, venv_path=self.venv_path)

            # Check amount of lines
            ref_lines = len(open(os.path.join(output_path, "ref.txt"), 'r').readlines())
            hyp_lines = len(open(os.path.join(output_path, "hyp.txt"), 'r').readlines())
            if ref_lines != hyp_lines:
                raise ValueError(f"The number of lines in 'ref.txt' ({ref_lines}) and 'hyp.txt' ({hyp_lines}) "
                                 f"does not match. If you see a 'CUDA out of memory' message, try again with "
                                 f"smaller batch.")

            print(f"\t- [INFO]: Translating time (beam={str(beam)}): {str(datetime.timedelta(seconds=time.time() - start_time))}")

    def score(self, model_ds: Dataset, eval_ds: Dataset, beams: List[int], metrics: Set[str], force_overwrite, **kwargs):
        print(f"=> [Score]: Started. ({model_ds.id2(as_path=True)})")

        # Check preprocessing
        _check_datasets(train_ds=model_ds, eval_ds=eval_ds)
        assert model_ds.dataset_lang_pair == eval_ds.dataset_lang_pair

        # Check supported metrics
        metrics_valid = _check_supported_metrics(metrics, self.METRICS2TOOL.keys())
        if not metrics_valid:
            return

        # Set run names
        run_name = model_ds.get_run_name(self.run_prefix)
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
                if force_overwrite or not os.path.exists(output_file):
                    py_cmd_api.compute_huggingface(src_file=src_file_path, hyp_file=hyp_file_path, ref_file=ref_file_path,
                                                   output_file=output_file, metrics=hg_metrics, trg_lang=model_ds.trg_lang,
                                                   use_cmd=self.use_cmd, venv_path=self.venv_path)

            # [CMD] Score: bleu, chrf and ter
            if self.TOOL2METRICS["sacrebleu"].intersection(metrics):
                ext = "json" if self.use_cmd else "json"
                output_file = os.path.join(scores_path, f"sacrebleu_scores.{ext}")
                if force_overwrite or not os.path.exists(output_file):
                    py_cmd_api.compute_sacrebleu(ref_file=ref_file_path, hyp_file=hyp_file_path, output_file=output_file, metrics=metrics, use_cmd=self.use_cmd, venv_path=self.venv_path)

            # [CMD] Score: bertscore
            if self.TOOL2METRICS["bertscore"].intersection(metrics):
                ext = "txt" if self.use_cmd else "json"
                output_file = os.path.join(scores_path, f"bertscore_scores.{ext}")
                if force_overwrite or not os.path.exists(output_file):
                    py_cmd_api.compute_bertscore(ref_file=ref_file_path, hyp_file=hyp_file_path, output_file=output_file, trg_lang=model_ds.trg_lang, use_cmd=self.use_cmd, venv_path=self.venv_path)

            # [CMD] Score: comet
            if self.TOOL2METRICS["comet"].intersection(metrics):
                ext = "txt" if self.use_cmd else "json"
                output_file = os.path.join(scores_path, f"comet_scores.{ext}")
                if force_overwrite or not os.path.exists(output_file):
                    py_cmd_api.compute_comet(src_file=src_file_path, ref_file=ref_file_path, hyp_file=hyp_file_path, output_file=output_file, use_cmd=self.use_cmd, venv_path=self.venv_path)

            # [CMD] Score: beer
            if self.TOOL2METRICS["beer"].intersection(metrics):
                ext = "txt" if self.use_cmd else "json"
                output_file = os.path.join(scores_path, f"beer_scores.{ext}")
                if force_overwrite or not os.path.exists(output_file):
                    py_cmd_api.compute_beer(ref_file=ref_file_path, hyp_file=hyp_file_path, output_file=output_file, use_cmd=self.use_cmd, venv_path=self.venv_path)

            # [CMD] Score: fairseq
            if self.TOOL2METRICS["fairseq"].intersection(metrics):
                ext = "txt" if self.use_cmd else "txt"
                output_file = os.path.join(scores_path, f"fairseq_scores.{ext}")
                if force_overwrite or not os.path.exists(output_file):
                    py_cmd_api.compute_fairseq(ref_file=ref_file_path, hyp_file=hyp_file_path, output_file=output_file, use_cmd=self.use_cmd, venv_path=self.venv_path)

            print(f"\t- [INFO]: Scoring time (beam={str(beam)}): {str(datetime.timedelta(seconds=time.time() - start_time))}")


    def parse_metrics(self, model_ds, eval_ds, beams: List[int], metrics: Set[str], force_overwrite, **kwargs):
        print(f"=> [Parsing]: Started. ({model_ds.id2(as_path=True)})")

        # Check preprocessing
        _check_datasets(train_ds=model_ds, eval_ds=eval_ds)
        assert model_ds.dataset_lang_pair == eval_ds.dataset_lang_pair

        # Check supported metrics
        metrics_valid = _check_supported_metrics(metrics, self.METRICS2TOOL.keys())
        if not metrics_valid:
            return

        # Metrics to retrieve
        metric_tools = self._get_metrics_tool(metrics)

        # Set run names
        run_name = model_ds.get_run_name(self.run_prefix)
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
        from pytorch_lightning.utilities.seed import seed_everything

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
        print(f"\t- [INFO]: Testing random seed ({seed}):")
        print(f"\t\t- random: {random.random()}")
        print(f"\t\t- numpy: {np.random.rand(1)}")
        print(f"\t\t- torch: {torch.rand(1)}")

        return seed
