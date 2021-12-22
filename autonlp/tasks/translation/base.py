import os.path
import shutil
from autonlp.utils import *

from abc import ABC, abstractmethod
from autonlp.cmd import tokenizers_entry
from autonlp.datasets import Dataset, DatasetBuilder
from autonlp.tasks.translation import metrics
from autonlp import utils

import pandas as pd


class BaseTranslator(ABC):

    # Global variables
    total_runs = 0
    METRICS_SUPPORTED = {"bleu", "chrf", "ter", "bertscore", "comet", "beer"}
    METRIC_PARSERS = {"sacrebleu": ("sacrebleu_scores.json", metrics.parse_sacrebleu),
                      "bertscore": ("bert_scores.txt", metrics.parse_bertscore),
                      "comet": ("comet_scores.txt", metrics.parse_comet),
                      "beer": ("beer_scores.txt", metrics.parse_beer)}

    def __init__(self, engine, run_prefix="model", num_gpus=None, conda_env_name=None, 
                 force_overwrite=False, interactive=True, **kwargs):
        # Store vars
        self.engine = engine
        self.run_prefix = run_prefix
        self.force_overwrite = force_overwrite
        self.interactive = interactive
        self.conda_env_name = conda_env_name
        self.train_ds = None

        # Parse gpu flag
        self.num_gpus = None if not num_gpus or num_gpus.strip().lower() == "all" else num_gpus

        # Other
        self.safe_seconds = 2

    def _make_empty_path(self, path, safe_seconds=0):
        # Check if the directory and can be delete it
        if self.force_overwrite and os.path.exists(path):
            print(f"=> [Existing data]: The contents of following directory are going to be deleted: {path}")
            res = ask_yes_or_no(question="Do you want to continue? [y/N]", interactive=self.interactive)
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

    def _check_datasets(self, train_ds=None, eval_ds=None):
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

    def fit(self, train_ds, *args, **kwargs):
        self.train_ds = train_ds
        self.preprocess(train_ds, *args, **kwargs)
        self.train(train_ds, *args, **kwargs)

    def predict(self, eval_datasets, beams=None, metrics=None, **kwargs):
        # Default metrics
        if metrics is None:
            metrics = {"bleu"}

        # Iterate over the evaluation datasets
        eval_scores = {}
        eval_datasets = eval_datasets if isinstance(eval_datasets, DatasetBuilder) else [eval_datasets]
        for eval_ds in eval_datasets:
            self.translate(train_ds=self.train_ds, eval_ds=eval_ds, beams=beams, **kwargs)
            self.score(train_ds=self.train_ds, eval_ds=eval_ds, beams=beams, metrics=metrics, **kwargs)
            scores = self.parse_metrics(train_ds=self.train_ds, eval_ds=eval_ds, beams=beams, metrics=metrics, **kwargs)
            eval_scores[str(eval_ds)] = scores
        return eval_scores

    @abstractmethod
    def _preprocess(self, *args, **kwargs):
        pass

    def preprocess(self, ds, *args, **kwargs):
        print("\t- [Preprocess]: Started.")

        # Check datasets
        if ds and not isinstance(ds, Dataset):
            raise TypeError("'ds' is not an instance of 'Dataset'")

        # Set vars
        src_lang = ds.src_lang
        trg_lang = ds.trg_lang
        model_data_bin_path = ds.get_model_data_bin(toolkit=self.engine)
        model_src_vocab_path = ds.get_src_trg_vocab_path()
        model_trg_vocab_path = ds.get_src_trg_vocab_path()
        train_path = os.path.join(ds.get_encoded_path(), ds.train_name)
        val_path = os.path.join(ds.get_encoded_path(), ds.val_name)
        test_path = os.path.join(ds.get_encoded_path(), ds.test_name)

        # Checks: Make sure the directory exist, and it is empty
        is_empty = self._make_empty_path(path=model_data_bin_path, safe_seconds=self.safe_seconds)
        if not is_empty:
            print("\t- [Preprocess]: Skipped. The output directory is not empty")
            return

        self._preprocess(*args, src_lang=src_lang, trg_lang=trg_lang, output_path=model_data_bin_path,
                         train_path=train_path, val_path=val_path, test_path=test_path,
                         src_vocab_path=model_src_vocab_path, trg_vocab_path=model_trg_vocab_path, **kwargs)

    @abstractmethod
    def _train(self, *args, **kwargs):
        pass

    def train(self, train_ds, *args, **kwargs):
        print("\t- [Train]: Started.")

        # Check datasets
        self._check_datasets(train_ds=train_ds)

        # Set run name
        run_name = f"{self.run_prefix}_{train_ds.subword_model}_{train_ds.vocab_size}"

        # Set paths
        data_bin_path = train_ds.get_model_data_bin(toolkit=self.engine)
        checkpoints_path = train_ds.get_model_checkpoints_path(toolkit=self.engine, run_name=run_name)
        logs_path = train_ds.get_model_logs_path(toolkit=self.engine, run_name=run_name)

        # Checks: Make sure the directory exist, and it is empty
        # is_empty = os.listdir(checkpoints_path) == []  # Too dangerous to allow overwrite
        is_empty = self._make_empty_path(path=checkpoints_path, safe_seconds=self.safe_seconds)
        if not is_empty:
            print("\t- [Train]: Skipped. The checkpoints directory is not empty")
            return

        self._train(*args, data_bin_path=data_bin_path, checkpoints_path=checkpoints_path, logs_path=logs_path, **kwargs)

    @abstractmethod
    def _translate(self, *args, **kwargs):
        pass

    def translate(self, train_ds, eval_ds, beams, max_gen_length=150, **kwargs):
        print("\t- [Translate]: Started.")

        # Check datasets
        self._check_datasets(train_ds=train_ds, eval_ds=eval_ds)

        # Check beams type
        if not isinstance(beams, list):
            raise ValueError("'beams' must be a list of integers")

        # Set run names
        run_name = f"{self.run_prefix}_{train_ds.subword_model}_{train_ds.vocab_size}"
        eval_name = f"{str(eval_ds)}"  # The subword model and vocab size depends on the trained model

        # Checkpoint path
        checkpoint_path = train_ds.get_model_checkpoints_path(self.engine, run_name, "checkpoint_best.pt")

        # [Trained model]: Create eval folder
        model_src_vocab_path = train_ds.get_src_trg_vocab_path()
        model_trg_vocab_path = train_ds.get_src_trg_vocab_path()
        model_eval_data_path = train_ds.get_model_eval_data_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name)
        model_eval_data_bin_path = train_ds.get_model_eval_data_bin_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name)
        make_dir([model_eval_data_path, model_eval_data_bin_path])

        # [Trained model]: SPM model path
        src_spm_model_path = train_ds.get_src_trg_vocab_path() + ".model"
        trg_spm_model_path = train_ds.get_src_trg_vocab_path() + ".model"

        # Checks: Make sure the directory exist, and it is empty
        is_empty = self._make_empty_path(path=model_eval_data_bin_path, safe_seconds=self.safe_seconds)
        if not is_empty:
            print("\t- [Translate]: Skipped. The output directory for the translation data is not empty")
            return

        # [Encode extern data]: Encode test data using the subword model of the trained model
        for ts_fname in [fname for fname in eval_ds.split_names_lang if eval_ds.test_name in fname]:
            ori_filename = eval_ds.get_split_path(ts_fname)
            new_filename = train_ds.get_model_eval_data_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name, fname=ts_fname)

            # Check if the file exists
            if self.force_overwrite or not os.path.exists(new_filename):
                assert src_spm_model_path == trg_spm_model_path
                tokenizers_entry.spm_encode(spm_model_path=src_spm_model_path, input_file=ori_filename, output_file=new_filename, conda_env_name=self.conda_env_name)

        # Preprocess external data
        test_path = os.path.join(model_eval_data_path, eval_ds.test_name)
        self._preprocess(src_lang=train_ds.src_lang, trg_lang=train_ds.trg_lang,
                        output_path=model_eval_data_bin_path,
                        train_path=None, val_path=None, test_path=test_path,
                        src_vocab_path=model_src_vocab_path, trg_vocab_path=model_trg_vocab_path, external_data=True,
                        **kwargs)

        # Iterate over beams
        for beam in beams:
            # Create output path (if needed)
            output_path = train_ds.get_model_beam_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name, beam=beam)
            make_dir(output_path)

            # Checks: Make sure the directory exist, and it is empty
            is_empty = self._make_empty_path(path=output_path, safe_seconds=self.safe_seconds)
            if not is_empty:
                print(f"\t- [Translate]: Skipped for beam={beam}. The output directory is not empty")
                return

            # Translate
            self._translate(src_lang=train_ds.src_lang, trg_lang=train_ds.trg_lang,
                            data_path=model_eval_data_bin_path, output_path=output_path,
                            checkpoint_path=checkpoint_path,
                            src_spm_model_path=src_spm_model_path, trg_spm_model_path=trg_spm_model_path,
                            beam_width=beam, max_gen_length=max_gen_length, **kwargs)

    def score(self, train_ds, eval_ds, beams=None, metrics=None, **kwargs):
        # Check datasets
        self._check_datasets(train_ds=train_ds, eval_ds=eval_ds)

        # Check beams type
        if not isinstance(beams, list):
            raise ValueError("'beams' must be a list of integers")

        # Check metrics type
        if not (isinstance(metrics, list) or isinstance(metrics, dict) or isinstance(metrics, set)):
            raise ValueError("'beams' must be a list, dict or set")

        # Check supported metrics
        metrics_diff = metrics.difference(self.METRICS_SUPPORTED)
        if metrics_diff:
            print(f"=> [WARNING] These metrics are not supported: {str(metrics_diff)}")
            if metrics == metrics_diff:
                print("\t- [Score]: Skipped. No valid metrics were found.")
                return

        # Set run names
        run_name = f"{self.run_prefix}_{train_ds.subword_model}_{train_ds.vocab_size}"
        eval_name = f"{str(eval_ds)}"  # The subword model and vocab size depends on the trained model

        # Iterate over beams
        for beam in beams:
            # Paths
            beam_path = train_ds.get_model_beam_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name, beam=beam)
            scores_path = train_ds.get_model_scores_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name, beam=beam)
            make_dir([scores_path])

            # Disable checks
            # # Checks: Make sure the directory exist, and it is empty
            # is_empty = self._make_empty_path(path=scores_path, safe_seconds=self.safe_seconds)
            # if not is_empty:
            #     print("\t- [Score]: Skipped. The output directory is not empty")
            #     continue

            # Set input files (results)
            src_file_path = os.path.join(beam_path, "src.txt")
            ref_file_path = os.path.join(beam_path, "ref.txt")
            hyp_file_path = os.path.join(beam_path, "hyp.txt")

            # Score: bleu, chrf and ter
            if metrics.intersection({"bleu", "chr", "ter"}):
                output_file = os.path.join(scores_path, "sacrebleu_scores.json")
                if self.force_overwrite or not os.path.exists(output_file):
                    tokenizers_entry.cmd_sacrebleu(ref_file=ref_file_path, hyp_file=hyp_file_path, output_file=output_file, metrics=metrics, conda_env_name=self.conda_env_name)

            # Score: bertscore
            if metrics.intersection({"bertscore"}):
                output_file = os.path.join(scores_path, "bertscore_scores.txt")
                if self.force_overwrite or not os.path.exists(output_file):
                    tokenizers_entry.cmd_bertscore(ref_file=ref_file_path, hyp_file=hyp_file_path, output_file=output_file, trg_lang=train_ds.trg_lang, conda_env_name=self.conda_env_name)

            # Score: comet
            if metrics.intersection({"comet"}):
                output_file = os.path.join(scores_path, "comet_scores.txt")
                if self.force_overwrite or not os.path.exists(output_file):
                    tokenizers_entry.cmd_cometscore(src_file=src_file_path, ref_file=ref_file_path, hyp_file=hyp_file_path, output_file=output_file, conda_env_name=self.conda_env_name)

            # Score: beer
            if metrics.intersection({"beer"}):
                output_file = os.path.join(scores_path, "beer_scores.txt")
                if self.force_overwrite or not os.path.exists(output_file):
                    tokenizers_entry.cmd_beer(ref_file=ref_file_path, hyp_file=hyp_file_path, output_file=output_file, conda_env_name=self.conda_env_name)

    def parse_metrics(self, train_ds, eval_ds, beams=None, metrics=None, **kwargs):
        # Check datasets
        self._check_datasets(train_ds=train_ds, eval_ds=eval_ds)

        # Check beams type
        if not isinstance(beams, list):
            raise ValueError("'beams' must be a list of integers")

        # Check metrics type
        if not (isinstance(metrics, list) or isinstance(metrics, dict) or isinstance(metrics, set)):
            raise ValueError("'beams' must be a list, dict or set")

        # Check supported metrics
        metrics_diff = metrics.difference(self.METRICS_SUPPORTED)
        if metrics_diff:
            print(f"=> [WARNING] These metrics are not supported: {str(metrics_diff)}")
            if metrics == metrics_diff:
                print("\t- [Score]: Skipped. No valid metrics were found.")
                return

        # Set run names
        run_name = f"{self.run_prefix}_{train_ds.subword_model}_{train_ds.vocab_size}"
        eval_name = f"{str(eval_ds)}"  # The subword model and vocab size depends on the trained model

        # Walk through beams
        scores = {
            "train_dataset": str(train_ds), "train_lines": train_ds.dataset_lines,
            "train_lang_pair": train_ds.dataset_lang_pair,
            "eval_dataset": str(eval_ds), "eval_lines": eval_ds.dataset_lines,
            "eval_lang_pair": eval_ds.dataset_lang_pair,
            "run_name": run_name, "subword_model": train_ds.subword_model, "vocab_size": train_ds.vocab_size,
            "beams": {}
        }

        # Iterate over beams
        for beam in beams:
            # Paths
            scores_path = train_ds.get_model_scores_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name,
                                                         beam=beam)

            # Walk through metric files
            beam_scores = {}
            for m_tool, (m_fname, m_parser) in self.METRIC_PARSERS.items():

                # Read file
                filename = os.path.join(scores_path, m_fname)
                if os.path.exists(filename):
                    with open(filename, 'r') as f:
                        m_scores = m_parser(text=f.readlines())
                        for key, value in m_scores.items():
                            m_name = f"{m_tool}_{key}".lower().strip()
                            beam_scores[m_name] = value
                else:
                    logging.info(f"There are no metrics for '{m_tool}'")

            # Add beam scores
            scores["beams"].update({f"beam_{str(beam)}": beam_scores})
        return scores

