import os.path
import shutil

from typing import List, Set, Iterable

from autonmt.utils import *

from abc import ABC, abstractmethod
from autonmt.cmd import cmd_tokenizers, cmd_metrics
from autonmt.datasets import Dataset, DatasetBuilder
from autonmt.tasks.translation.bundle import metrics


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


class BaseTranslator(ABC):

    # Global variables
    total_runs = 0
    METRICS_SUPPORTED = {"bleu", "chrf", "ter", "bertscore", "comet", "beer"}
    METRIC_PARSERS = {"sacrebleu": ("sacrebleu_scores.json", metrics.parse_sacrebleu),
                      "bertscore": ("bert_scores.txt", metrics.parse_bertscore),
                      "comet": ("comet_scores.txt", metrics.parse_comet),
                      "beer": ("beer_scores.txt", metrics.parse_beer)}

    def __init__(self, engine, run_prefix="model", num_gpus=None, conda_env_name=None, 
                 force_overwrite=False, interactive=True, model_ds=None, safe_seconds=2, **kwargs):
        # Store vars
        self.engine = engine
        self.run_prefix = run_prefix
        self.force_overwrite = force_overwrite
        self.interactive = interactive
        self.conda_env_name = conda_env_name

        # Add train_ds
        self.model_ds = model_ds
        _check_datasets(train_ds=self.model_ds) if self.model_ds else None

        # Parse gpu flag
        self.num_gpus = None if not num_gpus or num_gpus.strip().lower() == "all" else num_gpus

        # Other
        self.safe_seconds = safe_seconds

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

    def fit(self, train_ds: Dataset = None, batch_size=128, max_tokens=None, max_epochs=5, learning_rate=1e-3,
            weight_decay=0, clip_norm=1.0, patience=10, criterion="cross_entropy", optimizer="adam",
            checkpoints_path=None, logs_path=None, **kwargs):
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

        # Train and preprocess
        self.preprocess(train_ds, **kwargs)
        self.train(train_ds, batch_size=batch_size, max_tokens=max_tokens, max_epochs=max_epochs,
                   learning_rate=learning_rate, weight_decay=weight_decay, clip_norm=clip_norm, patience=patience,
                   criterion=criterion, optimizer=optimizer, **kwargs)

    def predict(self, eval_datasets: List[Dataset], model_ds: Dataset = None, beams: List[int] = None,
                metrics: Set[str] = None, batch_size=128, max_tokens=None, max_gen_length=150, **kwargs):
        print("=> [Predict]: Started.")

        # Get train_ds
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

        # Default metrics
        if metrics is None:
            metrics = {"bleu"}

        # Iterate over the evaluation datasets
        eval_scores = []
        eval_datasets = eval_datasets if isinstance(eval_datasets, DatasetBuilder) else [eval_datasets]
        for eval_ds in eval_datasets:
            self.translate(model_ds=model_ds, eval_ds=eval_ds, beams=beams, max_gen_length=max_gen_length,
                           batch_size=batch_size, max_tokens=max_tokens, **kwargs)
            self.score(model_ds=model_ds, eval_ds=eval_ds, beams=beams, metrics=metrics, **kwargs)
            model_scores = self.parse_metrics(model_ds=model_ds, eval_ds=eval_ds, beams=beams, metrics=metrics, **kwargs)
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
        train_path = os.path.join(ds.get_encoded_path(), ds.train_name)
        val_path = os.path.join(ds.get_encoded_path(), ds.val_name)
        test_path = os.path.join(ds.get_encoded_path(), ds.test_name)
        model_src_vocab_path = ds.get_src_trg_vocab_path()
        model_trg_vocab_path = ds.get_src_trg_vocab_path()
        model_data_bin_path = ds.get_model_data_bin(toolkit=self.engine)

        # Create dirs
        make_dir([model_data_bin_path])

        # Checks: Make sure the directory exist, and it is empty
        is_empty = self._make_empty_path(path=model_data_bin_path, safe_seconds=self.safe_seconds)
        if not is_empty:
            print("\t- [Preprocess]: Skipped. The output directory is not empty")
            return

        self._preprocess(src_lang=src_lang, trg_lang=trg_lang, output_path=model_data_bin_path,
                         train_path=train_path, val_path=val_path, test_path=test_path,
                         src_vocab_path=model_src_vocab_path, trg_vocab_path=model_trg_vocab_path, **kwargs)

    @abstractmethod
    def _train(self, *args, **kwargs):
        pass

    def train(self, train_ds: Dataset, **kwargs):
        print("=> [Train]: Started.")

        # Check datasets
        _check_datasets(train_ds=train_ds)

        # Set run name
        run_name = f"{self.run_prefix}_{train_ds.subword_model}_{train_ds.vocab_size}"

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

        self._train(data_bin_path=data_bin_path, checkpoints_path=checkpoints_path, logs_path=logs_path, **kwargs)

    @abstractmethod
    def _translate(self, *args, **kwargs):
        pass

    def translate(self, model_ds: Dataset, eval_ds: Dataset, beams: List[int], **kwargs):
        print("=> [Translate]: Started.")

        # Check datasets
        _check_datasets(train_ds=model_ds, eval_ds=eval_ds)

        # Set run names
        run_name = f"{self.run_prefix}_{model_ds.subword_model}_{model_ds.vocab_size}"
        eval_name = f"{str(eval_ds)}"  # The subword model and vocab size depends on the trained model

        # Checkpoint path
        checkpoint_path = model_ds.get_model_checkpoints_path(self.engine, run_name, "checkpoint_best.pt")

        # [Trained model]: Create eval folder
        model_src_vocab_path = model_ds.get_src_trg_vocab_path()
        model_trg_vocab_path = model_ds.get_src_trg_vocab_path()
        model_eval_data_path = model_ds.get_model_eval_data_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name)
        model_eval_data_bin_path = model_ds.get_model_eval_data_bin_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name)

        # Create dirs
        make_dir([model_eval_data_path, model_eval_data_bin_path])

        # [Trained model]: SPM model path
        src_spm_model_path = model_ds.get_src_trg_vocab_path() + ".model"
        trg_spm_model_path = model_ds.get_src_trg_vocab_path() + ".model"

        # Checks: Make sure the directory exist, and it is empty
        is_empty = self._make_empty_path(path=model_eval_data_bin_path, safe_seconds=self.safe_seconds)
        if not is_empty:
            print("\t- [Translate]: Skipped preprocessing. The output directory for the preprocessing data is not empty")
        else:
            # [Encode extern data]: Encode test data using the subword model of the trained model
            for ts_fname in [fname for fname in eval_ds.split_names_lang if eval_ds.test_name in fname]:
                ori_filename = eval_ds.get_split_path(ts_fname)
                new_filename = model_ds.get_model_eval_data_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name, fname=ts_fname)

                # Check if the file exists
                if self.force_overwrite or not os.path.exists(new_filename):
                    assert src_spm_model_path == trg_spm_model_path
                    cmd_tokenizers.spm_encode(spm_model_path=src_spm_model_path, input_file=ori_filename, output_file=new_filename, conda_env_name=self.conda_env_name)

            # Preprocess external data
            test_path = os.path.join(model_eval_data_path, eval_ds.test_name)
            self._preprocess(src_lang=model_ds.src_lang, trg_lang=model_ds.trg_lang,
                             output_path=model_eval_data_bin_path,
                             train_path=None, val_path=None, test_path=test_path,
                             src_vocab_path=model_src_vocab_path, trg_vocab_path=model_trg_vocab_path, external_data=True,
                             **kwargs)

        # Iterate over beams
        beams = list(set(beams))
        beams.sort(reverse=True)
        for beam in beams:
            # Create output path (if needed)
            output_path = model_ds.get_model_beam_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name, beam=beam)
            make_dir(output_path)

            # Checks: Make sure the directory exist, and it is empty
            is_empty = self._make_empty_path(path=output_path, safe_seconds=self.safe_seconds)
            if not is_empty:
                print(f"\t- [Translate]: Skipped for beam={beam}. The output directory is not empty")
                return

            # Translate
            self._translate(beam_width=beam, src_lang=model_ds.src_lang, trg_lang=model_ds.trg_lang,
                            data_path=model_eval_data_bin_path, output_path=output_path,
                            checkpoint_path=checkpoint_path,
                            src_spm_model_path=src_spm_model_path, trg_spm_model_path=trg_spm_model_path, **kwargs)

            # Postprocess tokenized files
            self.postprocess_tok_files(output_path=output_path,
                                       src_spm_model_path=src_spm_model_path, trg_spm_model_path=trg_spm_model_path)

    def postprocess_tok_files(self, output_path, src_spm_model_path, trg_spm_model_path):
        # Input files
        src_tok_path = os.path.join(output_path, "src.tok")
        ref_tok_path = os.path.join(output_path, "ref.tok")
        hyp_tok_path = os.path.join(output_path, "hyp.tok")

        # Output files
        src_txt_path = os.path.join(output_path, "src.txt")
        ref_txt_path = os.path.join(output_path, "ref.txt")
        hyp_txt_path = os.path.join(output_path, "hyp.txt")

        # Detokenize
        cmd_tokenizers.spm_decode(src_spm_model_path, input_file=src_tok_path, output_file=src_txt_path, conda_env_name=self.conda_env_name)
        cmd_tokenizers.spm_decode(trg_spm_model_path, input_file=ref_tok_path, output_file=ref_txt_path, conda_env_name=self.conda_env_name)
        cmd_tokenizers.spm_decode(trg_spm_model_path, input_file=hyp_tok_path, output_file=hyp_txt_path, conda_env_name=self.conda_env_name)


    def score(self, model_ds: Dataset, eval_ds: Dataset, beams: List[int], metrics: Set[str], **kwargs):
        print("=> [Score]: Started.")

        # Check datasets
        _check_datasets(train_ds=model_ds, eval_ds=eval_ds)

        # Check supported metrics
        metrics = set(metrics)
        metrics_diff = metrics.difference(self.METRICS_SUPPORTED)
        if metrics_diff:
            print(f"=> [WARNING] These metrics are not supported: {str(metrics_diff)}")
            if metrics == metrics_diff:
                print("\t- [Score]: Skipped. No valid metrics were found.")
                return

        # Set run names
        run_name = f"{self.run_prefix}_{model_ds.subword_model}_{model_ds.vocab_size}"
        eval_name = f"{str(eval_ds)}"  # The subword model and vocab size depends on the trained model

        # Iterate over beams
        beams = list(set(beams))
        beams.sort(reverse=True)
        for beam in beams:
            # Paths
            beam_path = model_ds.get_model_beam_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name, beam=beam)
            scores_path = model_ds.get_model_scores_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name, beam=beam)

            # Create dirs
            make_dir([scores_path])

            # Set input files (results)
            src_file_path = os.path.join(beam_path, "src.txt")
            ref_file_path = os.path.join(beam_path, "ref.txt")
            hyp_file_path = os.path.join(beam_path, "hyp.txt")

            # Score: bleu, chrf and ter
            if metrics.intersection({"bleu", "chr", "ter"}):
                output_file = os.path.join(scores_path, "sacrebleu_scores.json")
                if self.force_overwrite or not os.path.exists(output_file):
                    cmd_metrics.cmd_sacrebleu(ref_file=ref_file_path, hyp_file=hyp_file_path, output_file=output_file, metrics=metrics, conda_env_name=self.conda_env_name)

            # Score: bertscore
            if metrics.intersection({"bertscore"}):
                output_file = os.path.join(scores_path, "bertscore_scores.txt")
                if self.force_overwrite or not os.path.exists(output_file):
                    cmd_metrics.cmd_bertscore(ref_file=ref_file_path, hyp_file=hyp_file_path, output_file=output_file, trg_lang=model_ds.trg_lang, conda_env_name=self.conda_env_name)

            # Score: comet
            if metrics.intersection({"comet"}):
                output_file = os.path.join(scores_path, "comet_scores.txt")
                if self.force_overwrite or not os.path.exists(output_file):
                    cmd_metrics.cmd_cometscore(src_file=src_file_path, ref_file=ref_file_path, hyp_file=hyp_file_path, output_file=output_file, conda_env_name=self.conda_env_name)

            # Score: beer
            if metrics.intersection({"beer"}):
                output_file = os.path.join(scores_path, "beer_scores.txt")
                if self.force_overwrite or not os.path.exists(output_file):
                    cmd_metrics.cmd_beer(ref_file=ref_file_path, hyp_file=hyp_file_path, output_file=output_file, conda_env_name=self.conda_env_name)

    def parse_metrics(self, model_ds, eval_ds, beams: List[int], metrics: Set[str], **kwargs):
        print("=> [Parsing]: Started.")

        # Check datasets
        _check_datasets(train_ds=model_ds, eval_ds=eval_ds)

        # Check supported metrics
        metrics = set(metrics)
        metrics_diff = metrics.difference(self.METRICS_SUPPORTED)
        if metrics_diff:
            print(f"=> [WARNING] These metrics are not supported: {str(metrics_diff)}")
            if metrics == metrics_diff:
                print("\t- [Score]: Skipped. No valid metrics were found.")
                return

        # Set run names
        run_name = f"{self.run_prefix}_{model_ds.subword_model}_{model_ds.vocab_size}"
        eval_name = f"{str(eval_ds)}"  # The subword model and vocab size depends on the trained model

        # Walk through beams
        scores = {
            "train_dataset": str(model_ds), "train_lines": model_ds.dataset_lines,
            "train_lang_pair": model_ds.dataset_lang_pair,
            "eval_dataset": str(eval_ds), "eval_lines": eval_ds.dataset_lines,
            "eval_lang_pair": eval_ds.dataset_lang_pair,
            "run_name": run_name, "subword_model": model_ds.subword_model, "vocab_size": model_ds.vocab_size,
            "beams": {}
        }

        # Iterate over beams
        for beam in beams:
            # Paths
            scores_path = model_ds.get_model_scores_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name,
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

