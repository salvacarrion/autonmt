import os

import numpy as np

from autonmt.bundle import utils


class Dataset:
    # Level 0: Dataset name
    # Level 1: Dataset language pair
    # Level 2: Dataset size name
    # Level 3: data/models/stats/vocabs/plots
    # Level 4-data: raw/splits/pretokenized/encoded/...
    # Level 4-models: frameworks->runs->(run_name)->[checkpoints/eval/logs]
    def __init__(self, base_path, parent_ds, dataset_name, dataset_lang_pair, dataset_size_name, dataset_lines,
                 splits_sizes, subword_model, vocab_size, merge_vocabs, eval_mode,
                 preprocess_raw_fn=None, preprocess_splits_fn=None, preprocess_predict_fn=None,
                 train_name="train", val_name="val", test_name="test",
                 # Data
                 data_path="data", data_raw_path="0_raw", data_raw_preprocessed_path="1_raw_preprocessed",
                 data_splits_path="1_splits", data_splits_preprocessed_path="2_preprocessed",
                 data_pretokenized_path="3_pretokenized", data_encoded_path="4_encoded",
                 # Models
                 models_path="models", models_runs_path="runs", models_checkpoints_path="checkpoints",
                 model_logs_path="logs", models_eval_path="eval", models_eval_beam_path="beam", models_eval_beam_scores_path="scores",
                 # Stats
                 stats_path="stats",
                 # Vocabs
                 vocab_path="vocabs",
                 # Plots
                 plots_path="plots"):
        # Add properties
        self.base_path = base_path
        self.parent_ds = parent_ds
        self.dataset_name = dataset_name.strip()
        self.dataset_lang_pair = dataset_lang_pair.strip().lower()
        self.dataset_size_name = dataset_size_name.strip()
        self.dataset_lines = dataset_lines
        self.splits_sizes = splits_sizes
        self.langs = self.dataset_lang_pair.split("-")
        self.src_lang, self.trg_lang = self.langs

        # Dataset versions
        self.subword_model = str(subword_model).lower() if subword_model else subword_model
        self.vocab_size = str(vocab_size).lower() if vocab_size else vocab_size
        self.pretok_flag = (self.subword_model == "word")
        self.merge_vocabs = merge_vocabs
        self.eval_mode = eval_mode

        # Preprocessing
        self.preprocess_raw_fn = preprocess_raw_fn
        self.preprocess_splits_fn = preprocess_splits_fn
        self.preprocess_predict_fn = preprocess_predict_fn

        # Constants: split names
        self.train_name = train_name
        self.val_name = val_name
        self.test_name = test_name
        self.split_names = (self.train_name, self.val_name, self.test_name)
        self.split_names_lang = [f"{name}.{lang}" for name in self.split_names for lang in self.langs]

        # Data paths
        self.data_path = data_path
        self.data_raw_path = os.path.join(self.data_path, data_raw_path)
        self.data_raw_preprocessed_path = os.path.join(self.data_path, data_raw_preprocessed_path)
        self.data_splits_path = os.path.join(self.data_path, data_splits_path)
        self.data_splits_preprocessed_path = os.path.join(self.data_path, data_splits_preprocessed_path)
        self.data_pretokenized_path = os.path.join(self.data_path, data_pretokenized_path)
        self.data_encoded_path = os.path.join(self.data_path, data_encoded_path)

        # Models paths: toolkit/runs/model_name/[checkpoints, logs, eval]
        self.models_path = models_path
        self.models_runs_path = models_runs_path
        self.model_logs_path = model_logs_path
        self.models_checkpoints_path = models_checkpoints_path
        self.models_eval_path = models_eval_path
        # Models paths: Eval data
        self.models_eval_beam_path = models_eval_beam_path
        self.models_eval_beam_scores_path = models_eval_beam_scores_path

        # Stats paths
        self.stats_path = stats_path

        # Vocabs path
        self.vocab_path = vocab_path
        self.plots_path = plots_path

        # Filtering
        self.filter_train_langs = None
        self.filter_val_langs = None
        self.filter_test_langs = None

        # Other
        self.source_data = None  # Raw or Splits

    def __str__(self):
        if self.parent_ds:
            return '_'.join(list(self.id())).lower()
        else:
            return '_'.join(list(self.id()) + [str(self.subword_model), str(self.vocab_size)]).lower()

    def vocab_size_id(self):
        if self.subword_model in {None, "none"}:
            return ["none"]
        elif self.subword_model in {"bytes"}:
            return ["bytes"]
        else:
            return self.subword_model, self.vocab_size

    def id(self, as_path=False):
        t = self.dataset_name, self.dataset_lang_pair, self.dataset_size_name
        return os.path.join(*t) if as_path else t

    def id2(self, as_path=False):
        t = list(self.id()) + list(self.vocab_size_id())
        return os.path.join(*t) if as_path else t

    def get_path(self):
        return os.path.join(self.base_path, *self.id())

    def get_raw_path(self, fname=""):
        return os.path.join(self.base_path, *self.id(), self.data_raw_path, fname)

    def get_raw_preprocessed_path(self, fname=""):
        return os.path.join(self.base_path, *self.id(), self.data_raw_preprocessed_path, fname)

    def get_splits_preprocessed_path(self, fname=""):
        return os.path.join(self.base_path, *self.id(), self.data_splits_preprocessed_path, fname)

    def get_pretok_path(self, fname=""):
        return os.path.join(self.base_path, *self.id(), self.data_pretokenized_path, fname)

    def get_split_path(self, fname=""):
        return os.path.join(self.base_path, *self.id(), self.data_splits_path, fname)

    def get_encoded_path(self, fname=""):
        if self.subword_model in {None, "none"}:
            if self.pretok_flag:
                return os.path.join(self.base_path, *self.id(), self.data_pretokenized_path, fname)
            else:
                return os.path.join(self.base_path, *self.id(), self.data_splits_preprocessed_path, fname)
        else:
            return os.path.join(self.base_path, *self.id(), self.data_encoded_path, *self.vocab_size_id(), fname)

    def get_vocab_path(self, fname="", base=False):
        _vocab_size_id = [] if base else self.vocab_size_id()
        return os.path.join(self.base_path, *self.id(), self.vocab_path, *_vocab_size_id, fname)

    def get_vocab_file(self, lang=None):
        # "none" has no vocabs
        if self.subword_model in {None, "none"}:
            return None

        # Select vocab type
        if self.merge_vocabs:
            return os.path.join(self.base_path, *self.id(), self.vocab_path, *self.vocab_size_id(), f"{self.src_lang}-{self.trg_lang}")
        else:
            return os.path.join(self.base_path, *self.id(), self.vocab_path, *self.vocab_size_id(), f"{lang}")

    def get_toolkit_path(self, toolkit, fname=""):
        return os.path.join(self.base_path, *self.id(), self.models_path, toolkit, fname)

    def get_bin_data(self, toolkit, data_bin_name, fname=""):
        return os.path.join(self.get_toolkit_path(toolkit), data_bin_name, *self.vocab_size_id(), fname)

    def get_model_eval_path(self, toolkit, run_name, eval_name, fname=""):
        return os.path.join(self.base_path, *self.id(), self.models_path, toolkit, self.models_runs_path, run_name, self.models_eval_path, eval_name, fname)

    def get_model_eval_beam_path(self, toolkit, run_name, eval_name, beam=""):
        beam_n = f"beam{str(beam)}" if beam else ""
        return os.path.join(self.get_model_eval_path(toolkit, run_name, eval_name), self.models_eval_beam_path, beam_n)

    def get_model_eval_beam_scores_path(self, toolkit, run_name, eval_name, beam):
        return os.path.join(self.get_model_eval_beam_path(toolkit, run_name, eval_name, beam), self.models_eval_beam_scores_path)

    def get_model_eval_data_bin_path(self, toolkit, run_name, eval_name, data_bin_name, fname=""):
        return os.path.join(self.get_model_eval_path(toolkit, run_name, eval_name), data_bin_name, fname)

    def get_model_logs_path(self, toolkit, run_name, fname=""):
        return os.path.join(self.base_path, *self.id(), self.models_path, toolkit, self.models_runs_path, run_name, self.model_logs_path, fname)

    def get_model_checkpoints_path(self, toolkit, run_name, fname=""):
        return os.path.join(self.base_path, *self.id(), self.models_path, toolkit, self.models_runs_path, run_name, self.models_checkpoints_path, fname)

    def get_plots_path(self):
        return os.path.join(self.base_path, *self.id(), self.plots_path, *self.vocab_size_id())

    def get_stats_path(self, fname=""):
        return os.path.join(self.base_path, *self.id(), self.stats_path, *self.vocab_size_id(), fname)

    def get_raw_files(self):
        raw_files = [f for f in os.listdir(self.get_raw_path()) if f[-2:] in {self.src_lang, self.trg_lang}]

        # Check number of files
        if len(raw_files) != 2:
            raise ValueError(f"Invalid number of raw files. Found '{len(raw_files)}' files when expecting 2.")

        # Sort raw files
        src_path, trg_path = None, None
        for filename in raw_files:
            if filename[-2:].lower() == self.src_lang and src_path is None:  # Check extension
                src_path = self.get_raw_path(fname=filename)
            elif filename[-2:].lower() == self.trg_lang and trg_path is None:
                trg_path = self.get_raw_path(fname=filename)
            else:
                raise ValueError(f"Invalid file extension '{filename[-2:].lower()}' for file '{filename}'")
        assert os.path.isfile(src_path) and os.path.isfile(trg_path)  # Check files exist

        # Get filenames (consistency)
        src_path = os.path.basename(src_path)
        trg_path = os.path.basename(trg_path)

        return src_path, trg_path

    def get_raw_preprocessed_files(self):
        return [f"{self.raw_preprocessed_name}.{ext}" for ext in (self.src_lang, self.trg_lang)]

    def get_split_files(self):
        return [f"{fname}.{ext}" for fname in self.split_names for ext in (self.src_lang, self.trg_lang)]

    def get_compatible_datasets(self, ts_datasets):
        # Keep only relevant preprocessing
        compatible_datasets = []
        compatible_datasets_ids = set()
        for ds in ts_datasets:
            ds_name = '_'.join(os.path.join(*ds.id()[:2]))  # Exclude size name
            ds_ref_name = '_'.join(os.path.join(*ds.id()[:2]))

            # Check language compatibility
            if ds.langs != self.langs:
                print(f"Skipping '{ds_name}' as it is not compatible with the '{ds_ref_name}'")
                continue

            # Check if it has already been included
            if ds_name in compatible_datasets_ids:
                print(f"Skipping '{ds_name}' as a variant of it has already been included")
            else:
                compatible_datasets.append(ds)
                compatible_datasets_ids.add(ds_name)
        return compatible_datasets

    def get_eval_datasets(self, ts_datasets):
        if self.eval_mode == "compatible":
            compatible_datasets = self.get_compatible_datasets(ts_datasets)
            return compatible_datasets
        elif self.eval_mode == "same":
            return [eval_ds for eval_ds in ts_datasets if eval_ds.id() == self.id()]
        else:
            raise ValueError(f"Unknown 'eval_mode' ({str(self.eval_mode)})")

    def get_run_name(self, run_prefix):
        return f"{run_prefix}_{self.subword_model}_{self.vocab_size}".lower()

    def get_stats(self, splits=None, count_unknowns=False):
        def basic_stats(tokens, prefix=""):
            d = {
                f"{prefix}total_sentences": len(tokens),
                f"{prefix}total_tokens": int(tokens.sum()),
                f"{prefix}max_tokens": int(np.max(tokens)),
                f"{prefix}min_tokens": int(np.min(tokens)),
                f"{prefix}avg_tokens": float(np.average(tokens)),
                f"{prefix}std_tokens": float(np.std(tokens)),
                f"{prefix}percentile5_tokens": int(np.percentile(tokens, 5)),
                f"{prefix}percentile50_tokens": int(np.percentile(tokens, 50)),
                f"{prefix}percentile95_tokens": int(np.percentile(tokens, 95)),
            }
            return d

        if not splits:
            splits = self.get_split_files()

        split_stats = {}
        for fname in splits:
            split_name, split_lang = fname.split('.')

            # Count tokens per sentence
            tokens_per_sentence = utils.count_tokens_per_sentence(filename=self.get_encoded_path(fname))
            tokens_per_sentence = np.array(tokens_per_sentence)

            # Compute stats
            row = {
                "split": fname,
                "subword_model": self.subword_model,
                "vocab_size": self.vocab_size,
            }
            row.update(basic_stats(tokens_per_sentence, prefix=""))

            # Count unknowns
            if count_unknowns and self.subword_model not in {None, "none", "bytes"}:
                vocab_path = self.get_vocab_path(split_lang) + ".vocab"
                vocab_keys = set([line.split('\t')[0] for line in utils.read_file_lines(vocab_path, autoclean=False)][4:])
                lines = utils.read_file_lines(self.get_encoded_path(fname), autoclean=True)
                unknowns = [len(set(line.split(' ')).difference(vocab_keys)) for line in lines]
                unknowns = np.array(unknowns)
                row.update(basic_stats(unknowns, prefix="unknown_"))

            # Add stats
            split_stats[fname] = row
        return split_stats

    def has_raw_files(self, verbose=True):
        # Check if the split directory exists (...with all the data)
        raw_path = self.get_raw_path()

        # Check if path exists
        if os.path.exists(raw_path):
            try:
                # Check files
                raw_files = self.get_raw_files()
                raw_files = [self.get_raw_path(f) for f in raw_files]
                return all([os.path.exists(p) for p in raw_files]), raw_files
            except ValueError as e:
                if verbose:
                    print(f"=> [ERROR CAPTURED]: {e}")
        return False, []

    def has_split_files(self):
        # Check if the split directory exists (...with all the data)
        splits_path = self.get_split_path()

        # Check if path exists
        if os.path.exists(splits_path):
            # Check files
            split_files = [self.get_split_path(f) for f in self.get_split_files()]
            return all([os.path.exists(p) for p in split_files]), split_files
        return False, []
