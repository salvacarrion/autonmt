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
                 splits_sizes, subword_model, vocab_size, merge_vocabs,
                 preprocess_raw_fn=None, preprocess_splits_fn=None, preprocess_predict_fn=None,
                 train_name="train", val_name="val", test_name="test",
                 # Data
                 data_path="data", data_raw_path="0_raw", data_raw_preprocessed_path="1_raw_preprocessed",
                 data_splits_path="1_splits", data_splits_preprocessed_path="2_preprocessed",
                 data_pretokenized_path="3_pretokenized", data_encoded_path="4_encoded",
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
        self.pretok_flag = self.subword_model in {"word"}
        self.merge_vocabs = merge_vocabs

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
        self.raw_preprocessed_name = "data"

        # Data paths
        self.data_path = data_path
        self.data_raw_path = os.path.join(self.data_path, data_raw_path)
        self.data_raw_preprocessed_path = os.path.join(self.data_path, data_raw_preprocessed_path)
        self.data_splits_path = os.path.join(self.data_path, data_splits_path)
        self.data_splits_preprocessed_path = os.path.join(self.data_path, data_splits_preprocessed_path)
        self.data_pretokenized_path = os.path.join(self.data_path, data_pretokenized_path)
        self.data_encoded_path = os.path.join(self.data_path, data_encoded_path)

        # Models path
        self.models_path = "models"
        self.models_runs_path = "runs"

        # Stats paths
        self.stats_path = stats_path

        # Vocabs path
        self.vocab_path = vocab_path
        self.plots_path = plots_path

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

    def get_raw_auto_path(self, fname=""):
        if self.preprocess_raw_fn:
           return self.get_raw_preprocessed_path(fname)
        else:
            return self.get_raw_path(fname)

    def get_split_path(self, fname=""):
        return os.path.join(self.base_path, *self.id(), self.data_splits_path, fname)

    def get_splits_preprocessed_path(self, fname=""):
        return os.path.join(self.base_path, *self.id(), self.data_splits_preprocessed_path, fname)

    def get_splits_auto_path(self, fname=""):
        if self.preprocess_splits_fn:
            return self.get_splits_preprocessed_path(fname)
        else:
            return self.get_split_path(fname)

    def get_pretok_path(self, fname=""):
        return os.path.join(self.base_path, *self.id(), self.data_pretokenized_path, fname)

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
        toolkit_path = self.get_toolkit_path(toolkit)
        return os.path.join(toolkit_path, data_bin_name, *self.vocab_size_id(), fname)

    def get_runs_path(self, toolkit, fname=""):
        return os.path.join(self.base_path, *self.id(), self.models_path, toolkit, self.models_runs_path, fname)

    def get_plots_path(self):
        return os.path.join(self.base_path, *self.id(), self.plots_path, *self.vocab_size_id())

    def get_stats_path(self, fname=""):
        return os.path.join(self.base_path, *self.id(), self.stats_path, *self.vocab_size_id(), fname)

    def get_raw_fnames(self):
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

    def get_raw_preprocessed_fnames(self):
        return [f"{self.raw_preprocessed_name}.{ext}" for ext in (self.src_lang, self.trg_lang)]

    def get_split_fnames(self):
        return [f"{fname}.{ext}" for fname in self.split_names for ext in (self.src_lang, self.trg_lang)]

    def get_run_name(self, run_prefix):
        return f"{run_prefix}_{self.subword_model}_{self.vocab_size}".lower()

    def get_stats(self, splits=None, count_unknowns=False):

        if not splits:
            splits = self.get_split_fnames() # Split names

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
            row.update(utils.basic_stats(tokens_per_sentence, prefix=""))

            # Count unknowns
            if count_unknowns and self.subword_model not in {None, "none", "bytes"}:
                vocab_lang = self.dataset_lang_pair if self.merge_vocabs else split_lang
                vocab_path = self.get_vocab_path(vocab_lang) + ".vocab"
                vocab_keys = set([line.split('\t')[0] for line in utils.read_file_lines(vocab_path, autoclean=False)][4:])

                lines = utils.read_file_lines(self.get_encoded_path(fname), autoclean=True)
                unknowns = [len(set(line.split(' ')).difference(vocab_keys)) for line in lines]
                unknowns = np.array(unknowns)
                row.update(utils.basic_stats(unknowns, prefix="unknown_"))

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
                raw_files = self.get_raw_fnames()
                raw_files = [self.get_raw_path(f) for f in raw_files]
                return all([os.path.exists(p) for p in raw_files]), raw_files
            except ValueError as e:
                if verbose:
                    print(f"=> [ERROR CAPTURED]: {e}")
        return False, []

    def has_raw_preprocessed_files(self, verbose=True):
        # Check if the split directory exists (...with all the data)
        raw_preprocessed_path = self.get_raw_preprocessed_path()

        # Check if path exists
        if os.path.exists(raw_preprocessed_path):
            try:
                # Check files
                raw_preprocessed_files = self.get_raw_preprocessed_fnames()
                raw_preprocessed_files = [self.get_raw_preprocessed_path(f) for f in raw_preprocessed_files]
                return all([os.path.exists(p) for p in raw_preprocessed_files]), raw_preprocessed_files
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
            split_files = [self.get_split_path(f) for f in self.get_split_fnames()]
            return all([os.path.exists(p) for p in split_files]), split_files
        return False, []

    def check_vocab_folder_consistency(self, check_extra=False, custom_vocabs=False):
        default_vocab_extensions = ["model", "vocab"]

        if check_extra:
            default_vocab_extensions.append("vocabf")

        # Ignore datasets with no vocabs
        if self.subword_model in {None, "none", "bytes"}:
            return True

        # Check if it has a vocab folder
        vocab_path = self.get_vocab_path()
        if not os.path.exists(vocab_path):
            raise ValueError(f"=> [ERROR CAPTURED]: Vocab path does not exist: {vocab_path}")

        # Custom vocabs only need to check if all files exists
        if custom_vocabs:  # Any language
            num_expected_files = len(default_vocab_extensions) if self.merge_vocabs else 2*len(default_vocab_extensions)
        else:
            # Get expected vocab files
            lang_files = [f"{self.src_lang}-{self.trg_lang}"] if self.merge_vocabs else [self.src_lang, self.trg_lang]
            expected_files = [f"{self.get_vocab_file(lang=lang)}.{ext}" for lang in lang_files for ext in
                              default_vocab_extensions]

            # Check if all files exist
            missing_files = [os.path.split(f)[1] for f in expected_files if not os.path.exists(f)]
            if missing_files:
                raise ValueError(f"=> [ERROR CAPTURED]: Missing vocab files for dataset '{self.id(as_path=True)}': {missing_files}\n\t- Vocab path: {vocab_path}")

            # Get number of expected files
            num_expected_files = len(expected_files)

        # Check if there are extra files
        existing_files = [os.path.join(vocab_path, f) for f in os.listdir(vocab_path) if f.endswith(tuple(default_vocab_extensions))]
        if len(existing_files) != num_expected_files:
            msg = (f"Incorrect number of vocab files for dataset '{self.id(as_path=True)}'. Expected {num_expected_files}, found {len(existing_files)}."
                   f"\n\t- Reason: This can lead to potential vocabulary mismatches during training."
                   f"\n\t- Vocab path: {vocab_path}")
            if custom_vocabs:
                print(f"=> [WARNING]: {msg}")
            else:
                raise ValueError(f"=> [PROCESS ABORTED]: {msg}")
        return True
