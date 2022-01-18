import os
from autonmt.bundle import utils
import numpy as np

class Dataset:
    def __init__(self, base_path, parent_ds,
                 dataset_name, dataset_lang_pair, dataset_size_name, dataset_lines,
                 subword_model, vocab_size, merge_vocabs, eval_mode, normalization, strip_whitespace,
                 collapse_whitespace, letter_case, file_encoding,
                 train_name="train", val_name="val", test_name="test",
                 raw_path=os.path.join("data", "raw"), splits_path=os.path.join("data", "splits"),
                 encoded_path=os.path.join("data", "encoded"), normalized_path=os.path.join("data", "normalized"),
                 pretokenized_path=os.path.join("data", "pretokenized"),
                 models_path="models", models_data_bin_path="data-bin", models_runs_path="runs",
                 models_checkpoints_path="checkpoints", model_logs_path="logs", models_eval_path="eval",
                 models_eval_data_path="data", models_beam_path="beams", models_scores_path="scores",
                 vocab_path=os.path.join("vocabs"), plots_path="plots", stats_path="stats"):
        # Add properties
        self.base_path = base_path
        self.parent_ds = parent_ds
        self.dataset_name = dataset_name.strip()
        self.dataset_lang_pair = dataset_lang_pair.strip().lower()
        self.dataset_size_name = dataset_size_name.strip()
        self.dataset_lines = dataset_lines
        self.langs = self.dataset_lang_pair.split("-")
        self.src_lang, self.trg_lang = self.langs

        # Dataset versions
        self.subword_model = str(subword_model).lower() if subword_model else subword_model
        self.vocab_size = str(vocab_size).lower() if vocab_size else vocab_size
        self.pretok_flag = (self.subword_model == "word")
        self.merge_vocabs = merge_vocabs
        self.eval_mode = eval_mode

        # Constants: split names
        self.train_name = train_name
        self.val_name = val_name
        self.test_name = test_name
        self.split_names = (self.train_name, self.val_name, self.test_name)
        self.split_names_lang = [f"{name}.{lang}" for name in self.split_names for lang in self.langs]

        # Add default paths
        self.data_raw_path = raw_path
        self.data_splits_path = splits_path
        self.data_encoded_path = encoded_path
        self.data_normalized_path = normalized_path
        self.data_pretokenized_path = pretokenized_path
        self.models_path = models_path
        self.models_data_bin_path = models_data_bin_path
        self.models_runs_path = models_runs_path
        self.models_checkpoints_path = models_checkpoints_path
        self.model_logs_path = model_logs_path
        self.models_eval_path = models_eval_path
        self.models_eval_data_path = models_eval_data_path
        self.models_beam_path = models_beam_path
        self.models_scores_path = models_scores_path
        self.vocab_path = vocab_path
        self.plots_path = plots_path
        self.stats_path = stats_path

        # Encoding params
        self.normalization = normalization
        self.strip_whitespace = strip_whitespace
        self.collapse_whitespace = collapse_whitespace
        self.letter_case = letter_case
        self.file_encoding = file_encoding

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

    def id(self, as_path=False, include_size=True):
        if include_size:
            t = self.dataset_name, self.dataset_lang_pair, self.dataset_size_name
        else:
            t = self.dataset_name, self.dataset_lang_pair
        return os.path.join(*t) if as_path else t

    def id2(self, as_path=False):
        t = list(self.id()) + list(self.vocab_size_id())
        return os.path.join(*t) if as_path else t

    def get_path(self):
        return os.path.join(self.base_path, *self.id())

    def get_raw_path(self, fname=""):
        return os.path.join(self.base_path, *self.id(), self.data_raw_path, fname)

    def get_normalized_path(self, fname=""):
        return os.path.join(self.base_path, *self.id(), self.data_normalized_path, fname)

    def get_pretok_path(self, fname=""):
        return os.path.join(self.base_path, *self.id(), self.data_pretokenized_path, fname)

    def get_split_path(self, fname=""):
        return os.path.join(self.base_path, *self.id(), self.data_splits_path, fname)

    def get_encoded_path(self, fname=""):
        if self.subword_model in {None, "none"}:
            if self.pretok_flag:
                return os.path.join(self.base_path, *self.id(), self.data_pretokenized_path, fname)
            else:
                return os.path.join(self.base_path, *self.id(), self.data_normalized_path, fname)
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

    def get_model_data_bin(self, toolkit, fname=""):
        return os.path.join(self.base_path, *self.id(), self.models_path, toolkit, self.models_data_bin_path, *self.vocab_size_id(), fname)

    def get_model_eval_path(self, toolkit, run_name, eval_name):
        return os.path.join(self.base_path, *self.id(), self.models_path, toolkit, self.models_runs_path, run_name, self.models_eval_path, eval_name)

    def get_model_eval_data_path(self, toolkit, run_name, eval_name, fname=""):
        return os.path.join(self.base_path, *self.id(), self.models_path, toolkit, self.models_runs_path, run_name, self.models_eval_path, eval_name, self.models_eval_data_path, fname)

    def get_model_eval_data_bin_path(self, toolkit, run_name, eval_name, fname=""):
        return os.path.join(self.base_path, *self.id(), self.models_path, toolkit, self.models_runs_path, run_name, self.models_eval_path, eval_name, self.models_data_bin_path, fname)

    def get_model_beam_path(self, toolkit, run_name, eval_name, beam=""):
        beam_n = f"beam{str(beam)}" if beam else ""
        return os.path.join(self.base_path, *self.id(), self.models_path, toolkit, self.models_runs_path, run_name, self.models_eval_path, eval_name, self.models_beam_path, beam_n)

    def get_model_scores_path(self, toolkit, run_name, eval_name, beam):
        return os.path.join(self.get_model_beam_path(toolkit, run_name, eval_name, beam), self.models_scores_path)

    def get_model_logs_path(self, toolkit, run_name, fname=""):
        return os.path.join(self.base_path, *self.id(), self.models_path, toolkit, self.models_runs_path, run_name, self.model_logs_path, fname)

    def get_model_checkpoints_path(self, toolkit, run_name, fname=""):
        return os.path.join(self.base_path, *self.id(), self.models_path, toolkit, self.models_runs_path, run_name, self.models_checkpoints_path, fname)

    def get_plots_path(self):
        return os.path.join(self.base_path, *self.id(), self.plots_path, *self.vocab_size_id())

    def get_stats_path(self, fname=""):
        return os.path.join(self.base_path, *self.id(), self.stats_path, *self.vocab_size_id(), fname)

    def get_split_files(self):
        return [f"{fname}.{ext}" for fname in self.split_names for ext in (self.src_lang, self.trg_lang)]

    def get_compatible_datasets(self, ts_datasets):
        # Keep only relevant preprocessing
        compatible_datasets = []
        compatible_datasets_ids = set()
        for ds in ts_datasets:
            ds_name = '_'.join(ds.id(include_size=False))
            ds_ref_name = '_'.join(ds.id(include_size=False))

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
                vocab_keys = set([line.split('\t')[0] for line in utils.read_file_lines(vocab_path)][4:])
                lines = utils.read_file_lines(self.get_encoded_path(fname))
                unknowns = [len(set(line.split(' ')).difference(vocab_keys)) for line in lines]
                unknowns = np.array(unknowns)
                row.update(basic_stats(unknowns, prefix="unknown_"))

            # Add stats
            split_stats[fname] = row
        return split_stats
