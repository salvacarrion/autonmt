import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from autonmt.bundle import utils
from autonmt.bundle.enums import SubwordModel
from autonmt.bundle.logger import get_logger

log = get_logger(__name__)


class DatasetLayout:
    """Pure path computer for a dataset variant.

    On-disk layout (everything under ``base_path/<dataset>/<lang-pair>/<size>/``)::

        data/0_raw/                  raw files (e.g. data.es / data.en)
        data/1_raw_preprocessed/     optional cleanup of raw files
        data/1_splits/               train/val/test splits
        data/2_preprocessed/         per-split cleanup
        data/3_pretokenized/         post-Moses pretokenization (subword=word)
        data/4_encoded/<sw>/<vs>/    subword-encoded files
        vocabs/<sw>/<vs>/            SentencePiece models + vocab files
        stats/<sw>/<vs>/             per-split statistics
        plots/<sw>/<vs>/             plots
        models/<toolkit>/runs/...    checkpoints, logs, eval

    Holding the identity (name/lang-pair/size/subword/vocab_size) and the path
    constants in one place makes paths trivially inspectable and testable
    without touching disk.
    """

    def __init__(self, base_path: str, dataset_name: str, dataset_lang_pair: str,
                 dataset_size_name: str,
                 subword_model: Optional[SubwordModel] = None,
                 vocab_size: Optional[str] = None,
                 byte_fallback: bool = False,
                 merge_vocabs: bool = False,
                 train_name: str = "train", val_name: str = "val", test_name: str = "test",
                 data_path: str = "data", data_raw_path: str = "0_raw",
                 data_raw_preprocessed_path: str = "1_raw_preprocessed",
                 data_splits_path: str = "1_splits",
                 data_splits_preprocessed_path: str = "2_preprocessed",
                 data_pretokenized_path: str = "3_pretokenized",
                 data_encoded_path: str = "4_encoded",
                 stats_path: str = "stats", vocab_path: str = "vocabs",
                 plots_path: str = "plots", models_path: str = "models",
                 models_runs_path: str = "runs"):
        self.base_path = base_path
        self.dataset_name = dataset_name
        self.dataset_lang_pair = dataset_lang_pair
        self.dataset_size_name = dataset_size_name
        self.src_lang, self.trg_lang = dataset_lang_pair.split("-")
        self.langs = (self.src_lang, self.trg_lang)

        self.subword_model = subword_model
        self.vocab_size = vocab_size
        self.byte_fallback = bool(byte_fallback)
        self.merge_vocabs = merge_vocabs

        self.train_name = train_name
        self.val_name = val_name
        self.test_name = test_name
        self.split_names = (train_name, val_name, test_name)
        self.raw_preprocessed_name = "data"

        self.data_path = data_path
        self.data_raw_path = os.path.join(data_path, data_raw_path)
        self.data_raw_preprocessed_path = os.path.join(data_path, data_raw_preprocessed_path)
        self.data_splits_path = os.path.join(data_path, data_splits_path)
        self.data_splits_preprocessed_path = os.path.join(data_path, data_splits_preprocessed_path)
        self.data_pretokenized_path = os.path.join(data_path, data_pretokenized_path)
        self.data_encoded_path = os.path.join(data_path, data_encoded_path)

        self.stats_path = stats_path
        self.vocab_path = vocab_path
        self.plots_path = plots_path
        self.models_path = models_path
        self.models_runs_path = models_runs_path

    @property
    def pretok_flag(self) -> bool:
        return self.subword_model is SubwordModel.WORD

    def id(self, as_path: bool = False):
        t = (self.dataset_name, self.dataset_lang_pair, self.dataset_size_name)
        return os.path.join(*t) if as_path else t

    def vocab_size_id(self) -> Sequence[str]:
        if self.subword_model in {None, "none"}:
            return ["none"]
        if self.subword_model in {"bytes"}:
            return ["bytes"]
        sw = str(self.subword_model)
        if self.byte_fallback:
            sw = f"{sw}+bytes"
        return (sw, str(self.vocab_size))

    def id2(self, as_path: bool = False):
        t = list(self.id()) + list(self.vocab_size_id())
        return os.path.join(*t) if as_path else t

    # --- Paths -----------------------------------------------------------

    def get_path(self) -> str:
        return os.path.join(self.base_path, *self.id())

    def get_raw_path(self, fname: str = "") -> str:
        return os.path.join(self.base_path, *self.id(), self.data_raw_path, fname)

    def get_raw_preprocessed_path(self, fname: str = "") -> str:
        return os.path.join(self.base_path, *self.id(), self.data_raw_preprocessed_path, fname)

    def get_split_path(self, fname: str = "") -> str:
        return os.path.join(self.base_path, *self.id(), self.data_splits_path, fname)

    def get_splits_preprocessed_path(self, fname: str = "") -> str:
        return os.path.join(self.base_path, *self.id(), self.data_splits_preprocessed_path, fname)

    def get_pretok_path(self, fname: str = "") -> str:
        return os.path.join(self.base_path, *self.id(), self.data_pretokenized_path, fname)

    def get_encoded_path(self, fname: str = "") -> str:
        if self.subword_model in {None, "none"}:
            if self.pretok_flag:
                return self.get_pretok_path(fname)
            return self.get_splits_preprocessed_path(fname)
        return os.path.join(self.base_path, *self.id(), self.data_encoded_path,
                            *self.vocab_size_id(), fname)

    def get_vocab_path(self, fname: str = "", base: bool = False) -> str:
        _vocab_size_id = [] if base else self.vocab_size_id()
        return os.path.join(self.base_path, *self.id(), self.vocab_path, *_vocab_size_id, fname)

    def get_vocab_file(self, lang: Optional[str] = None) -> Optional[str]:
        if self.subword_model in {None, "none"}:
            return None
        if self.merge_vocabs:
            return os.path.join(self.base_path, *self.id(), self.vocab_path,
                                *self.vocab_size_id(), f"{self.src_lang}-{self.trg_lang}")
        return os.path.join(self.base_path, *self.id(), self.vocab_path,
                            *self.vocab_size_id(), f"{lang}")

    def get_toolkit_path(self, toolkit: str, fname: str = "") -> str:
        return os.path.join(self.base_path, *self.id(), self.models_path, toolkit, fname)

    def get_bin_data(self, toolkit: str, data_bin_name: str, fname: str = "") -> str:
        return os.path.join(self.get_toolkit_path(toolkit), data_bin_name,
                            *self.vocab_size_id(), fname)

    def get_runs_path(self, toolkit: str, fname: str = "") -> str:
        return os.path.join(self.base_path, *self.id(), self.models_path, toolkit,
                            self.models_runs_path, fname)

    def get_plots_path(self) -> str:
        return os.path.join(self.base_path, *self.id(), self.plots_path,
                            *self.vocab_size_id())

    def get_stats_path(self, fname: str = "") -> str:
        return os.path.join(self.base_path, *self.id(), self.stats_path,
                            *self.vocab_size_id(), fname)

    def get_split_fnames(self) -> List[str]:
        return [f"{fname}.{ext}" for fname in self.split_names for ext in self.langs]

    def get_raw_preprocessed_fnames(self) -> List[str]:
        return [f"{self.raw_preprocessed_name}.{ext}" for ext in self.langs]


class Dataset:
    """A dataset variant: identity + state + on-disk stage checks.

    Path computation lives in :class:`DatasetLayout` (accessible via ``self.layout``);
    the ``get_*_path`` / ``get_*_fnames`` methods here delegate to it so existing user
    code (``ds.get_raw_path()`` etc.) keeps working unchanged.
    """

    # Level 0: Dataset name
    # Level 1: Dataset language pair
    # Level 2: Dataset size name
    # Level 3: data/models/stats/vocabs/plots
    # Level 4-data: raw/splits/pretokenized/encoded/...
    # Level 4-models: frameworks->runs->(run_name)->[checkpoints/eval/logs]
    def __init__(self, base_path, parent_ds, dataset_name, dataset_lang_pair, dataset_size_name, dataset_lines,
                 splits_sizes, subword_model, vocab_size, merge_vocabs, byte_fallback=False,
                 preprocess_raw_fn=None, preprocess_splits_fn=None, preprocess_predict_fn=None,
                 train_name="train", val_name="val", test_name="test",
                 data_path="data", data_raw_path="0_raw", data_raw_preprocessed_path="1_raw_preprocessed",
                 data_splits_path="1_splits", data_splits_preprocessed_path="2_preprocessed",
                 data_pretokenized_path="3_pretokenized", data_encoded_path="4_encoded",
                 stats_path="stats", vocab_path="vocabs", plots_path="plots"):
        self.parent_ds = parent_ds
        self.dataset_name = dataset_name.strip()
        self.dataset_lang_pair = dataset_lang_pair.strip().lower()
        self.dataset_size_name = dataset_size_name.strip()
        self.dataset_lines = dataset_lines
        self.splits_sizes = splits_sizes
        self.langs = self.dataset_lang_pair.split("-")
        self.src_lang, self.trg_lang = self.langs

        # SubwordModel is a str-Enum so existing equality checks against bare strings
        # ("word", "bytes", ...) keep working.
        # Sugar: a "<model>+bytes" string is split into model + byte_fallback=True.
        if isinstance(subword_model, str) and subword_model.lower().endswith("+bytes"):
            subword_model = subword_model[: -len("+bytes")]
            byte_fallback = True
        if subword_model is None or (isinstance(subword_model, str) and subword_model.lower() == "none"):
            self.subword_model = None
        else:
            self.subword_model = SubwordModel.coerce(subword_model)
        self.vocab_size = str(vocab_size).lower() if vocab_size else vocab_size
        self.byte_fallback = bool(byte_fallback)
        self.merge_vocabs = merge_vocabs

        # byte_fallback only makes sense for SentencePiece-trained models
        if self.byte_fallback and self.subword_model not in (SubwordModel.BPE, SubwordModel.UNIGRAM,
                                                              SubwordModel.CHAR, SubwordModel.WORD):
            raise ValueError(
                f"byte_fallback=True is incompatible with subword_model={self.subword_model!r}. "
                f"It only applies to SentencePiece models (bpe, unigram, char, word)."
            )

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

        # Path layout (single source of truth for path computation)
        self.layout = DatasetLayout(
            base_path=base_path, dataset_name=self.dataset_name,
            dataset_lang_pair=self.dataset_lang_pair,
            dataset_size_name=self.dataset_size_name,
            subword_model=self.subword_model, vocab_size=self.vocab_size,
            byte_fallback=self.byte_fallback,
            merge_vocabs=merge_vocabs,
            train_name=train_name, val_name=val_name, test_name=test_name,
            data_path=data_path, data_raw_path=data_raw_path,
            data_raw_preprocessed_path=data_raw_preprocessed_path,
            data_splits_path=data_splits_path,
            data_splits_preprocessed_path=data_splits_preprocessed_path,
            data_pretokenized_path=data_pretokenized_path,
            data_encoded_path=data_encoded_path,
            stats_path=stats_path, vocab_path=vocab_path, plots_path=plots_path,
        )

        # Other
        self.source_data = None  # Raw or Splits

    # --- Backwards-compatible attribute facades --------------------------
    # Older user code reads things like ``ds.base_path`` and ``ds.data_raw_path``
    # directly; expose them as properties that defer to ``self.layout``.

    @property
    def base_path(self) -> str:
        return self.layout.base_path

    @property
    def pretok_flag(self) -> bool:
        return self.layout.pretok_flag

    @property
    def data_path(self) -> str:
        return self.layout.data_path

    @property
    def data_raw_path(self) -> str:
        return self.layout.data_raw_path

    @property
    def data_raw_preprocessed_path(self) -> str:
        return self.layout.data_raw_preprocessed_path

    @property
    def data_splits_path(self) -> str:
        return self.layout.data_splits_path

    @property
    def data_splits_preprocessed_path(self) -> str:
        return self.layout.data_splits_preprocessed_path

    @property
    def data_pretokenized_path(self) -> str:
        return self.layout.data_pretokenized_path

    @property
    def data_encoded_path(self) -> str:
        return self.layout.data_encoded_path

    @property
    def stats_path(self) -> str:
        return self.layout.stats_path

    @property
    def vocab_path(self) -> str:
        return self.layout.vocab_path

    @property
    def plots_path(self) -> str:
        return self.layout.plots_path

    @property
    def models_path(self) -> str:
        return self.layout.models_path

    @property
    def models_runs_path(self) -> str:
        return self.layout.models_runs_path

    def __str__(self) -> str:
        if self.parent_ds:
            return '_'.join(list(self.id())).lower()
        return '_'.join(list(self.id()) + [str(self.subword_model), str(self.vocab_size)]).lower()

    # --- Path facades (delegate to layout) -------------------------------

    def id(self, as_path: bool = False):
        return self.layout.id(as_path=as_path)

    def id2(self, as_path: bool = False):
        return self.layout.id2(as_path=as_path)

    def vocab_size_id(self):
        return self.layout.vocab_size_id()

    def get_path(self) -> str:
        return self.layout.get_path()

    def get_raw_path(self, fname: str = "") -> str:
        return self.layout.get_raw_path(fname)

    def get_raw_preprocessed_path(self, fname: str = "") -> str:
        return self.layout.get_raw_preprocessed_path(fname)

    def get_raw_auto_path(self, fname: str = "") -> str:
        if self.preprocess_raw_fn:
            return self.get_raw_preprocessed_path(fname)
        return self.get_raw_path(fname)

    def get_split_path(self, fname: str = "") -> str:
        return self.layout.get_split_path(fname)

    def get_splits_preprocessed_path(self, fname: str = "") -> str:
        return self.layout.get_splits_preprocessed_path(fname)

    def get_splits_auto_path(self, fname: str = "") -> str:
        if self.preprocess_splits_fn:
            return self.get_splits_preprocessed_path(fname)
        return self.get_split_path(fname)

    def get_pretok_path(self, fname: str = "") -> str:
        return self.layout.get_pretok_path(fname)

    def get_encoded_path(self, fname: str = "") -> str:
        return self.layout.get_encoded_path(fname)

    def get_vocab_path(self, fname: str = "", base: bool = False) -> str:
        return self.layout.get_vocab_path(fname=fname, base=base)

    def get_vocab_file(self, lang: Optional[str] = None) -> Optional[str]:
        return self.layout.get_vocab_file(lang=lang)

    def get_toolkit_path(self, toolkit: str, fname: str = "") -> str:
        return self.layout.get_toolkit_path(toolkit=toolkit, fname=fname)

    def get_bin_data(self, toolkit: str, data_bin_name: str, fname: str = "") -> str:
        return self.layout.get_bin_data(toolkit=toolkit, data_bin_name=data_bin_name, fname=fname)

    def get_runs_path(self, toolkit: str, fname: str = "") -> str:
        return self.layout.get_runs_path(toolkit=toolkit, fname=fname)

    def get_plots_path(self) -> str:
        return self.layout.get_plots_path()

    def get_stats_path(self, fname: str = "") -> str:
        return self.layout.get_stats_path(fname=fname)

    def get_split_fnames(self) -> List[str]:
        return self.layout.get_split_fnames()

    def get_raw_preprocessed_fnames(self) -> List[str]:
        return self.layout.get_raw_preprocessed_fnames()

    def get_run_name(self, run_prefix: str) -> str:
        sw = str(self.subword_model)
        if self.byte_fallback:
            sw = f"{sw}+bytes"
        return f"{run_prefix}_{sw}_{self.vocab_size}".lower()

    # --- Disk inspection -------------------------------------------------

    def get_raw_fnames(self) -> Tuple[str, str]:
        raw_files = [f for f in os.listdir(self.get_raw_path()) if f[-2:] in {self.src_lang, self.trg_lang}]

        if len(raw_files) != 2:
            raise ValueError(f"Invalid number of raw files. Found '{len(raw_files)}' files when expecting 2.")

        src_path, trg_path = None, None
        for filename in raw_files:
            if filename[-2:].lower() == self.src_lang and src_path is None:
                src_path = self.get_raw_path(fname=filename)
            elif filename[-2:].lower() == self.trg_lang and trg_path is None:
                trg_path = self.get_raw_path(fname=filename)
            else:
                raise ValueError(f"Invalid file extension '{filename[-2:].lower()}' for file '{filename}'")
        assert os.path.isfile(src_path) and os.path.isfile(trg_path)

        return os.path.basename(src_path), os.path.basename(trg_path)

    def has_raw_files(self, verbose: bool = True):
        raw_path = self.get_raw_path()
        if os.path.exists(raw_path):
            try:
                raw_files = self.get_raw_fnames()
                raw_files = [self.get_raw_path(f) for f in raw_files]
                return all(os.path.exists(p) for p in raw_files), raw_files
            except ValueError as e:
                if verbose:
                    log.warning(f"=> [ERROR CAPTURED]: {e}")
        return False, []

    def has_raw_preprocessed_files(self, verbose: bool = True):
        raw_preprocessed_path = self.get_raw_preprocessed_path()
        if os.path.exists(raw_preprocessed_path):
            try:
                files = self.get_raw_preprocessed_fnames()
                files = [self.get_raw_preprocessed_path(f) for f in files]
                return all(os.path.exists(p) for p in files), files
            except ValueError as e:
                if verbose:
                    log.warning(f"=> [ERROR CAPTURED]: {e}")
        return False, []

    def has_split_files(self):
        splits_path = self.get_split_path()
        if os.path.exists(splits_path):
            split_files = [self.get_split_path(f) for f in self.get_split_fnames()]
            return all(os.path.exists(p) for p in split_files), split_files
        return False, []

    # --- Stats -----------------------------------------------------------

    def get_stats(self, splits=None, count_unknowns: bool = False):
        if not splits:
            splits = self.get_split_fnames()

        split_stats = {}
        for fname in splits:
            split_name, split_lang = fname.split('.')

            tokens_per_sentence = utils.count_tokens_per_sentence(filename=self.get_encoded_path(fname))
            tokens_per_sentence = np.array(tokens_per_sentence)

            row = {
                "split": fname,
                "subword_model": self.subword_model,
                "vocab_size": self.vocab_size,
            }
            row.update(utils.basic_stats(tokens_per_sentence, prefix=""))

            if count_unknowns and self.subword_model not in {None, "none", "bytes"}:
                vocab_lang = self.dataset_lang_pair if self.merge_vocabs else split_lang
                vocab_path = self.get_vocab_path(vocab_lang) + ".vocab"
                vocab_keys = set([line.split('\t')[0] for line in utils.read_file_lines(vocab_path, autoclean=False)][4:])

                lines = utils.read_file_lines(self.get_encoded_path(fname), autoclean=True)
                unknowns = [len(set(line.split(' ')).difference(vocab_keys)) for line in lines]
                row.update(utils.basic_stats(np.array(unknowns), prefix="unknown_"))

            split_stats[fname] = row
        return split_stats

    # --- Vocab consistency ----------------------------------------------

    def check_vocab_folder_consistency(self, check_extra: bool = False, custom_vocabs: bool = False) -> bool:
        default_vocab_extensions = ["model", "vocab"]
        if check_extra:
            default_vocab_extensions.append("vocabf")

        if self.subword_model in {None, "none", "bytes"}:
            return True

        vocab_path = self.get_vocab_path()
        if not os.path.exists(vocab_path):
            raise ValueError(f"=> [ERROR CAPTURED]: Vocab path does not exist: {vocab_path}")

        if custom_vocabs:
            num_expected_files = len(default_vocab_extensions) if self.merge_vocabs else 2 * len(default_vocab_extensions)
        else:
            lang_files = [f"{self.src_lang}-{self.trg_lang}"] if self.merge_vocabs else [self.src_lang, self.trg_lang]
            expected_files = [f"{self.get_vocab_file(lang=lang)}.{ext}" for lang in lang_files for ext in
                              default_vocab_extensions]
            missing_files = [os.path.split(f)[1] for f in expected_files if not os.path.exists(f)]
            if missing_files:
                raise ValueError(f"=> [ERROR CAPTURED]: Missing vocab files for dataset '{self.id(as_path=True)}': {missing_files}\n\t- Vocab path: {vocab_path}")
            num_expected_files = len(expected_files)

        existing_files = [os.path.join(vocab_path, f) for f in os.listdir(vocab_path)
                          if f.endswith(tuple(default_vocab_extensions))]
        if len(existing_files) != num_expected_files:
            msg = (f"Incorrect number of vocab files for dataset '{self.id(as_path=True)}'. "
                   f"Expected {num_expected_files}, found {len(existing_files)}."
                   f"\n\t- Reason: This can lead to potential vocabulary mismatches during training."
                   f"\n\t- Vocab path: {vocab_path}")
            if custom_vocabs:
                log.warning(f"=> [WARNING]: {msg}")
            else:
                raise ValueError(f"=> [PROCESS ABORTED]: {msg}")
        return True
