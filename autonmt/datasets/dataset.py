"""Dataset variants: on-disk layout + builder-time state.

``DatasetLayout`` is a path engine — given an identity (name / lang-pair / size
/ subword model / vocab size) it computes every directory the framework writes
into. It has no I/O and no preprocessing knowledge, so it can be used in tests
without touching disk.

``Dataset`` extends the layout with the per-variant state the rest of the
pipeline needs: target line count, split sizes, the preprocessing callbacks
the user provided, and the disk-inspection helpers (``has_raw_files``,
``get_stats`` …).
"""
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from autonmt.utils.fileio import read_file_lines
from autonmt.datasets.stats import basic_stats, count_tokens_per_sentence
from autonmt.utils.enums import SubwordModel, has_vocab, is_bytes_only, is_no_model
from autonmt.utils.logger import get_logger

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
    """

    def __init__(self, base_path: str, dataset_name: str, dataset_lang_pair: str,
                 dataset_size_name: str,
                 subword_model=None,
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
        # Sugar: "<model>+bytes" → (model, byte_fallback=True). Single parse site.
        subword_model, sugar_bf = SubwordModel.parse_with_byte_fallback(
            subword_model, default_byte_fallback=byte_fallback)
        byte_fallback = sugar_bf

        self.base_path = base_path
        self.dataset_name = dataset_name
        self.dataset_lang_pair = dataset_lang_pair
        self.dataset_size_name = dataset_size_name
        self.src_lang, self.tgt_lang = dataset_lang_pair.split("-")
        self.langs = (self.src_lang, self.tgt_lang)

        self.subword_model = subword_model
        self.vocab_size = str(vocab_size).lower() if vocab_size else vocab_size
        self.byte_fallback = bool(byte_fallback)
        self.merge_vocabs = merge_vocabs

        # byte_fallback is a SentencePiece-only flag; reject obviously meaningless combos.
        if self.byte_fallback and subword_model is not None and not subword_model.uses_sentencepiece:
            raise ValueError(
                f"byte_fallback=True is incompatible with subword_model={subword_model!r}. "
                f"It only applies to SentencePiece models (bpe, unigram, char, word)."
            )

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

    # --- Identity --------------------------------------------------------

    @property
    def pretok_flag(self) -> bool:
        return self.subword_model is SubwordModel.WORD

    @property
    def split_names_lang(self) -> List[str]:
        return [f"{name}.{lang}" for name in self.split_names for lang in self.langs]

    def base_id(self, as_path: bool = False):
        t = (self.dataset_name, self.dataset_lang_pair, self.dataset_size_name)
        return os.path.join(*t) if as_path else t

    def vocab_size_id(self) -> Sequence[str]:
        if is_no_model(self.subword_model):
            return ["none"]
        if is_bytes_only(self.subword_model):
            return ["bytes"]
        sw = str(self.subword_model)
        if self.byte_fallback:
            sw = f"{sw}+bytes"
        return (sw, str(self.vocab_size))

    def variant_id(self, as_path: bool = False):
        t = list(self.base_id()) + list(self.vocab_size_id())
        return os.path.join(*t) if as_path else t

    def get_run_name(self, run_prefix: str) -> str:
        sw = str(self.subword_model)
        if self.byte_fallback:
            sw = f"{sw}+bytes"
        return f"{run_prefix}_{sw}_{self.vocab_size}".lower()

    # --- Paths -----------------------------------------------------------

    def _path(self, *parts: str) -> str:
        return os.path.join(self.base_path, *self.base_id(), *parts)

    def get_path(self) -> str:
        return self._path()

    def get_raw_path(self, fname: str = "") -> str:
        return self._path(self.data_raw_path, fname)

    def get_raw_preprocessed_path(self, fname: str = "") -> str:
        return self._path(self.data_raw_preprocessed_path, fname)

    def get_split_path(self, fname: str = "") -> str:
        return self._path(self.data_splits_path, fname)

    def get_splits_preprocessed_path(self, fname: str = "") -> str:
        return self._path(self.data_splits_preprocessed_path, fname)

    def get_pretok_path(self, fname: str = "") -> str:
        return self._path(self.data_pretokenized_path, fname)

    def get_encoded_path(self, fname: str = "") -> str:
        if is_no_model(self.subword_model):
            if self.pretok_flag:
                return self.get_pretok_path(fname)
            return self.get_splits_preprocessed_path(fname)
        return self._path(self.data_encoded_path, *self.vocab_size_id(), fname)

    def get_vocab_path(self, fname: str = "", base: bool = False) -> str:
        _vocab_size_id = [] if base else self.vocab_size_id()
        return self._path(self.vocab_path, *_vocab_size_id, fname)

    def get_vocab_file(self, lang: Optional[str] = None) -> Optional[str]:
        if not has_vocab(self.subword_model):
            return None
        lang_tag = f"{self.src_lang}-{self.tgt_lang}" if self.merge_vocabs else lang
        return self._path(self.vocab_path, *self.vocab_size_id(), f"{lang_tag}")

    def get_toolkit_path(self, toolkit: str, fname: str = "") -> str:
        return self._path(self.models_path, toolkit, fname)

    def get_bin_data(self, toolkit: str, data_bin_name: str, fname: str = "") -> str:
        return os.path.join(self.get_toolkit_path(toolkit), data_bin_name,
                            *self.vocab_size_id(), fname)

    def get_runs_path(self, toolkit: str, fname: str = "") -> str:
        return self._path(self.models_path, toolkit, self.models_runs_path, fname)

    def get_plots_path(self) -> str:
        return self._path(self.plots_path, *self.vocab_size_id())

    def get_stats_path(self, fname: str = "") -> str:
        return self._path(self.stats_path, *self.vocab_size_id(), fname)

    # --- Filenames -------------------------------------------------------

    def get_split_fnames(self) -> List[str]:
        return [f"{fname}.{ext}" for fname in self.split_names for ext in self.langs]

    def get_raw_preprocessed_fnames(self) -> List[str]:
        return [f"{self.raw_preprocessed_name}.{ext}" for ext in self.langs]


class Dataset(DatasetLayout):
    """A dataset variant: identity + state + on-disk stage checks.

    Extends :class:`DatasetLayout` (the pure path engine) with the per-variant
    state the pipeline mutates / consumes: the target line count, split sizes,
    user preprocessing callbacks, and disk-inspection helpers.
    """

    def __init__(self, base_path, parent_ds, dataset_name, dataset_lang_pair, dataset_size_name,
                 dataset_lines, splits_sizes, subword_model, vocab_size, merge_vocabs,
                 byte_fallback=False,
                 preprocess_raw_fn=None, preprocess_splits_fn=None, preprocess_predict_fn=None,
                 **layout_kwargs):
        super().__init__(
            base_path=base_path,
            dataset_name=dataset_name.strip(),
            dataset_lang_pair=dataset_lang_pair.strip().lower(),
            dataset_size_name=dataset_size_name.strip(),
            subword_model=subword_model, vocab_size=vocab_size,
            byte_fallback=byte_fallback, merge_vocabs=merge_vocabs,
            **layout_kwargs,
        )
        self.parent_ds = parent_ds
        self.dataset_lines = dataset_lines
        self.splits_sizes = splits_sizes

        self.preprocess_raw_fn = preprocess_raw_fn
        self.preprocess_splits_fn = preprocess_splits_fn
        self.preprocess_predict_fn = preprocess_predict_fn

        self.source_data = None  # set later by DatasetBuilder

    def __str__(self) -> str:
        parts = list(self.base_id())
        if not self.parent_ds:
            parts += [str(self.subword_model), str(self.vocab_size)]
        return '_'.join(parts).lower()

    # --- "Auto" paths: pick preprocessed dir if the user gave a preprocess fn ---

    def get_raw_auto_path(self, fname: str = "") -> str:
        if self.preprocess_raw_fn:
            return self.get_raw_preprocessed_path(fname)
        return self.get_raw_path(fname)

    def get_splits_auto_path(self, fname: str = "") -> str:
        if self.preprocess_splits_fn:
            return self.get_splits_preprocessed_path(fname)
        return self.get_split_path(fname)

    # --- Disk inspection -------------------------------------------------

    def get_raw_fnames(self) -> Tuple[str, str]:
        raw_files = [f for f in os.listdir(self.get_raw_path())
                     if f[-2:] in {self.src_lang, self.tgt_lang}]

        if len(raw_files) != 2:
            raise ValueError(f"Invalid number of raw files. Found '{len(raw_files)}' files when expecting 2.")

        src_path, tgt_path = None, None
        for filename in raw_files:
            ext = filename[-2:].lower()
            if ext == self.src_lang and src_path is None:
                src_path = self.get_raw_path(fname=filename)
            elif ext == self.tgt_lang and tgt_path is None:
                tgt_path = self.get_raw_path(fname=filename)
            else:
                raise ValueError(f"Invalid file extension '{ext}' for file '{filename}'")
        assert os.path.isfile(src_path) and os.path.isfile(tgt_path)

        return os.path.basename(src_path), os.path.basename(tgt_path)

    def has_raw_files(self, verbose: bool = True):
        if not os.path.exists(self.get_raw_path()):
            return False, []
        try:
            raw_files = [self.get_raw_path(f) for f in self.get_raw_fnames()]
            return all(os.path.exists(p) for p in raw_files), raw_files
        except ValueError as e:
            if verbose:
                log.warning(f"=> [ERROR CAPTURED]: {e}")
            return False, []

    def has_raw_preprocessed_files(self, verbose: bool = True):
        if not os.path.exists(self.get_raw_preprocessed_path()):
            return False, []
        try:
            files = [self.get_raw_preprocessed_path(f) for f in self.get_raw_preprocessed_fnames()]
            return all(os.path.exists(p) for p in files), files
        except ValueError as e:
            if verbose:
                log.warning(f"=> [ERROR CAPTURED]: {e}")
            return False, []

    def has_split_files(self):
        if not os.path.exists(self.get_split_path()):
            return False, []
        split_files = [self.get_split_path(f) for f in self.get_split_fnames()]
        return all(os.path.exists(p) for p in split_files), split_files

    # --- Stats -----------------------------------------------------------

    def get_stats(self, splits=None, count_unknowns: bool = False):
        if not splits:
            splits = self.get_split_fnames()

        split_stats = {}
        for fname in splits:
            split_name, split_lang = fname.split('.')

            tokens_per_sentence = np.array(
                count_tokens_per_sentence(filename=self.get_encoded_path(fname)))

            row = {
                "split": fname,
                "subword_model": self.subword_model,
                "vocab_size": self.vocab_size,
            }
            row.update(basic_stats(tokens_per_sentence, prefix=""))

            if count_unknowns and has_vocab(self.subword_model):
                vocab_lang = self.dataset_lang_pair if self.merge_vocabs else split_lang
                vocab_path = self.get_vocab_path(vocab_lang) + ".vocab"
                vocab_keys = set(
                    line.split('\t')[0]
                    for line in read_file_lines(vocab_path, autoclean=False)[4:]
                )

                lines = read_file_lines(self.get_encoded_path(fname), autoclean=True)
                unknowns = [len(set(line.split(' ')).difference(vocab_keys)) for line in lines]
                row.update(basic_stats(np.array(unknowns), prefix="unknown_"))

            split_stats[fname] = row
        return split_stats

    # --- Vocab builders --------------------------------------------------

    def build_vocabs(self, max_tokens=None, **kwargs):
        """Build the src/tgt vocabularies for this dataset variant.

        Returns ``(src_vocab, tgt_vocab)``. When ``self.merge_vocabs`` is True,
        both entries are the same shared :class:`Vocabulary` instance (loaded
        from the joint vocab file), so the caller can still write
        ``src_vocab, tgt_vocab = ds.build_vocabs(...)`` without special-casing.

        ``max_tokens`` and any extra ``**kwargs`` are forwarded to the
        :class:`Vocabulary` constructor.
        """
        from autonmt.vocabularies import Vocabulary

        if self.merge_vocabs:
            shared = Vocabulary(max_tokens=max_tokens, **kwargs).build_from_ds(
                ds=self, lang=self.dataset_lang_pair)
            return shared, shared

        src_vocab = Vocabulary(max_tokens=max_tokens, **kwargs).build_from_ds(
            ds=self, lang=self.src_lang)
        tgt_vocab = Vocabulary(max_tokens=max_tokens, **kwargs).build_from_ds(
            ds=self, lang=self.tgt_lang)
        return src_vocab, tgt_vocab

    # --- Vocab consistency ----------------------------------------------

    def check_vocab_folder_consistency(self, check_extra: bool = False, custom_vocabs: bool = False) -> bool:
        if not has_vocab(self.subword_model):
            return True

        default_extensions = ["model", "vocab"]
        if check_extra:
            default_extensions.append("vocabf")

        vocab_path = self.get_vocab_path()
        if not os.path.exists(vocab_path):
            raise ValueError(f"=> [ERROR CAPTURED]: Vocab path does not exist: {vocab_path}")

        if custom_vocabs:
            num_expected_files = len(default_extensions) * (1 if self.merge_vocabs else 2)
        else:
            lang_files = ([f"{self.src_lang}-{self.tgt_lang}"] if self.merge_vocabs
                          else [self.src_lang, self.tgt_lang])
            expected_files = [f"{self.get_vocab_file(lang=lang)}.{ext}"
                              for lang in lang_files for ext in default_extensions]
            missing_files = [os.path.split(f)[1] for f in expected_files if not os.path.exists(f)]
            if missing_files:
                raise ValueError(
                    f"=> [ERROR CAPTURED]: Missing vocab files for dataset '{self.base_id(as_path=True)}': "
                    f"{missing_files}\n\t- Vocab path: {vocab_path}"
                )
            num_expected_files = len(expected_files)

        existing_files = [os.path.join(vocab_path, f) for f in os.listdir(vocab_path)
                          if f.endswith(tuple(default_extensions))]
        if len(existing_files) != num_expected_files:
            msg = (f"Incorrect number of vocab files for dataset '{self.base_id(as_path=True)}'. "
                   f"Expected {num_expected_files}, found {len(existing_files)}."
                   f"\n\t- Reason: This can lead to potential vocabulary mismatches during training."
                   f"\n\t- Vocab path: {vocab_path}")
            if custom_vocabs:
                log.warning(f"=> {msg}")
            else:
                raise ValueError(f"=> [PROCESS ABORTED]: {msg}")
        return True
