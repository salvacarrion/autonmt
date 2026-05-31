"""Cross-product unroller + stage orchestrator for dataset variants.

Each user-declared (dataset × lang-pair × size × subword × vocab_size) cell
becomes one :class:`Dataset`. The builder unrolls the grid, materialises files
on disk in fixed stages (raw → splits → preprocessed → encoded → stats), and
caches enough state to iterate cleanly over the variants.

Plotting is intentionally NOT a builder responsibility: call
``autonmt.reporting.report.DatasetReport(ds).generate(...)`` after ``build()``
if you want diagnostic plots.
"""
import os
import random
import shutil
from itertools import islice

from autonmt.utils.enums import SourceData, SubwordModel, is_no_model
from autonmt.utils.fileio import (
    count_file_lines,
    make_dir,
    read_file_lines,
    save_json,
    write_file_lines,
)
from autonmt.utils.logger import get_logger
from autonmt.datasets.stats import parse_split_size, shuffle_in_order
from autonmt.datasets.dataset import Dataset
from autonmt.datasets.encoding import encode_file, pretokenize_file
from autonmt.vocabularies import vocab_builder

log = get_logger(__name__)


class DatasetBuilder:
    """Unrolls a (dataset × lang-pair × size × subword × vocab) grid and builds it on disk.

    Declare the experiment's data axes once; :meth:`build` materialises every
    cell as a :class:`~autonmt.datasets.dataset.Dataset` under ``base_path`` in
    fixed stages (raw → splits → preprocessed → encoded → stats/vocabs).
    :meth:`get_train_ds` / :meth:`get_test_ds` return the lists the experiment
    loop iterates over.

    Parameters
    ----------
    base_path : str
        Root directory under which all dataset variants are written.
    datasets : list of dict
        Dataset declarations; each gives a name, ``languages`` (``"xx-yy"``)
        and split sizes.
    encoding : list of dict, optional
        Subword-model / vocab-size axes to unroll. When omitted, only the
        parent (un-encoded) datasets are produced.
    merge_vocabs : bool, default False
        Train a single shared source+target vocabulary instead of one per side.
    preprocess_raw_fn : callable, optional
        Hook applied to the raw corpus before it is split.
    preprocess_splits_fn : callable, optional
        Hook applied to each split after splitting.
    random_seed : int, default 42
        Seeds the shuffles inside :meth:`build` for reproducibility.
    """

    DEFAULT_SPLIT_SIZES = (None, 1000, 1000)
    REF_SIZE_NAME = "original"

    # SentencePiece training defaults — exposed as class attributes so a user
    # can subclass to override without monkey-patching.
    INPUT_SENTENCE_SIZE = 1_000_000
    CHARACTER_COVERAGE = 1.0
    SPLIT_DIGITS = True

    def __init__(self, base_path, datasets, encoding=None, merge_vocabs=False,
                 preprocess_raw_fn=None, preprocess_splits_fn=None,
                 random_seed=42):
        self.base_path = base_path
        self.datasets = datasets
        self.encoding = encoding
        self.merge_vocabs = merge_vocabs
        # Seeds the stdlib RNG used by the shuffles inside build() (merge-vocab
        # SPM concat, optional split shuffling) so builds are reproducible.
        self.random_seed = random_seed

        self.preprocess_raw_fn = preprocess_raw_fn
        self.preprocess_splits_fn = preprocess_splits_fn

        self._check_lang_pairs()

        # Reference datasets (size="original") — every variant ultimately reads
        # from these via _create_reduced_versions.
        self.ds_refs = {
            str(ds): ds
            for ds in self._unroll_datasets(encodings=None, parent_ds=True, ref_size_only=True)
        }
        # One dataset per (name × lang × size) — used for preprocessing splits.
        self.ds_list_parents = self._unroll_datasets(encodings=None, parent_ds=True)
        # Full cross-product (only populated when encoding is provided).
        self.ds_list = []
        if encoding:
            self.ds_list = self._unroll_datasets(
                encodings=self._unroll_encoding(encoding), parent_ds=False)

    # --- Iteration / accessors -----------------------------------------

    def __iter__(self):
        return iter(self.ds_list)

    def __len__(self):
        return len(self.ds_list)

    def get_ds(self, ignore_variants=False):
        if self.ds_list and not ignore_variants:
            return self.ds_list
        return self.ds_list_parents

    def get_train_ds(self):
        return self.get_ds(ignore_variants=False)

    def get_test_ds(self):
        return self.get_ds(ignore_variants=True)

    # --- Validation -----------------------------------------------------

    def _check_lang_pairs(self):
        for d in self.datasets:
            for lang_pair in d.get("languages"):
                if len(lang_pair) != 5 or lang_pair[2] != '-':
                    raise ValueError("Language pairs must be defined with this format: 'xx-yy'")

    # --- Cross-product unrolling ---------------------------------------

    def _unroll_encoding(self, encoding):
        valid_enc = []
        seen = set()
        for enc in encoding:
            entry_fallback = bool(enc.get("byte_fallback", False))
            for model_raw in enc.get("subword_models"):
                model, bf = SubwordModel.parse_with_byte_fallback(
                    model_raw, default_byte_fallback=entry_fallback)
                # Canonical string form so dedup works across the "+bytes" sugar
                # and the explicit ``byte_fallback`` flag.
                canonical_model = str(model) if model is not None else "none"
                for size in enc.get("vocab_sizes"):
                    key = f"{canonical_model}_{size}_{int(bf)}"
                    if key in seen:
                        log.warning(f"Ignoring repeated entry in encoding "
                                    f"(subword={canonical_model}; size={size}; byte_fallback={bf})")
                        continue
                    valid_enc.append({"subword_model": canonical_model,
                                      "vocab_size": size, "byte_fallback": bf})
                    seen.add(key)
        return valid_enc

    def _unroll_datasets(self, encodings=None, parent_ds=None, ref_size_only=False):
        ds_unrolled = []
        empty_enc = [{"subword_model": None, "vocab_size": None, "byte_fallback": False}]
        encs = encodings if encodings else empty_enc

        for ds_entry in self.datasets:
            ds_name = ds_entry["name"]
            ds_splits_sizes = ds_entry.get("split_sizes", self.DEFAULT_SPLIT_SIZES)

            for ds_lang_pair in ds_entry["languages"]:
                sizes = [(self.REF_SIZE_NAME, None)] if ref_size_only else ds_entry["sizes"]
                for ds_size_name, ds_lines in sizes:
                    # Footgun guard: the size named REF_SIZE_NAME ("original")
                    # is the un-truncated reference; ``_create_reduced_versions``
                    # skips it. A caller passing ``("original", 5000)`` would
                    # silently get the full corpus.
                    if ds_size_name == self.REF_SIZE_NAME and ds_lines is not None:
                        log.warning(
                            f"\t- Size '({ds_size_name!r}, {ds_lines})' for "
                            f"dataset {ds_name!r}/{ds_lang_pair}: the line cap "
                            f"({ds_lines}) is IGNORED because '{ds_size_name}' "
                            f"is the reference (un-truncated) variant. To cap "
                            f"the training set, rename the size — e.g. "
                            f"('{ds_lines // 1000}k', {ds_lines})."
                        )
                    base = dict(
                        base_path=self.base_path,
                        dataset_name=ds_name,
                        dataset_lang_pair=ds_lang_pair,
                        dataset_size_name=ds_size_name,
                        dataset_lines=ds_lines,
                        merge_vocabs=self.merge_vocabs,
                        preprocess_raw_fn=self.preprocess_raw_fn,
                        preprocess_splits_fn=self.preprocess_splits_fn,
                        splits_sizes=ds_splits_sizes,
                        parent_ds=parent_ds,
                    )
                    for enc in encs:
                        ds_unrolled.append(Dataset(**base, **enc))
        return ds_unrolled

    # --- Public entrypoint ---------------------------------------------

    def build(self, force_overwrite=False, verbose=False):
        log.info(f"=> Building datasets...")
        log.info(f"\t- base_path={self.base_path}")

        # Seed the stdlib RNG so the shuffles below (merge-vocab SPM concat,
        # optional split/line shuffling) are reproducible across runs.
        random.seed(self.random_seed)

        # Classify which source each reference dataset has on disk and validate it.
        self._check_dir_structure(force_overwrite=force_overwrite)

        # Materialise files in order.
        self._preprocess_raw_files(force_overwrite=force_overwrite)
        self._create_splits()
        self._create_reduced_versions(force_overwrite=force_overwrite)
        self._preprocess_split_files(force_overwrite=force_overwrite)

        if not self.encoding:
            log.warning("\t- No encoding was specified")
            return self

        self._train_tokenizer(force_overwrite=force_overwrite)
        self._encode_datasets(force_overwrite=force_overwrite)
        self._export_vocab_frequencies(force_overwrite=force_overwrite)
        self._compute_stats(force_overwrite=force_overwrite, print_stats=verbose)
        return self

    # --- Stage: directory structure check ------------------------------

    def _check_dir_structure(self, force_overwrite):
        log.info(f"=> Checking directory structure...")
        invalid = False

        for ds in self.ds_refs.values():
            log.info(f"\t=> Checking dataset: '{ds.base_id(as_path=True)}'")
            raw_ok, raw_files = ds.has_raw_files(verbose=False)
            splits_ok, split_files = ds.has_split_files()

            # Splits take priority unless the user explicitly asked to rebuild from raw.
            use_splits = splits_ok and (not raw_ok or not force_overwrite)
            use_raw = raw_ok and not use_splits

            if use_splits:
                ds.source_data = SourceData.SPLITS
                if not self._validate_line_counts_pairwise(split_files):
                    log.error("\t\t- [Invalid data]: split files within each pair have different line counts.")
                    invalid = True
            elif use_raw:
                ds.source_data = SourceData.RAW
                if not self._validate_line_counts_pairwise(raw_files):
                    log.error("\t\t- [Invalid data]: the source and target raw files do not "
                              "have the same number of lines.")
                    invalid = True
            else:
                ds.source_data = None
                invalid = True
                self._report_missing_layout(ds)

        if invalid:
            raise FileNotFoundError(
                "Missing or invalid dataset files. Add the required raw/splits files in their "
                "respective folders and re-run."
            )

    @staticmethod
    def _validate_line_counts_pairwise(files):
        """Files are stored in (src, tgt, src, tgt, ...) order; every adjacent pair must match."""
        for i in range(0, len(files), 2):
            if count_file_lines(files[i]) != count_file_lines(files[i + 1]):
                return False
        return True

    @staticmethod
    def _report_missing_layout(ds):
        for kind, path in (("raw", ds.get_raw_path()), ("splits", ds.get_split_path())):
            label = "Missing folder" if not os.path.exists(path) else "Invalid data"
            log.error(f"\t\t=> [{label}]: '{kind}' at {path}")
        log.warning(f"\t\t=> [ACTION REQUIRED] Add a valid dataset to at least one of:")
        log.info(f"\t\t\t- '{ds.data_raw_path}': parallel corpus "
                 f"(e.g. 'data.{ds.src_lang}' and 'data.{ds.tgt_lang}')")
        log.info(f"\t\t\t- '{ds.data_splits_path}': pre-split corpus "
                 f"(e.g. '[train,val,test].[{ds.src_lang},{ds.tgt_lang}]')")

    # --- Stage: preprocessing the raw/split files ----------------------

    def _preprocess_files(self, ds, input_sets, output_sets, preprocess_fn, force_overwrite):
        for i, ((src_in, tgt_in), (src_out, tgt_out)) in enumerate(zip(input_sets, output_sets), 1):
            if not force_overwrite and all(os.path.exists(f) for f in (src_out, tgt_out)):
                continue
            log.info(f"\t=> Preprocessing file-pair ({i}/{len(input_sets)}) dataset '{ds.base_id(as_path=True)}'")
            src_lines = read_file_lines(src_in, autoclean=True)
            tgt_lines = read_file_lines(tgt_in, autoclean=True)

            if len(src_lines) != len(tgt_lines):
                log.error(f"=> The source and target files do not have the same number of lines "
                          f"({len(src_lines)} != {len(tgt_lines)})")
                log.info(f"\t- Source file: {src_in}")
                log.info(f"\t- Target file: {tgt_in}")

            data = {"src": {"lang": ds.src_lang, "lines": src_lines},
                    "tgt": {"lang": ds.tgt_lang, "lines": tgt_lines}}
            src_lines, tgt_lines = preprocess_fn(data, ds)

            write_file_lines(src_lines, filename=src_out, insert_break_line=True)
            write_file_lines(tgt_lines, filename=tgt_out, insert_break_line=True)

    def _preprocess_raw_files(self, force_overwrite):
        if self.preprocess_raw_fn is None:
            return

        log.info(f"=> Checking raw files...")
        for ds in self.ds_refs.values():
            if ds.source_data != SourceData.RAW:
                continue  # Splits supplied directly, or a previous run already preprocessed.
            ds.source_data = SourceData.RAW_PREPROCESSED
            make_dir(ds.get_raw_preprocessed_path())

            input_paths = [ds.get_raw_path(f) for f in ds.get_raw_fnames()]
            output_paths = [ds.get_raw_preprocessed_path(f) for f in ds.get_raw_preprocessed_fnames()]
            self._preprocess_files(
                ds, (input_paths[0:2],), (output_paths[0:2],),
                self.preprocess_raw_fn, force_overwrite,
            )

    def _preprocess_split_files(self, force_overwrite):
        if self.preprocess_splits_fn is None:
            return

        log.info(f"=> Preprocessing split files...")
        for ds in self.ds_list_parents:
            make_dir(ds.get_splits_preprocessed_path())

            in_paths = [ds.get_split_path(f) for f in ds.get_split_fnames()]
            out_paths = [ds.get_splits_preprocessed_path(f) for f in ds.get_split_fnames()]
            # Pair (src, tgt) per split: train, val, test.
            input_sets = tuple(in_paths[i:i + 2] for i in range(0, len(in_paths), 2))
            output_sets = tuple(out_paths[i:i + 2] for i in range(0, len(out_paths), 2))
            self._preprocess_files(ds, input_sets, output_sets,
                                   self.preprocess_splits_fn, force_overwrite)

    # --- Stage: split creation -----------------------------------------

    def _create_splits(self):
        # force_overwrite is implicit in ds.source_data (set by _check_dir_structure).
        log.info(f"=> Checking partitions...")
        for ds in self.ds_refs.values():
            if ds.source_data == SourceData.SPLITS:
                # User supplied splits directly; nothing to materialise.
                if all(os.path.exists(ds.get_split_path(f)) for f in ds.get_split_fnames()):
                    log.info(f"\t=> Partitions already exist for '{ds.base_id(as_path=True)}'")
                    continue
                raise ValueError(f"\t=> Some partitions are missing for '{ds.base_id(as_path=True)}'")

            log.info(f"\t=> Creating dataset partitions for '{ds.base_id(as_path=True)}'")
            self._materialise_splits_from_raw(ds)

    def _materialise_splits_from_raw(self, ds):
        if ds.source_data == SourceData.RAW:
            src_path, tgt_path = [ds.get_raw_path(f) for f in ds.get_raw_fnames()]
        elif ds.source_data == SourceData.RAW_PREPROCESSED:
            src_path, tgt_path = [ds.get_raw_preprocessed_path(f)
                                  for f in ds.get_raw_preprocessed_fnames()]
        else:
            raise ValueError(
                f"\t=> Invalid value for 'ds.source_data': {ds.source_data} "
                f"('raw', 'raw_preprocessed', or 'splits')"
            )
        assert os.path.isfile(src_path) and os.path.isfile(tgt_path)

        log.info(f"\t=> Processing from '{ds.source_data}'...")
        src_lines = read_file_lines(src_path)
        tgt_lines = read_file_lines(tgt_path)
        if len(src_lines) != len(tgt_lines):
            # zip would silently truncate; raw files must be aligned line-for-line.
            raise ValueError(
                f"Raw source/target line count mismatch for '{ds.base_id(as_path=True)}': "
                f"{len(src_lines)} ({src_path}) vs {len(tgt_lines)} ({tgt_path})"
            )
        lines = list(zip(src_lines, tgt_lines))

        _, val_size, test_size = ds.splits_sizes
        val_size = parse_split_size(val_size, max_ds_size=len(lines))
        test_size = parse_split_size(test_size, max_ds_size=len(lines))
        if (val_size + test_size) > len(lines):
            raise ValueError(f"\t=> The validation and test sets exceed the size of the dataset")

        train_lines = lines[:-(val_size + test_size)]
        val_lines = lines[-(val_size + test_size):-test_size]
        test_lines = lines[-test_size:]

        make_dir(ds.get_split_path())
        for split_lines, split_name in (
            (val_lines, ds.val_name),
            (test_lines, ds.test_name),
            (train_lines, ds.train_name),
        ):
            cols = list(zip(*split_lines))
            for i, lang in enumerate(ds.langs):
                savepath = ds.get_split_path(f"{split_name}.{lang}")
                write_file_lines(cols[i], savepath)
                log.info(f"\t\t- Partition saved: {split_name}.{lang}")

    def _create_reduced_versions(self, force_overwrite):
        log.info("=> Creating reduced versions...")
        for ds in self.ds_list_parents:
            if ds.base_id()[2] == self.REF_SIZE_NAME:
                continue  # The reference dataset itself; nothing to reduce.

            ref_id_parts = ds.base_id()[0], ds.base_id()[1], self.REF_SIZE_NAME
            make_dir(ds.get_split_path())
            log.info(f"\t=> Creating reduced version: {ds.base_id(as_path=True)}")

            for fname in ds.get_split_fnames():
                ref_filename = os.path.join(self.base_path, *ref_id_parts,
                                            ds.data_splits_path, fname)
                new_filename = ds.get_split_path(fname)
                if not force_overwrite and os.path.exists(new_filename):
                    continue

                log.info(f"\t\t- Creating split file: {fname}...")
                if fname.split('.')[0] == ds.train_name:
                    self._truncate_train_file(ref_filename, new_filename, ds.dataset_lines)
                else:
                    # val/test inherited from "original" unmodified.
                    shutil.copy(ref_filename, new_filename)

    @staticmethod
    def _truncate_train_file(ref_filename, new_filename, n_lines):
        with open(ref_filename, 'rb') as fin:
            lines = list(islice(fin, n_lines))  # streamed; doesn't load the whole file
        if n_lines is not None and len(lines) != n_lines:
            raise ValueError(
                f"[REDUCING FILES]: Not enough lines ({len(lines)} < {n_lines}) "
                f"in the training set: {ref_filename}")
        with open(new_filename, 'wb') as fout:
            fout.writelines(lines)

    # --- Stage: tokenizer training -------------------------------------

    def _pretokenize(self, ds, force_overwrite):
        if not ds.pretok_flag or is_no_model(ds.subword_model) or ds.subword_model is SubwordModel.BYTES:
            return

        make_dir([ds.get_pretok_path()])
        log.info(f"\t- Pretokenizing splits: {ds.base_id(as_path=True)}")
        for fname in ds.get_split_fnames():
            log.info(f"\t\t- Pretokenizing split file: {fname}...")
            lang = fname.split(".")[1]
            pretokenize_file(
                input_file=ds.get_splits_auto_path(fname),
                output_file=ds.get_pretok_path(fname),
                lang=lang, force_overwrite=force_overwrite,
            )

    def _train_tokenizer(self, force_overwrite):
        log.info(f"=> Building vocabularies...")
        for ds in self.ds_list:
            log.info(f"\t- Building vocabulary: {ds.variant_id(as_path=True)}")
            make_dir([ds.get_vocab_path(), ds.get_pretok_path()])

            if is_no_model(ds.subword_model):
                # Empty vocab dir, but kept for plots/stats consistency.
                continue

            if ds.subword_model is SubwordModel.BYTES:
                vocab_builder.write_bytes_vocab(ds, force_overwrite=force_overwrite)
                continue

            # SentencePiece-trained models: optionally pretokenise (word) and train.
            self._pretokenize(ds, force_overwrite)
            vocab_builder.train_spm(
                ds, force_overwrite=force_overwrite,
                input_sentence_size=self.INPUT_SENTENCE_SIZE,
                character_coverage=self.CHARACTER_COVERAGE,
                split_digits=self.SPLIT_DIGITS,
            )
            log.info(f"=> Checking existing vocabularies...")
            ds.check_vocab_folder_consistency()

    # --- Stage: encoding -----------------------------------------------

    def _encode_datasets(self, force_overwrite):
        log.info(f"=> Encoding datasets...")
        for ds in self.ds_list:
            if is_no_model(ds.subword_model):
                continue

            make_dir([ds.get_encoded_path()])
            log.info(f"\t- Encoding dataset: {ds.variant_id(as_path=True)}")
            for fname in ds.get_split_fnames():
                lang = fname.split('.')[-1]
                input_file = (ds.get_pretok_path(fname) if ds.pretok_flag
                              else ds.get_splits_auto_path(fname))
                model_path = (ds.get_vocab_file() if self.merge_vocabs
                              else ds.get_vocab_file(lang=lang)) + ".model"
                encode_file(
                    input_file=input_file,
                    output_file=ds.get_encoded_path(fname),
                    model_vocab_path=model_path,
                    subword_model=ds.subword_model,
                    force_overwrite=force_overwrite,
                )

    # --- Stage: vocab frequencies -------------------------------------

    def _export_vocab_frequencies(self, force_overwrite, normalize_freq=False):
        """Important: .vocabf is for plotting/inspection only."""
        for ds in self.ds_list:
            if is_no_model(ds.subword_model):
                continue
            log.info(f"\t- Exporting frequency vocab: {ds.variant_id(as_path=True)}")
            vocab_builder.export_frequencies(
                ds, force_overwrite=force_overwrite, normalize_freq=normalize_freq,
            )

    # --- Stage: stats --------------------------------------------------

    def _compute_stats(self, force_overwrite, print_stats=True):
        import json
        log.info(f"=> Computing stats... (base_path={self.base_path})")

        for ds in self.ds_list:
            log.info(f"\t- Computing stats for dataset: {ds.variant_id(as_path=True)}")
            make_dir(ds.get_stats_path())

            savepath = ds.get_stats_path("stats.json")
            if not force_overwrite and os.path.exists(savepath):
                continue
            stats = ds.get_stats(count_unknowns=True)
            save_json(stats, savepath=savepath)
            if print_stats:
                log.info(json.dumps(stats, indent=4))


def merge_datasets(builder: DatasetBuilder, name, language_pair="xx-yy",
                   dataset_size_name="original", shuffle_lines=False,
                   use_preprocessed_splits=False, preprocess_fn=None,
                   force_overwrite=False):
    """Concatenate every train_ds inside ``builder`` into one synthetic dataset.

    Used to assemble a multi-corpus baseline (e.g. all Europarl pairs into one
    file). Lives outside ``DatasetBuilder`` because it only depends on the
    builder's accessors, not its internal state.
    """
    log.info(f"=> Merging datasets... (base_path={builder.base_path})")

    ds = Dataset(
        base_path=builder.base_path, parent_ds=None,
        dataset_name=name, dataset_lang_pair=language_pair,
        dataset_size_name=dataset_size_name, dataset_lines=None,
        splits_sizes=builder.DEFAULT_SPLIT_SIZES,
        subword_model=None, vocab_size=None, merge_vocabs=None,
    )

    if not force_overwrite and all(os.path.exists(ds.get_split_path(f))
                                   for f in ds.get_split_fnames()):
        log.info(f"\t=> Merged dataset already exist for '{ds.base_id(as_path=True)}'")
        return

    src_train, tgt_train = [], []
    src_val, tgt_val = [], []
    src_test, tgt_test = [], []

    for ds_i in builder.get_train_ds():
        log.info(f"\t- Reading dataset: {ds_i.variant_id(as_path=True)}")
        path_fn = (ds_i.get_splits_preprocessed_path if use_preprocessed_splits
                   else ds_i.get_split_path)

        def _read(split, lang):
            return read_file_lines(path_fn(fname=f"{split}.{lang}"), autoclean=False)

        src_tr, tgt_tr = _read(ds_i.train_name, ds_i.src_lang), _read(ds_i.train_name, ds_i.tgt_lang)
        src_vl, tgt_vl = _read(ds_i.val_name, ds_i.src_lang),   _read(ds_i.val_name, ds_i.tgt_lang)
        src_ts, tgt_ts = _read(ds_i.test_name, ds_i.src_lang),  _read(ds_i.test_name, ds_i.tgt_lang)
        assert len(src_tr) == len(tgt_tr)
        assert len(src_vl) == len(tgt_vl)
        assert len(src_ts) == len(tgt_ts)

        if preprocess_fn:
            src_tr, tgt_tr = preprocess_fn(x=src_tr, y=tgt_tr, ds=ds_i)
            src_vl, tgt_vl = preprocess_fn(x=src_vl, y=tgt_vl, ds=ds_i)
            src_ts, tgt_ts = preprocess_fn(x=src_ts, y=tgt_ts, ds=ds_i)

        src_train += src_tr; tgt_train += tgt_tr
        src_val += src_vl;   tgt_val += tgt_vl
        src_test += src_ts;  tgt_test += tgt_ts

    if shuffle_lines:
        log.info(f"\t- Shuffling lines...")
        src_train, tgt_train = shuffle_in_order(src_train, tgt_train)
        src_val, tgt_val = shuffle_in_order(src_val, tgt_val)
        src_test, tgt_test = shuffle_in_order(src_test, tgt_test)

    make_dir(ds.get_split_path())
    for src_lines, tgt_lines, fname in (
        (src_train, tgt_train, ds.train_name),
        (src_val, tgt_val, ds.val_name),
        (src_test, tgt_test, ds.test_name),
    ):
        write_file_lines(src_lines, ds.get_split_path(f"{fname}.{ds.src_lang}"))
        write_file_lines(tgt_lines, ds.get_split_path(f"{fname}.{ds.tgt_lang}"))
        log.info(f"\t\t- Partitions saved: {fname}.{ds.src_lang} and {fname}.{ds.tgt_lang}")
