import os.path
import shutil
from collections import Counter
from itertools import islice

import numpy as np
import pandas as pd

from autonmt.preprocessing import tokenizers

from autonmt.bundle import utils, plots
from autonmt.bundle.utils import *
from autonmt.preprocessing.dataset import Dataset
from autonmt.preprocessing.processors import pretokenize_file, encode_file


class DatasetBuilder:

    def __init__(self, base_path, datasets, encoding=None, merge_vocabs=False, eval_mode="same",
                 preprocess_raw_fn=None, preprocess_splits_fn=None, preprocess_predict_fn=None):
        self.base_path = base_path
        self.datasets = datasets
        self.encoding = encoding
        self.merge_vocabs = merge_vocabs
        self.eval_mode = eval_mode

        # Set processing functions
        self.preprocess_raw_fn = preprocess_raw_fn  # Process function for the raw files
        self.preprocess_splits_fn = preprocess_splits_fn  # Process function for the splits files
        self.preprocess_predict_fn = preprocess_predict_fn  # Process function for the prediction files

        # Constants
        self.ref_size_name = "original"
        self.default_split_sizes = (None, 1000, 1000)

        # Tokenizer
        self.input_sentence_size = 1000000
        self.character_coverage = 1.0
        self.truncate_at = 1024

        # Other
        self.ds_refs = {str(ds): ds for ds in self._unroll_datasets(encodings=None, parent_ds=True, ref_size_only=True)}  # Reference datasets (must exist, but might not appear in the user code)
        self.ds_list_parents = self._unroll_datasets(encodings=None, parent_ds=True) # main preprocessing only
        self.ds_list = []
        if encoding:  # Add datasets with subwords, vocabs,...
            self.ds_list = self._unroll_datasets(encodings=self._unroll_encoding(encoding), parent_ds=False)

        # Checks
        self.checks()

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.ds_list):
            result = self.ds_list[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    def __len__(self):
        return len(self.ds_list)

    def checks(self):
        for d in self.datasets:
            # Language pair format
            for lang_pair in d.get("languages"):
                if len(lang_pair) != 5 or lang_pair[2] != '-':
                    raise ValueError("Language pairs must be defined with this format: 'xx-yy'")

    def _unroll_encoding(self, encoding):
        valid_enc = []
        keys = set()
        for enc in encoding:
            for model in enc.get("subword_models"):
                for size in enc.get("vocab_sizes"):
                    key = f"{model}_{size}"
                    if key not in keys:
                        valid_enc.append({"subword_model": model, "vocab_size": size})
                        keys.add(key)
                    else:
                        print(f"[WARNING]: Ignoring repeated entry in encoding (subword={model}; size={size})")
        return valid_enc

    def _unroll_datasets(self, encodings=None, parent_ds=None, ref_size_only=False):
        ds_unrolled = []

        # LEVEL 0: Dataset entries
        for ds_entry in self.datasets:
            ds_name = ds_entry["name"]
            ds_splits_sizes = ds_entry.get("split_sizes", self.default_split_sizes)

            # LEVEL 1: Dataset languages
            for ds_lang_pair in ds_entry["languages"]:

                # LEVEL 2: Dataset sizes
                sizes = [(self.ref_size_name, None)] if ref_size_only else ds_entry["sizes"]
                for ds_size_name, ds_lines in sizes:
                    ds_params = dict(base_path=self.base_path,
                                     dataset_name=ds_name, dataset_lang_pair=ds_lang_pair,
                                     dataset_size_name=ds_size_name, dataset_lines=ds_lines,
                                     merge_vocabs=self.merge_vocabs, eval_mode=self.eval_mode,
                                     preprocess_raw_fn=self.preprocess_raw_fn,
                                     preprocess_splits_fn=self.preprocess_splits_fn,
                                     preprocess_predict_fn=self.preprocess_predict_fn)

                    # Add encodings
                    ds_encs = encodings if encodings else [{"subword_model": None, "vocab_size": None}]
                    for ds_enc in ds_encs:
                        ds = Dataset(parent_ds=parent_ds, splits_sizes=ds_splits_sizes, **ds_params, **ds_enc)
                        ds_unrolled.append(ds)

        return ds_unrolled

    def get_ds(self, ignore_variants=False):
        if self.ds_list:
            return self.ds_list_parents if ignore_variants else self.ds_list
        else:
            return self.ds_list_parents

    def get_train_ds(self):
        return self.get_ds(ignore_variants=False)

    def get_test_ds(self):
        return self.get_ds(ignore_variants=True)

    def build(self, make_plots=False, force_overwrite=False):
        print(f"=> Building datasets...")
        print(f"\t- base_path={self.base_path}")

        # Check directories, creates the ones missing, and check if the data is valid
        self._check_dir_structure(skip_file_checks=True, force_overwrite=force_overwrite, interactive=False)

        # Preprocess raw files, create partitions, and process splits
        self._preprocess_raw_files(force_overwrite=force_overwrite)
        self._create_partitions(use_ref_partitions=True, force_overwrite=force_overwrite)
        self._create_reduced_versions(force_overwrite=force_overwrite)
        self._preprocess_split_files(force_overwrite=force_overwrite)

        # Ignore further preprocessing if there is not encoding
        if not self.encoding:
            print("\t- [WARNING]: No encoding was specified")
        else:
            # Train model (applies pre-tokenization if needed)
            self._train_tokenizer(force_overwrite=force_overwrite)

            # Encode preprocessing
            self._encode_datasets(force_overwrite=force_overwrite)
            self._export_vocab_frequencies(force_overwrite=force_overwrite)

            # Compute stats
            self._compute_stats(force_overwrite=force_overwrite)

            # Make plot
            if make_plots:
                self._plot_datasets(force_overwrite=force_overwrite)

        return self

    def _check_dir_structure(self, skip_file_checks, force_overwrite, interactive):
        print(f"=> Checking directory structure...")
        invalid_structure = False

        # Each reference ds must contain either a raw folder or a partitions folder
        for ds in self.ds_refs.values():  # Datasets
            print(f"\t=> Checking dataset: '{ds.id(as_path=True)}'")

            # Truth table
            raw, raw_files_path = ds.has_raw_files(verbose=True)
            splits, split_files_path = ds.has_split_files()
            force = force_overwrite

            # Note: splits data has priority over raw data
            # Check if split data exists, and if it is valid
            if (splits and not raw) or (splits and raw and not force):  # Use split data
                ds.source_data = "splits"

                # Check that each file has the same number of lines
                if not skip_file_checks:
                    print("\t\t- Checking that each split pair contains the same number of lines...")
                    if count_file_lines(split_files_path[0]) != count_file_lines(split_files_path[1]) or \
                            count_file_lines(split_files_path[2]) != count_file_lines(split_files_path[3]) or \
                            count_file_lines(split_files_path[4]) != count_file_lines(split_files_path[5]):
                        print("\t\t- [Invalid data]: We found the 'raw' folder, but the source and target files do not "
                                "have the same number of lines.")
                        invalid_structure = True
                        continue

            # Check if raw data exists, and if it is valid
            elif (not splits and raw) or (splits and raw and force):  # Use raw data  (force_overwrite=True or False, as long as there is raw data)
                ds.source_data = "raw"

                # Check that each file has the same number of lines
                if not skip_file_checks:
                    print("\t\t- Checking that each language pair contains the same number of lines...")
                    if count_file_lines(raw_files_path[0]) != count_file_lines(raw_files_path[1]):
                        print("\t\t- [Invalid data]: We found the 'raw' folder, but the source and target files do not "
                              "have the same number of lines.")
                        invalid_structure = True
                        continue

            # Splits or raw folder either missing or with invalid contents
            elif not splits and not raw:  # Create directories
                ds.source_data = None
                invalid_structure = True
                missing_folders = False

                # Problems with the raw folders
                if not os.path.exists(ds.get_raw_path()):
                    missing_folders = True
                    print(f"\t\t=> [Missing folder]: We could not find the 'raw' folder at: {ds.get_path()}")
                else:
                    print(f"\t\t=> [Invalid data]: The files in the 'raw' data were invalid: {ds.get_path()}")

                # Problems with the splits folders
                if not os.path.exists(ds.get_split_path()):
                    missing_folders = True
                    print(f"\t\t=> [Missing folder]: We could not find the 'splits' folder at: {ds.get_path()}")
                else:
                    print(f"\t\t=> [Invalid data]: The files in the 'splits' data were invalid: {ds.get_path()}")

                # Create missing folders
                if missing_folders:
                    res = ask_yes_or_no(question="Do you want to create the missing directories?", interactive=interactive)
                    if res:
                        print(f"\t\t\t- Creating missing directories...")
                        make_dir(ds.get_raw_path()) if res else None
                        make_dir(ds.get_split_path()) if res else None

                # Summary
                print(f"\t\t=> [ACTION REQUIRED] You need to add a valid dataset to at least one of these folders:")
                print(
                    f"\t\t\t- '{ds.data_raw_path}': Used to create partitions from a parallel corpus (e.g. 'data.{ds.src_lang}' and 'data.{ds.trg_lang}')")
                print(
                    f"\t\t\t- '{ds.data_splits_path}': Used when the partitions are available (e.g. '[train,val,test].[{ds.src_lang},{ds.trg_lang}]'")

        # Stop program if there are files missing
        if invalid_structure:
            print("=> [ERROR]: Closing program due to missing files")
            print("*** Restart the program when these files are added in their respective folders ***")
            exit(0)

    def _preprocess_files(self, ds, input_sets, output_sets, preprocess_fn, force_overwrite):
        # Preprocess files pairs
        for i, (input_paths, output_paths) in enumerate(zip(input_sets, output_sets), 1):
            (src_in, trg_in), (src_out, trg_out) = input_paths, output_paths
            # Preprocessed files (if needed)
            if force_overwrite or not all(os.path.exists(f) for f in [src_out, trg_out]):
                print(f"\t=> Preprocessing file-pair ({i}/{len(input_sets)}) dataset '{ds.id(as_path=True)}'")
                # Read lines (+minor cleaning)
                src_lines = read_file_lines(ds.get_split_path(src_in), autoclean=True)
                tgt_lines = read_file_lines(ds.get_split_path(trg_in), autoclean=True)

                # Check that each file pair has the same number of lines
                if len(src_lines) != len(tgt_lines):
                    print(f"=> [ERROR]: The source and target files do not have the same number of lines "
                          f"({len(src_lines)} != {len(tgt_lines)})")
                    print(f"\t- Source file: {ds.get_split_path(src_in)}")
                    print(f"\t- Target file: {ds.get_split_path(src_out)}")

                # Preprocess lines
                src_lines, tgt_lines = preprocess_fn(src_lines, tgt_lines)

                # Write lines (do not overwrite original files)
                write_file_lines(src_lines, filename=f"{ds.get_splits_preprocessed_path(src_out)}", insert_break_line=True)
                write_file_lines(tgt_lines, filename=f"{ds.get_splits_preprocessed_path(trg_out)}", insert_break_line=True)
    def _preprocess_raw_files(self, force_overwrite):
        # Note: If raw_preprocess exists, but it is not preprocessed, the flag won't be updated
        if self.preprocess_raw_fn is None:
            return

        print(f"=> Preprocessing raw files...")
        for ds_key, ds in self.ds_refs.items():  # Dataset
            # Update source data
            if self.ds_refs[ds_key].source_data == "raw":  # Do not preprocess if we are going to use splits or other
                self.ds_refs[ds_key].source_data = "raw_preprocessed"

                # Create directory
                make_dir(ds.get_raw_preprocessed_path())

                # File pairs
                input_paths = [ds.get_raw_path(f) for f in ds.get_raw_files()]
                output_paths = [ds.get_raw_preprocessed_path(f) for f in ds.get_raw_preprocessed_files()]

                # data.xx, data.yy
                input_sets = (input_paths[0:2],)
                output_sets = (output_paths[0:2],)

                # Preprocess files
                self._preprocess_files(ds, input_sets, output_sets, self.preprocess_raw_fn, force_overwrite)

    def _preprocess_split_files(self, force_overwrite):
        print(f"=> Preprocessing split files...")

        for ds in self.ds_list_parents:  # Dataset
            # Create directory
            make_dir(ds.get_splits_preprocessed_path())

            # File pairs
            input_paths = [ds.get_split_path(f) for f in ds.get_split_files()]
            output_paths = [ds.get_splits_preprocessed_path(f) for f in ds.get_split_files()]

            # train.xx, val.xx, test.xx
            input_sets = (input_paths[0:2], input_paths[2:4], input_paths[4:6])
            output_sets = (output_paths[0:2], output_paths[2:4], output_paths[4:6])

            # Preprocess files
            self._preprocess_files(ds, input_sets, output_sets, self.preprocess_splits_fn, force_overwrite)


    def _create_partitions(self, use_ref_partitions, force_overwrite):
        print(f"=> Creating partitions...")

        # Select reference datasets
        if use_ref_partitions:
            datasets = self.ds_refs.values()
        else:  # Use original partitions when the dataset is a reduced version of the original
            datasets = self.ds_list_parents

        # Create partitions for each dataset
        for ds in datasets:
            # Check if partitions already exist
            if all([os.path.exists(ds.get_split_path(f)) for f in ds.get_split_files()]) and not force_overwrite:
                print(f"\t=> Partitions already exist for '{ds.id(as_path=True)}'")
                continue

            # Create partitions (overwrite all partitions even if only one  is missing - due to the randomization)
            print(f"\t=> Creating dataset partitions for '{ds.id(as_path=True)}'")
            if ds.source_data in {"raw", "raw_preprocessed"}:  # Use raw data  (force_overwrite=True or False, as long as there is raw data)
                # Get raw/raw_preprocessed files
                if ds.source_data == "raw":
                    src_path, trg_path = [ds.get_raw_path(f) for f in ds.get_raw_files()]
                elif ds.source_data == "raw_preprocessed":
                    src_path, trg_path = [ds.get_raw_preprocessed_path(f) for f in ds.get_raw_preprocessed_files()]
                else:
                    raise ValueError(f"Invalid value for source data: {ds.source_data}")
                assert os.path.isfile(src_path) and os.path.isfile(trg_path)

                # Read lines, clean and shuffle
                print(f"\t=> Processing from '{ds.source_data.replace('_', ' ')}' files...")
                lines = [(src, trg) for src, trg in zip(read_file_lines(src_path), read_file_lines(trg_path))]
                random.shuffle(lines)  # again...

                # Parse split sizes
                train_size, val_size, test_size = ds.splits_sizes
                val_size = utils.parse_split_size(val_size, max_ds_size=len(lines))
                test_size = utils.parse_split_size(test_size, max_ds_size=len(lines))
                if (val_size + test_size) > len(lines):
                    raise ValueError(f"The validation and test sets exceed the size of the dataset")

                # Create partitions
                train_lines = lines[:-(val_size + test_size)]
                val_lines = lines[-(val_size + test_size):-test_size]
                test_lines = lines[-test_size:]

                # Create split folder
                utils.make_dir(ds.get_split_path())

                # Save partitions
                _splits = [(val_lines, ds.val_name), (test_lines, ds.test_name), (train_lines, ds.train_name)]
                for split_lines, split_name in _splits:
                    for i, lang in enumerate([ds.src_lang, ds.trg_lang]):  # Languages
                        # Save partition
                        savepath = ds.get_split_path(f"{split_name}.{lang}")
                        utils.write_file_lines(list(zip(*split_lines))[i], savepath)
                        print(f"\t\t- Partition saved: {split_name}.{lang}")
            else:
                pass  # Use existing split data

    def _create_reduced_versions(self, force_overwrite):
        print("=> Creating reduced versions...")

        # Create reduce splits
        for ds in self.ds_list_parents:  # Dataset
            # Ignore if this is the reference dataset
            ds_ref = ds.id()[0], ds.id()[1], self.ref_size_name
            if ds.id()[2] == self.ref_size_name:
                continue

            # Create new splits folder *****
            make_dir(ds.get_split_path())

            # Add truncated splits
            print(f"\t=> Creating reduced version: {ds.id(as_path=True)}")
            for fname in ds.get_split_files():
                ref_filename = os.path.join(self.base_path, *ds_ref, ds.data_splits_path, fname)
                new_filename = ds.get_split_path(fname)

                # Copy partitions
                if force_overwrite or not os.path.exists(new_filename):
                    print(f"\t\t- Creating split file: {fname}...")

                    if fname.split('.')[0] == ds.train_name:  # train.xx
                        with open(ref_filename, 'rb') as fin:
                            lines = list(islice(fin, ds.dataset_lines))  # Copy n lines efficiently
                            if len(lines) == ds.dataset_lines:
                                with open(new_filename, 'wb') as fout:
                                    fout.writelines(lines)
                            else:
                                raise ValueError(f"[REDUCING FILES]: Not enough lines ({ len(lines)} < {ds.dataset_lines}) in the training set: {ref_filename}")
                    else:  # val.xx, test.xx
                        # Copy val/test files from "original" (split_size is not enforced for split files)
                        shutil.copy(ref_filename, new_filename)


    def _pretokenize(self, ds, force_overwrite):
        # Check if this needs pretokenization
        if not ds.pretok_flag:
            return

        # Ignore dataset
        if ds.subword_model in {None, "none", "bytes"}:
            return

        # Create paths
        pretokenize_path = ds.get_pretok_path()
        make_dir([pretokenize_path])

        print(f"\t- Pretokenizing splits: {ds.id(as_path=True)}")
        for fname in ds.get_split_files():
            print(f"\t\t- Pretokenizing split file: {fname}...")
            lang = fname.split(".")[1]
            input_file = ds.get_splits_preprocessed_path(fname)
            output_file = ds.get_pretok_path(fname)

            # Pretokenize
            pretokenize_file(input_file=input_file, output_file=output_file, lang=lang,
                             force_overwrite=force_overwrite)

    def _train_tokenizer(self, force_overwrite):
        print(f"=> Building vocabularies...")

        for ds in self:  # Dataset
            src_lang, trg_lang = ds.id()[1].split("-")

            # Create paths
            vocab_path = ds.get_vocab_path()
            tmp_path = os.path.join(ds.get_vocab_path(base=True), "_tmp")
            pretokenize_path = ds.get_pretok_path()
            make_dir([vocab_path, tmp_path, pretokenize_path])
            print(f"\t- Building vocabulary: {ds.id2(as_path=True)}")

            # Ignore dataset but create directories (just in case... for plots or stats)
            if ds.subword_model in {None, "none", "bytes"}:
                continue

            # Pretokenize (if needed - words)
            self._pretokenize(ds, force_overwrite)

            # Get train files
            file_path_fn = ds.get_pretok_path if ds.pretok_flag else ds.get_splits_preprocessed_path
            src_train_path = file_path_fn(fname=f"{ds.train_name}.{src_lang}")
            trg_train_path = file_path_fn(fname=f"{ds.train_name}.{trg_lang}")

            # One or two models
            if self.merge_vocabs:  # One model
                concat_train_path = os.path.join(tmp_path, f"{ds.train_name}.{src_lang}-{trg_lang}")

                # Concat files
                if force_overwrite or not os.path.exists(concat_train_path):
                    # Read files
                    lines = read_file_lines(src_train_path, autoclean=True)
                    lines += read_file_lines(trg_train_path, autoclean=True)

                    # Shuffle lines: Just in case because can spm_train load the first X lines of corpus by default
                    random.shuffle(lines)

                    # Save file
                    write_file_lines(lines=lines, filename=concat_train_path, insert_break_line=True)
                files = [(concat_train_path, f"{src_lang}-{trg_lang}")]
            else:  # Two models
                files = [(src_train_path, f"{src_lang}"), (trg_train_path, f"{trg_lang}")]

            # Train models
            for input_file, ext in files:
                output_file = ds.get_vocab_file(lang=ext)  # without extension
                if force_overwrite or not os.path.exists(f"{output_file}.model"):
                    tokenizers.spm_train(input_file=input_file, model_prefix=output_file, subword_model=ds.subword_model,
                                         vocab_size=ds.vocab_size, input_sentence_size=self.input_sentence_size,
                                         character_coverage=self.character_coverage)
                    assert os.path.exists(f"{output_file}.model")

    def _encode_datasets(self, force_overwrite):
        print(f"=> Building datasets...")
        for ds in self:  # Dataset
            # Ignore dataset
            if ds.subword_model in {None, "none"}:
                continue

            # Create paths
            encoded_path = ds.get_encoded_path()
            make_dir([encoded_path])

            # Encode files
            print(f"\t- Encoding dataset: {ds.id2(as_path=True)}")
            for fname in ds.get_split_files():
                lang = fname.split('.')[-1]
                data_path = ds.data_pretokenized_path if ds.pretok_flag else ds.data_splits_preprocessed_path
                input_file = os.path.join(ds.base_path, *ds.id(), data_path, fname)
                output_file = ds.get_encoded_path(fname)

                # Encode file
                encode_file(ds=ds, input_file=input_file, output_file=output_file,
                            lang=lang, merge_vocabs=self.merge_vocabs, truncate_at=self.truncate_at,
                            force_overwrite=force_overwrite)

    def _export_vocab_frequencies(self, force_overwrite, normalize_freq=False):
        """
        Important: .vocabf should be used only for plotting
        """
        for ds in self:  # Dataset
            src_lang, trg_lang = ds.id()[1].split("-")
            spm_model = False

            # Select split function
            if ds.subword_model in {None, "none"}:
                continue
            elif ds.subword_model in {"bytes"}:
                split_fn = lambda x: x.split(' ')
            else:
                split_fn = lambda x: x.split(' ')
                spm_model = True

            # Get langs
            if self.merge_vocabs:
                lang_files = [f"{src_lang}-{trg_lang}"]
            else:
                lang_files = [src_lang, trg_lang]

            # Check if file/files exists
            print(f"\t- Exporting frequency vocab: {ds.id2(as_path=True)}")
            vocab_files = [ds.get_vocab_path(fname=f)+".vocabf" for f in lang_files]
            if force_overwrite or not all([os.path.exists(f) for f in vocab_files]):
                # Get train paths
                src_train_path = ds.get_encoded_path(f"{ds.train_name}.{src_lang}")
                trg_train_path = ds.get_encoded_path(f"{ds.train_name}.{trg_lang}")

                #  Convert to counters
                src_vocabf = build_counter_low_mem(src_train_path, split_fn=split_fn)
                trg_vocabf = build_counter_low_mem(trg_train_path, split_fn=split_fn)

                if not spm_model:
                    vocabs = [src_vocabf + trg_vocabf] if self.merge_vocabs else [src_vocabf, trg_vocabf]

                else:
                    if self.merge_vocabs:
                        vocabf_lang = [(src_vocabf+trg_vocabf, f"{src_lang}-{trg_lang}")]
                    else:
                        vocabf_lang = [(src_vocabf, src_lang), (trg_vocabf, trg_lang)]

                    # Count tokens
                    vocabs = []
                    for vocabf, lang_file in vocabf_lang:
                        # Get the exact vocab from SPM
                        spm_vocab_lines = read_file_lines(ds.get_vocab_path(fname=lang_file) + ".vocab", autoclean=False)
                        spm_vocab_lines = spm_vocab_lines[4:]  # Remove special tokens
                        spm_vocab = {l.split('\t')[0]: 0 for l in spm_vocab_lines}

                        # Important: SPM might end with words that won't use during the encoding (of the training)
                        # Only count tokens that exists in the vocabulary
                        c = Counter({k: v for k, v in vocabf.items() if k in spm_vocab})
                        vocabs.append(c)

                # Save vocabs
                for vocab, vocab_path in zip(vocabs, vocab_files):
                    # Normalize (if requested)
                    if normalize_freq:
                        vocab = utils.norm_counter(vocab)

                    # Sort frequencies
                    vocab_frequencies = vocab.most_common()

                    # Save vocab
                    if force_overwrite or not os.path.exists(vocab_path):
                        lines = [f"{pair[0]}\t{pair[1]}" for pair in vocab_frequencies]
                        write_file_lines(lines=lines, filename=vocab_path, insert_break_line=True)

    def _compute_stats(self, force_overwrite):
        print(f"=> Computing stats... (base_path={self.base_path})")

        # Walk through preprocessing
        for ds in self:
            print(f"\t- Computing stats for dataset: {ds.id2(as_path=True)}")

            # Get path
            make_dir(ds.get_stats_path())

            # Save file
            savepath = ds.get_stats_path("stats.json")
            if force_overwrite or not os.path.exists(savepath):
                # Compute and save stats
                stats = ds.get_stats(count_unknowns=True)
                save_json(stats, savepath=savepath)

    def _plot_datasets(self, force_overwrite, save_figures=True, show_figures=False, add_dataset_title=True, vocab_top_k=None):
        print(f"=> Plotting started... (base_path={self.base_path})")
        print(f"- [WARNING]: Matplotlib might miss some images if the loop is too fast")

        # Set default vars
        if vocab_top_k is None:
            vocab_top_k = [50]

        # Set backend
        if save_figures:
            plots.set_non_gui_backend()
            if show_figures:
                raise ValueError("'save_fig' is incompatible with 'show_fig'")

        # Walk through preprocessing
        for ds in self:  # Dataset
            ds_name, lang_pair, ds_size_name = ds.id()
            src_lang, trg_lang = ds.id()[1].split("-")

            # Set base path
            ds_title = f"{ds_name.title()} ({lang_pair}; {ds.subword_model}; {ds.vocab_size})"
            vocab_name = f"_{ds.vocab_size}" if ds.vocab_size else ""
            suffix_fname = f"{ds_name}_{ds_size_name}_{lang_pair}__{ds.subword_model}{vocab_name}".lower().replace('/', '_')
            print(f"\t- Creating plots for: {ds.id2(as_path=True)}")

            # Set paths and create dirs
            vocab_path = ds.get_vocab_path()
            encoded_path = ds.get_encoded_path()
            plots_encoded_path = ds.get_plots_path()
            make_dir(plots_encoded_path)

            print(f"\t\t- Creating 'Sentence length distribution' plots...")
            split_stats = {}
            for fname in ds.get_split_files():
                split_name, split_lang = fname.split('.')

                # Ignore dataset
                tokens_per_sentence = utils.count_tokens_per_sentence(filename=ds.get_encoded_path(fname))
                tokens_per_sentence = np.array(tokens_per_sentence)

                # Compute data
                row = {
                    "total_sentences": len(tokens_per_sentence),
                    "total_tokens": int(tokens_per_sentence.sum()),
                    "max_tokens": int(np.max(tokens_per_sentence)),
                    "min_tokens": int(np.min(tokens_per_sentence)),
                    "avg_tokens": float(np.average(tokens_per_sentence)),
                    "std_tokens": float(np.std(tokens_per_sentence)),
                    "percentile5_tokens": int(np.percentile(tokens_per_sentence, 5)),
                    "percentile50_tokens": int(np.percentile(tokens_per_sentence, 50)),
                    "percentile95_tokens": int(np.percentile(tokens_per_sentence, 95)),
                    "split": split_name,
                    "lang": split_lang,
                }
                split_stats[fname] = row

                # Plot sentence length distribution (by tokens' length)
                df = pd.DataFrame(tokens_per_sentence, columns=["frequency"])
                title = f"Sentence length distribution ({split_name.title()} - {split_lang})"
                title = title if not add_dataset_title else f"{ds_title}:\n{title}"
                p_fname = f"sent_distr_{split_name}_{split_lang}__{suffix_fname}".lower()
                plots.histogram(data=df, x="frequency", output_dir=plots_encoded_path, fname=p_fname,
                                title=title, xlabel="Tokens per sentence", ylabel="Frequency", bins=100,
                                aspect_ratio=(6, 4), size=1.5, save_fig=save_figures, show_fig=show_figures,
                                overwrite=force_overwrite)

            # Create dataframe
            df = pd.DataFrame(list(split_stats.values()))

            # Save data
            utils.save_json(split_stats, savepath=os.path.join(plots_encoded_path, f"stats__{suffix_fname}.json"))
            # df.to_csv(os.path.join(plots_encoded_path, f"stats__{base_fname}.csv"), index=False)

            # Plot split size (by the number o f sentences)
            print(f"\t\t- Creating 'Split sizes' plots...")
            title = f"Split sizes (by number of sentences)"
            title = title if not add_dataset_title else f"{ds_title}:\n{title}"
            p_fname = f"split_size_sent__{suffix_fname}".lower()
            plots.catplot(data=df, x="split", y="total_sentences", hue="lang",
                          title=title, xlabel="Dataset partitions", ylabel="Num. of sentences",
                          leyend_title=None,
                          output_dir=plots_encoded_path, fname=p_fname, aspect_ratio=(6, 4), size=1.0,
                          save_fig=save_figures, show_fig=show_figures, overwrite=force_overwrite)

            # Plot split size (by token number)
            if ds.subword_model not in {None, "none"}:
                title = f"Split sizes (by number of tokens - {ds.subword_model.title()})"
                title = title if not add_dataset_title else f"{ds_title}:\n{title}"
                p_fname = f"split_size_tok__{suffix_fname}".lower()
                plots.catplot(data=df, x="split", y="total_tokens", hue="lang",
                              title=title, xlabel="Dataset partitions", ylabel="Num. of tokens", leyend_title=None,
                              output_dir=plots_encoded_path, fname=p_fname, aspect_ratio=(6, 4), size=1.0,
                              save_fig=save_figures, show_fig=show_figures, overwrite=force_overwrite)

            # Plot vocabulary frequency
            if ds.subword_model not in {None, "none"}:
                print(f"\t\t- Creating 'Vocabulary distribution' plots...")

                # Load vocabulary
                if self.merge_vocabs:
                    lang_files = [f"{src_lang}-{trg_lang}"]
                else:
                    lang_files = [src_lang, trg_lang]

                # Read vocabs
                for lang_file in lang_files:
                    vocab_freq_path = os.path.join(vocab_path, lang_file + ".vocabf")
                    with open(vocab_freq_path, 'r') as f:
                        rows = [line.split('\t') for line in f.readlines()]
                        df = pd.DataFrame(rows, columns=["token", "frequency"])
                        df["frequency"] = df["frequency"].apply(lambda x: int(x.strip())).astype(int)
                        df = df.sort_values(by='frequency', ascending=False, na_position='last')

                    for top_k in vocab_top_k:
                        title = f"Vocabulary distribution (top {str(top_k)} {ds.subword_model.title()}; {lang_file})"
                        title = title if not add_dataset_title else f"{ds_title}:\n{title}"
                        p_fname = f"vocab_distr_{lang_file}_top{str(top_k)}__{suffix_fname}".lower()
                        plots.barplot(data=df.head(top_k), x="token", y="frequency",
                                      output_dir=plots_encoded_path, fname=p_fname,
                                      title=title, xlabel="Token frequency", ylabel="Frequency",
                                      aspect_ratio=(6, 4), size=1.25, save_fig=save_figures, show_fig=show_figures,
                                      overwrite=force_overwrite)
