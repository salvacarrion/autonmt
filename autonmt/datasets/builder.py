import random
import shutil
from itertools import islice

import numpy as np
import pandas as pd
import tqdm

from autonmt.utils import *
from autonmt import plots, utils
from autonmt.datasets.dataset import Dataset
from collections import Counter

from autonmt import py_cmd_api


def pretokenize_file(input_file, output_file, lang, force_overwrite, **kwargs):
    # Tokenize
    if force_overwrite or not os.path.exists(output_file):
        py_cmd_api.moses_tokenizer(input_file=input_file, output_file=output_file, lang=lang, **kwargs)


def encode_file(ds, input_file, output_file, output_file_pretok, lang, merge_vocabs, force_overwrite, **kwargs):
    if force_overwrite or not os.path.exists(output_file):
        # Copy file
        if ds.subword_model in {None, "none"}:
            shutil.copyfile(input_file, output_file)

        elif ds.subword_model in {"bytes"}:
            # Save file as UTF8 and make sure everything uses NFKC
            lines = read_file_lines(input_file)
            lines = [preprocess_text(line, normalization="NFKC") for line in lines]
            lines = [" ".join([hex(x) for x in line.encode()]) for line in lines] if ds.bytes_as_words else lines
            write_file_lines(lines=lines, filename=output_file, encoding="utf8")

        else:
            # Pretokenize file if needed (used during the translation code)
            if output_file_pretok and ds.subword_model in {"word"}:
                pretokenize_file(input_file=input_file, output_file=output_file_pretok, lang=lang,
                                 force_overwrite=force_overwrite, **kwargs)
                input_file = output_file_pretok

            # Select model
            if merge_vocabs:
                model_path = ds.get_vocab_file() + ".model"
            else:
                model_path = ds.get_vocab_file(lang=lang) + ".model"

            # Encode files
            py_cmd_api.spm_encode(spm_model_path=model_path,
                                  input_file=input_file, output_file=output_file, **kwargs)


class DatasetBuilder:
    SUPPORTED_SUBWORD_MODELS = {"none", "word", "char", "char+bytes", "bpe", "unigram", "bytes"}

    def __init__(self, base_path, datasets, subword_models, vocab_sizes, merge_vocabs=True, bytes_as_words=False,
                 force_overwrite=False,
                 interactive=True, use_cmd=False, conda_env_name=None):
        self.base_path = base_path
        self.datasets = datasets
        self.subword_models = [x.strip().lower() for x in subword_models]
        self.vocab_sizes = vocab_sizes
        self.merge_vocabs = merge_vocabs
        self.bytes_as_words = bytes_as_words
        self.force_overwrite = force_overwrite
        self.interactive = interactive
        self.use_cmd = use_cmd
        self.conda_env_name = conda_env_name

        self.ref_size_name = "original"

        # Check subword models
        sw_diff = set(self.subword_models).difference(self.SUPPORTED_SUBWORD_MODELS)
        if sw_diff:
            raise ValueError(f"These subword models are not supported: {str(sw_diff)}")

        # Other
        self.ds_list = self._unroll_datasets(include_variants=True)  # includes subwords, vocabs,...
        self.ds_list_main = self._unroll_datasets(include_variants=False)  # main datasets only

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

    def _unroll_datasets(self, include_variants=True):
        ds_list_tmp = []

        # LEVEL 0: Dataset names
        for ds in self.datasets:  # Dataset
            # LEVEL 1: Languages
            for lang_pair in ds["languages"]:  # Languages
                # LEVEL 2: Sizes
                for ds_size_name, ds_max_lines in ds["sizes"]:  # Lengths
                    base_params = dict(base_path=self.base_path, dataset_name=ds["name"], dataset_lang_pair=lang_pair,
                                       dataset_size_name=ds_size_name, dataset_lines=ds_max_lines,
                                       merge_vocabs=self.merge_vocabs, bytes_as_words=self.bytes_as_words)

                    if include_variants:
                        for subword_model in self.subword_models:  # unigram, bpe, char, or word
                            # To avoid modifying the base params
                            params = dict(base_params)

                            # Set subword model config
                            if subword_model in {None, "none"}:
                                print(f"\t- [INFO]: Overriding vocabulary for none (None)")
                                vocab_sizes = [None]
                                params["encoded_path"] = os.path.join("data", "splits")
                            elif subword_model in {"bytes"}:
                                print(f"\t- [INFO]: Overriding vocabulary for bytes (256 + special tokens)")
                                vocab_sizes = [None]
                            else:
                                vocab_sizes = self.vocab_sizes

                            # Create Datasets
                            for vocab_size in vocab_sizes:
                                _ds = Dataset(subword_model=subword_model, vocab_size=vocab_size, parent_ds=False, **params)
                                ds_list_tmp.append(_ds)
                    else:
                        _ds = Dataset(subword_model=None, vocab_size=None, parent_ds=True, **base_params)
                        ds_list_tmp.append(_ds)
        return ds_list_tmp

    def iter_main(self):
        return self.ds_list_main

    def build(self, encode=True, val_size=(0.1, 5000), test_size=(0.1, 5000), shuffle=True, force_pretok=False,
              make_plots=False, safe=True):
        print(f"=> Building datasets...")
        print(f"\t- base_path={self.base_path}")

        # Create splits
        self._create_splits(val_size=val_size, test_size=test_size, shuffle=shuffle, safe=safe)

        # Create version for different sizes
        self._create_reduced_versions()

        # Pretokenize (if needed)
        self._pretokenize(force_pretok=force_pretok)

        # Build vocabs
        self._build_vocab(force_pretok=force_pretok)

        # Encode datasets
        if encode:
            self._encode_datasets()
            self._export_vocab_frequencies()

        # Make plot
        if make_plots:
            self._plot_datasets()

        return self

    def _get_ds_alias(self, ds_name, lang_pair, ds_size):
        return os.path.join(ds_name, lang_pair, ds_size)

    def _create_splits(self, val_size, test_size, shuffle, safe=True, safe_seconds=3):
        print("=> Creating splits...")
        if self.force_overwrite and safe:
            print(f"\t[INFO]: No splits will be overwritten despite the flag 'force_overwrite.\n"
                  f"\t        If you want to overwrite the splits, add the flag 'safe=False")

        # Create reduce splits
        for ds in self.iter_main():  # Dataset
            # Ignore if this is NOT a reference dataset
            if ds.id()[2] != self.ref_size_name:
                continue

            # Get language
            src_lang, trg_lang = ds.id()[1].split("-")

            # [dar data]: Check if the raw directory exists (...with all the data)
            raw_path = ds.get_raw_path()
            flag_raw_exists = os.path.exists(raw_path)
            flag_raw_files_okay = False
            raw_files = []
            if flag_raw_exists:  # Check if the raw files exists
                raw_files = [f for f in os.listdir(raw_path) if f[-2:] in {src_lang, trg_lang}]
                flag_raw_files_okay = len(raw_files) == 2

            # [split data]: Check if the splits directory exists
            splits_path = ds.get_split_path()
            split_files = [os.path.join(splits_path, fname) for fname in ds.get_split_files()]
            flag_splits_exists = os.path.exists(splits_path)
            flag_splits_files_okay = all([os.path.exists(p) for p in split_files])

            # If there is split data, ignore
            bypass = False
            if flag_splits_exists and flag_splits_files_okay:  # Splits okay => continue
                if self.force_overwrite and not safe:  # Overwrite?
                    bypass = True
                else:
                    continue

            # Create splits
            if flag_raw_exists and flag_raw_files_okay:  # Raw okay => Create splits partitions
                print(f"\t=> Creating splits from raw files: {self._get_ds_alias(*ds.id())}")
                if bypass:
                    print(f"\t\t[WARNING] Overwriting split files... (waiting {safe_seconds} seconds)")
                    time.sleep(safe_seconds)

                # Create splits folder
                make_dir(splits_path)

                # Read raw files
                src_lines, trg_lines = None, None
                for filename in raw_files:
                    with open(os.path.join(raw_path, filename), 'r') as f:
                        if filename[-2:].lower() == src_lang:  # Check extension
                            src_lines = f.readlines()
                        else:
                            trg_lines = f.readlines()

                # Clean lines
                lines = utils.preprocess_pairs(src_lines, trg_lines, shuffle=shuffle)

                # Parse split sizes
                val_size = utils.parse_split_size(val_size, max_ds_size=len(lines))
                test_size = utils.parse_split_size(test_size, max_ds_size=len(lines))
                if (val_size + test_size) > len(lines):
                    raise ValueError(f"The validation and test sets exceed the size of the dataset")

                # Create partitions
                train_lines = lines[:-(val_size + test_size)]
                val_lines = lines[-(val_size + test_size):-test_size]
                test_lines = lines[-test_size:]

                # Save partitions
                _splits = [(train_lines, ds.train_name), (val_lines, ds.val_name), (test_lines, ds.test_name)]
                for split_lines, split_name in _splits:
                    for i, lang in enumerate([src_lang, trg_lang]):  # Languages
                        savepath = os.path.join(splits_path, f"{split_name}.{lang}")
                        with open(savepath, 'w') as fs:
                            lines = [line[i] + '\n' for line in split_lines]  # split_lines is a tuple (src, trg)
                            fs.writelines(lines)
                            print(f"\t- Partition saved: {split_name}.{lang}")
            else:  # Create folders
                if not flag_raw_exists or not flag_splits_exists:
                    print(f"=> [Missing data]: We couldn't find either the 'raw' folder or the 'splits' folder.")
                    res = ask_yes_or_no(question="Do you want to create the missing directories?",
                                        interactive=self.interactive)
                    if res:
                        make_dir(raw_path) if res else None
                        make_dir(splits_path) if res else None
                        print("=> Directories created. ")

                # Notify about the datasets missing
                print(f"You need to add your dataset to at least one of these folders:")
                print(
                    f"\t- The '{ds.data_raw_path}' folder is used when you have two files (e.g. 'data.ru' and 'data.en')")
                print(
                    f"\t- The '{ds.data_splits_path}' folder is used when you have the train, val and test splits (e.g. '[train,val,test].[ru,en]'")
                print("*** Restart the program when these files are added ***")
                exit(0)

    def _create_reduced_versions(self):
        print("=> Creating reduced versions...")

        # Create reduce splits
        for ds in self.iter_main():  # Dataset
            # Ignore if this is the reference dataset
            ds_ref = ds.id()[0], ds.id()[1], self.ref_size_name
            if ds.id()[2] == self.ref_size_name:
                continue

            print(f"\t=> Checking dataset: {self._get_ds_alias(*ds.id())}")
            src_lang, trg_lang = ds.id()[1].split("-")

            # Create new splits folder *****
            make_dir(ds.get_split_path())

            # Add truncated splits
            for fname in ds.get_split_files():
                ori_filename = os.path.join(self.base_path, *ds_ref, ds.data_splits_path, fname)
                new_filename = ds.get_split_path(fname)

                # Copy n lines efficiently
                if self.force_overwrite or not os.path.exists(new_filename):
                    with open(ori_filename, 'r') as fin, open(new_filename, 'w') as fout:
                        lines = list(islice(fin, ds.dataset_lines))
                        fout.writelines(lines)
                        print(f"\t\t=> Creating split file: {fname}")

    def _pretokenize(self, force_pretok=False):
        print(f"=> Pretokenizing files... (only applied if needed)")

        for ds in self:  # Dataset
            pretok_flag = ds.pretok_flag or force_pretok

            # Check if this needs pretokenization
            if not pretok_flag:
                continue

            # Ignore dataset
            if ds.subword_model in {None, "none", "bytes"}:
                continue

            # Create paths
            pretokenize_path = ds.get_pretok_path()
            make_dir([pretokenize_path])

            print(f"\t- Pretokenizing splits")
            for fname in ds.get_split_files():
                lang = fname.split(".")[1]
                input_file = ds.get_split_path(fname)
                output_file = ds.get_pretok_path(fname)

                # Pretokenize
                pretokenize_file(input_file=input_file, output_file=output_file, lang=lang,
                                 force_overwrite=self.force_overwrite,
                                 use_cmd=self.use_cmd, conda_env_name=self.conda_env_name)

    def _build_vocab(self, force_pretok=False, input_sentence_size=1000000):
        print(f"=> Building vocabularies...")

        for ds in self:  # Dataset
            src_lang, trg_lang = ds.id()[1].split("-")

            # Create paths
            vocab_path = ds.get_vocab_path()
            tmp_path = os.path.join(ds.get_vocab_path(base=True), "_tmp")
            pretokenize_path = ds.get_pretok_path()
            make_dir([vocab_path, tmp_path, pretokenize_path])
            print(f"\t- Building vocabulary: {vocab_path}")

            # Ignore dataset but create directories (just in case... for plots or stats)
            if ds.subword_model in {None, "none", "bytes"}:
                continue

            # Get train files
            file_path_fn = ds.get_pretok_path if ds.pretok_flag or force_pretok else ds.get_split_path
            src_train_path = file_path_fn(fname=f"{ds.train_name}.{src_lang}")
            trg_train_path = file_path_fn(fname=f"{ds.train_name}.{trg_lang}")

            # One or two models
            if self.merge_vocabs:  # One model
                concat_train_path = os.path.join(tmp_path, f"{ds.train_name}.{src_lang}-{trg_lang}")

                # Concat files
                if self.force_overwrite or not os.path.exists(concat_train_path):
                    # Read files
                    lines = read_file_lines(src_train_path)
                    lines += read_file_lines(trg_train_path)

                    # Shuffle lines: Just in case because can spm_train load the first X lines of corpus by default
                    random.shuffle(lines)

                    # Save file
                    write_file_lines(lines=lines, filename=concat_train_path)
                files = [(concat_train_path, f"{src_lang}-{trg_lang}")]
            else:  # Two models
                files = [(src_train_path, f"{src_lang}"), (trg_train_path, f"{trg_lang}")]

            # Train models
            for input_file, ext in files:
                output_file = ds.get_vocab_file(lang=ext)  # without extension
                if self.force_overwrite or not os.path.exists(f"{output_file}.model"):
                    py_cmd_api.spm_train(input_file=input_file, model_prefix=output_file, subword_model=ds.subword_model,
                                         vocab_size=ds.vocab_size, input_sentence_size=input_sentence_size,
                                         use_cmd=self.use_cmd, conda_env_name=self.conda_env_name)

    def _encode_datasets(self, force_pretok=False):
        print(f"=> Building datasets...")

        for ds in self:  # Dataset
            src_lang, trg_lang = ds.id()[1].split("-")
            pretok_flag = ds.pretok_flag or force_pretok

            # Ignore dataset
            if ds.subword_model in {None, "none"}:
                continue

            # Create paths
            encoded_path = ds.get_encoded_path()
            make_dir([encoded_path])
            print(f"\t- Encoding dataset: {encoded_path}")

            # Encode files
            for fname in ds.get_split_files():
                lang = fname.split('.')[-1]
                data_path = ds.data_pretokenized_path if pretok_flag else ds.data_splits_path
                input_file = os.path.join(ds.base_path, *ds.id(), data_path, fname)
                output_file = ds.get_encoded_path(fname)

                # Encode file
                encode_file(ds=ds, input_file=input_file, output_file=output_file, output_file_pretok=None,
                            lang=lang, merge_vocabs=self.merge_vocabs, force_overwrite=self.force_overwrite,
                            use_cmd=self.use_cmd, conda_env_name=self.conda_env_name)

    def _export_vocab_frequencies(self, normalize=False):
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
                if ds.bytes_as_words:
                    split_fn = lambda x: x.split(' ')
                else:
                    split_fn = lambda x: [x for x in x.encode()]
            else:
                split_fn = lambda x: x.split(' ')
                spm_model = True

            # Get langs
            if self.merge_vocabs:
                lang_files = [f"{src_lang}-{trg_lang}"]
            else:
                lang_files = [src_lang, trg_lang]

            # Check if file/files exists
            vocab_files = [ds.get_vocab_path(fname=f)+".vocabf" for f in lang_files]
            if self.force_overwrite or not all([os.path.exists(f) for f in vocab_files]):
                # Get train paths
                src_train_path = ds.get_encoded_path(f"{ds.train_name}.{src_lang}")
                trg_train_path = ds.get_encoded_path(f"{ds.train_name}.{trg_lang}")

                # Read files
                src_lines = read_file_lines(src_train_path)
                trg_lines = read_file_lines(trg_train_path)

                # Get tokens
                src_tokens = flatten([split_fn(line) for line in src_lines])
                trg_tokens = flatten([split_fn(line) for line in trg_lines])

                if not spm_model:
                    #  Convert to counters
                    src_vocab = Counter(src_tokens)
                    trg_vocab = Counter(trg_tokens)
                    vocabs = [src_vocab + trg_vocab] if self.merge_vocabs else [src_vocab, trg_vocab]

                else:
                    if self.merge_vocabs:
                        tokens_lang = [(src_tokens+trg_tokens, f"{src_lang}-{trg_lang}")]
                    else:
                        tokens_lang = [(src_tokens, src_lang), (trg_tokens, trg_lang)]

                    # Count tokens
                    vocabs = []
                    for tokens, lang_file in tokens_lang:
                        # Get the exact vocab from SPM
                        spm_vocab_lines = read_file_lines(ds.get_vocab_path(fname=lang_file) + ".vocab")
                        spm_vocab_lines = spm_vocab_lines[4:]  # Remove special tokens
                        spm_vocab = {l.split('\t')[0]: 0 for l in spm_vocab_lines}

                        # Only count tokens that exists in the vocabulary
                        c = Counter(tok for tok in tokens if tok in spm_vocab)
                        vocabs.append(c)

                # Save vocabs
                for vocab, vocab_path in zip(vocabs, vocab_files):
                    # Normalize (if requested)
                    if normalize:
                        vocab = utils.norm_counter(vocab)

                    # Sort frequencies
                    vocab_frequencies = vocab.most_common()

                    # Save vocab
                    if self.force_overwrite or not os.path.exists(vocab_path):
                        lines = [f"{pair[0]}\t{pair[1]}\n" for pair in vocab_frequencies]
                        write_file_lines(lines=lines, filename=vocab_path)

    def _plot_datasets(self, save_figures=True, show_figures=False, add_dataset_title=True, vocab_top_k=None):
        print(f"=> Plotting started... (base_path={self.base_path})")
        print(f"- [WARNING]: Matplotlib might miss some images if the loop is too fast")

        # Set default vars
        if vocab_top_k is None:
            vocab_top_k = [100, 150]

        # Set backend
        if save_figures:
            plots.set_non_gui_backend()
            if show_figures:
                raise ValueError("'save_fig' is incompatible with 'show_fig'")

        # Walk through datasets
        for ds in self:  # Dataset
            ds_name, lang_pair, ds_size_name = ds.id()
            src_lang, trg_lang = ds.id()[1].split("-")

            # Set base path
            ds_title = f"{ds_name.title()} ({lang_pair}; {ds.subword_model}; {ds.vocab_size})"
            vocab_name = f"_{ds.vocab_size}" if ds.vocab_size else ""
            suffix_fname = f"{ds_name}_{ds_size_name}_{lang_pair}__{ds.subword_model}{vocab_name}".lower()
            print(f"\t- Creating plots for: {suffix_fname}")

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
                if ds.subword_model == "bytes":
                    tokens_by_sentence = utils.get_tokens_by_sentence(filename=os.path.join(encoded_path, fname),
                                                                      split_fn=lambda x: x.encode())
                else:
                    tokens_by_sentence = utils.get_tokens_by_sentence(filename=os.path.join(encoded_path, fname))

                # List to array
                tokens_by_sentence = np.array(tokens_by_sentence)

                # Compute data
                row = {
                    "total_sentences": len(tokens_by_sentence),
                    "total_tokens": int(tokens_by_sentence.sum()),
                    "max_tokens": int(np.max(tokens_by_sentence)),
                    "min_tokens": int(np.min(tokens_by_sentence)),
                    "avg_tokens": float(np.average(tokens_by_sentence)),
                    "std_tokens": float(np.std(tokens_by_sentence)),
                    "percentile5_tokens": int(np.percentile(tokens_by_sentence, 5)),
                    "percentile50_tokens": int(np.percentile(tokens_by_sentence, 50)),
                    "percentile95_tokens": int(np.percentile(tokens_by_sentence, 95)),
                    "split": split_name,
                    "lang": split_lang,
                }
                split_stats[fname] = row

                # Plot sentence length distribution (by tokens' length)
                df = pd.DataFrame(tokens_by_sentence, columns=["frequency"])
                title = f"Sentence length distribution ({split_name.title()} - {split_lang})"
                title = title if not add_dataset_title else f"{ds_title}:\n{title}"
                p_fname = f"sent_distr_{split_name}_{split_lang}__{suffix_fname}".lower()
                plots.histogram(data=df, x="frequency", output_dir=plots_encoded_path, fname=p_fname,
                                title=title, xlabel="Tokens per sentence", ylabel="Frequency", bins=100,
                                aspect_ratio=(6, 4), size=1.5, save_fig=save_figures, show_fig=show_figures,
                                overwrite=self.force_overwrite)

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
                          output_dir=plots_encoded_path, fname=p_fname, aspect_ratio=(8, 4), size=1.0,
                          save_fig=save_figures, show_fig=show_figures, overwrite=self.force_overwrite)

            # Plot split size (by token number)
            if ds.subword_model not in {None, "none"}:
                title = f"Split sizes (by number of tokens - {ds.subword_model.title()})"
                title = title if not add_dataset_title else f"{ds_title}:\n{title}"
                p_fname = f"split_size_tok__{suffix_fname}".lower()
                plots.catplot(data=df, x="split", y="total_tokens", hue="lang",
                              title=title, xlabel="Dataset partitions", ylabel="Num. of tokens", leyend_title=None,
                              output_dir=plots_encoded_path, fname=p_fname, aspect_ratio=(8, 4), size=1.0,
                              save_fig=save_figures, show_fig=show_figures, overwrite=self.force_overwrite)

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
                        df["frequency"] = df["frequency"].astype(int)
                        df = df.sort_values(by='frequency', ascending=False, na_position='last')

                    for top_k in vocab_top_k:
                        title = f"Vocabulary distribution (top {str(top_k)} {ds.subword_model.title()}; {lang_file})"
                        title = title if not add_dataset_title else f"{ds_title}:\n{title}"
                        p_fname = f"vocab_distr_{lang_file}_top{str(top_k)}__{suffix_fname}".lower()
                        plots.barplot(data=df.head(top_k), x="token", y="frequency",
                                      output_dir=plots_encoded_path, fname=p_fname,
                                      title=title, xlabel="Tokens", ylabel="Frequency",
                                      aspect_ratio=(12, 4), size=1.5, save_fig=save_figures, show_fig=show_figures,
                                      overwrite=self.force_overwrite)
