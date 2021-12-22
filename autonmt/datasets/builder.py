import random
from itertools import islice

import numpy as np
import pandas as pd
import tqdm

random.seed(123)

from autonmt.utils import *
from autonmt import plots, utils
from autonmt.datasets.dataset import Dataset
from autonmt.cmd import cmd_tokenizers


class DatasetBuilder:
    def __init__(self, base_path, datasets, subword_models, vocab_sizes, force_overwrite=False, interactive=True):
        self.base_path = base_path
        self.datasets = datasets
        self.subword_models = subword_models
        self.vocab_sizes = vocab_sizes
        self.force_overwrite = force_overwrite
        self.interactive = interactive
        
        self.ref_size_name = "original"
        
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
                    params = dict(base_path=self.base_path, dataset_name=ds["name"], dataset_lang_pair=lang_pair,
                                  dataset_size_name=ds_size_name, dataset_lines=ds_max_lines)

                    if include_variants:
                        for subword_model in self.subword_models:  # unigram, bpe, char, or word
                            for vocab_size in self.vocab_sizes:
                                _ds = Dataset(subword_model=subword_model, vocab_size=vocab_size, **params)
                                ds_list_tmp.append(_ds)
                    else:
                        _ds = Dataset(**params)
                        ds_list_tmp.append(_ds)
        return ds_list_tmp

    def iter_main(self):
        return self.ds_list_main

    def build(self, encode=True, val_size=(0.1, 5000), test_size=(0.1, 5000), shuffle=True,
              make_plots=False, safe=True):
        print(f"=> Building datasets...")
        print(f"\t- base_path={self.base_path}")

        # Create splits
        self._create_splits(val_size=val_size, test_size=test_size, shuffle=shuffle, safe=safe)

        # Create version for different sizes
        self._create_reduced_versions()

        # Build vocabs
        self._build_vocab()

        # Encode datasets
        if encode:
            self._encode_datasets()

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

    def _build_vocab(self, character_coverage=1.0, force_pretok=False, merge_vocabs=True):
        print(f"=> Building vocabularies...")

        for ds in self:  # Dataset
            src_lang, trg_lang = ds.id()[1].split("-")
            pretok_flag = ds.pretok_flag or force_pretok

            # Create paths
            vocab_path = ds.get_vocab_path()
            concat_path = os.path.join(ds.get_vocab_path(base=True), "_tmp")
            pretokenize_path = ds.get_pretok_path()
            make_dir([vocab_path, concat_path, pretokenize_path])
            print(f"\t- Building vocabulary: {vocab_path}")

            # Pretokenize all (if needed)
            if pretok_flag:
                print(f"\t- Pretokenizing splits because 'subword_model'='{ds.subword_model}'")
                for fname in ds.get_split_files():
                    ori_filename = ds.get_split_path(fname)
                    new_filename = ds.get_pretok_path(fname)
                    lang = fname.split(".")[1]

                    # Tokenize
                    if self.force_overwrite or not os.path.exists(new_filename):
                        cmd_tokenizers.moses_tokenizer(input_file=ori_filename, output_file=new_filename, lang=lang)
                        print(f"\t\t- Pretokenized file: {fname}")

            # Concatenate train files
            if not merge_vocabs:
                raise NotImplementedError("Only merge vocabs is allowed")
            else:
                # Concat training sets
                train_concat_fname = "train_pretok.txt" if pretok_flag else "train_raw.txt"
                new_filename = os.path.join(concat_path, train_concat_fname)

                # Concat train files
                if self.force_overwrite or not os.path.isfile(new_filename):
                    lines = []
                    for lang in [src_lang, trg_lang]:
                        fname = f"{ds.train_name}.{lang}"

                        # Get file to concat (split or pretok)
                        data_path = ds.data_pretokenized_path if ds.pretok_flag else ds.data_splits_path
                        filename = os.path.join(ds.base_path, *ds.id(), data_path, fname)

                        # Concatenate train files
                        with open(filename, 'r') as infile:
                            lines += infile.readlines()

                    # Shuffle lines (just in case)
                    random.shuffle(lines)

                    # Save new file
                    with open(new_filename, 'w') as outfile:
                        outfile.writelines(lines)

                    # Remove lines from memory
                    del lines

            # Train model
            model_prefix = ds.get_src_trg_vocab_path()
            if self.force_overwrite or not os.path.exists(f"{model_prefix}.model"):
                cmd_tokenizers.spm_train(input_file=new_filename, model_prefix=model_prefix, vocab_size=ds.vocab_size,
                                           character_coverage=character_coverage, subword_model=ds.subword_model)

    def _encode_datasets(self, export_frequencies=True, force_pretok=False):
        print(f"=> Building datasets...")

        for ds in self:  # Dataset
            src_lang, trg_lang = ds.id()[1].split("-")
            pretok_flag = ds.pretok_flag or force_pretok

            # Create paths
            encoded_path = ds.get_encoded_path()
            make_dir([encoded_path])
            print(f"\t- Encoding dataset: {encoded_path}")

            # Encode files
            for fname in ds.get_split_files():
                data_path = ds.data_pretokenized_path if pretok_flag else ds.data_splits_path
                ori_filename = os.path.join(ds.base_path, *ds.id(), data_path, fname)
                new_filename = ds.get_encoded_path(fname)

                # Encode
                if self.force_overwrite or not os.path.exists(new_filename):
                    cmd_tokenizers.spm_encode(spm_model_path=ds.get_src_trg_vocab_path()+".model",
                                                input_file=ori_filename, output_file=new_filename)
                    print(f"\t\t - Encoded file: {fname}")

            # Export vocab frequencies
            if export_frequencies:
                self._export_vocab_frequencies(ds)

    def _export_vocab_frequencies(self, ds):
        src_lang, trg_lang = ds.id()[1].split("-")

        # Get vocab paths
        spm_vocab_path = ds.get_src_trg_vocab_path() + ".vocab"
        spm_vocab_freq_path = ds.get_src_trg_vocab_path() + ".vocabf"

        # Create vocabs and export it
        if self.force_overwrite or not os.path.exists(spm_vocab_freq_path):
            # Load vocab
            vocabs = {l.strip().split('\t')[0] for l in open(spm_vocab_path, 'r').readlines()}

            # Count tokens
            vocab_frequencies = defaultdict(int)
            for fname in [f"{ds.train_name}.{src_lang}", f"{ds.train_name}.{trg_lang}"]:
                filename = ds.get_encoded_path(fname)
                with open(filename, 'r') as f:
                    for line in tqdm(f):
                        tokens = line.strip().split(' ')
                        for tok in tokens:
                            if tok in vocabs:  # Count only the tokens that exists in the vocab
                                vocab_frequencies[tok] += 1

            # Sort frequencies
            vocab_frequencies = sorted(list(vocab_frequencies.items()), key=lambda x: x[1], reverse=True)

            # Save file
            with open(spm_vocab_freq_path, 'w') as f:
                f.writelines([f"{pair[0]}\t{pair[1]}\n" for pair in vocab_frequencies])

    def _plot_datasets(self,save_figures=True, show_figures=False,
                       add_dataset_title=True):
        print(f"=> Plotting started... (base_path={self.base_path})")
        print(f"- [WARNING]: Matplotlib might miss some images if the loop is too fast")

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
            suffix_fname = f"{ds_name}_{ds_size_name}_{lang_pair}__{ds.subword_model}_{ds.vocab_size}"
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
                tokens_by_sentence = np.array(
                    utils.get_tokens_by_sentence(filename=os.path.join(encoded_path, fname)))

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

                # Plot sentence length distribution (by tokens' length): 3x2
                df = pd.DataFrame(tokens_by_sentence, columns=["frequency"])
                title = f"Sentence length distribution"
                title = title if not add_dataset_title else f"{ds_title}: {title}"
                p_fname = f"sent_distr_{split_name}_{split_lang}__{suffix_fname}"
                plots.histogram(data=df, x="frequency", output_dir=plots_encoded_path, fname=p_fname,
                                title=title, xlabel="Tokens per sentence", ylabel="Frequency", bins=100,
                                aspect_ratio=(6, 4), size=1.5, save_fig=save_figures, show_fig=show_figures,
                                overwrite=self.force_overwrite)

            # Create dataframe
            df = pd.DataFrame(list(split_stats.values()))

            # Save data
            utils.save_json(split_stats, savepath=os.path.join(plots_encoded_path, f"stats__{suffix_fname}.json"))
            # df.to_csv(os.path.join(plots_encoded_path, f"stats__{base_fname}.csv"), index=False)

            # Plot split size (by its sentence number): 1
            print(f"\t\t- Creating 'Split sizes' plots...")
            title = f"Split sizes (by sentences)"
            title = title if not add_dataset_title else f"{ds_title}: {title}"
            p_fname = f"split_size_sent__{suffix_fname}"
            plots.catplot(data=df, x="split", y="total_sentences", hue="lang",
                          title=title, xlabel="Dataset partitions", ylabel="Num. of sentences",
                          leyend_title=None,
                          output_dir=plots_encoded_path, fname=p_fname, aspect_ratio=(8, 4), size=1.0,
                          save_fig=save_figures, show_fig=show_figures, overwrite=self.force_overwrite)

            # Plot split size (by token number): 1
            title = f"Split sizes (by tokens)"
            title = title if not add_dataset_title else f"{ds_title}: {title}"
            p_fname = f"split_size_tok__{suffix_fname}"
            plots.catplot(data=df, x="split", y="total_tokens", hue="lang",
                          title=title, xlabel="Dataset partitions", ylabel="Num. of tokens", leyend_title=None,
                          output_dir=plots_encoded_path, fname=p_fname, aspect_ratio=(8, 4), size=1.0,
                          save_fig=save_figures, show_fig=show_figures, overwrite=self.force_overwrite)

            # Plot vocabulary frequency: 1
            print(f"\t\t- Creating 'Vocabulary distribution' plots...")

            # Load vocabulary
            vocab_freq_path = os.path.join(vocab_path, f"spm_{src_lang}-{trg_lang}.vocabf")
            with open(vocab_freq_path, 'r') as f:
                rows = [line.split('\t') for line in f.readlines()]
                df = pd.DataFrame(rows, columns=["token", "frequency"])
                df["frequency"] = df["frequency"].astype(int)
                df = df.sort_values(by='frequency', ascending=False, na_position='last')

            for top_k in [100, 150]:
                title = f"Vocabulary distribution (top {str(top_k)})"
                title = title if not add_dataset_title else f"{ds_title}: {title}"
                p_fname = f"vocab_distr_top{str(top_k)}__{suffix_fname}"
                plots.barplot(data=df.head(top_k), x="token", y="frequency",
                              output_dir=plots_encoded_path, fname=p_fname,
                              title=title, xlabel="Tokens", ylabel="Frequency",
                              aspect_ratio=(12, 4), size=1.5, save_fig=save_figures, show_fig=show_figures,
                              overwrite=self.force_overwrite)
