import os
import math
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from autonmt.tasks.translation.bundle.vocabulary import BaseVocabulary, Vocabulary
from autonmt.utils import read_file_lines


class TranslationDataset(Dataset):
    def __init__(self, file_prefix, src_lang, trg_lang, src_vocab=None, trg_vocab=None):
        # Get src/trg file paths
        split_name = os.path.split(file_prefix)[1]
        src_file_path = file_prefix.strip() + f".{src_lang}"
        trg_file_path = file_prefix.strip() + f".{trg_lang}"

        # Read files
        self.src_lines = read_file_lines(filename=src_file_path)
        self.trg_lines = read_file_lines(filename=trg_file_path)
        assert len(self.src_lines) == len(self.trg_lines)

        # Read langs
        self.src_lang = src_lang
        self.trg_lang = trg_lang

        # Read vocabs: src
        if src_vocab and isinstance(src_vocab, str):
            self.src_vocab = Vocabulary(lang=src_lang).build_from_vocab(filename=src_vocab)
        elif src_vocab and isinstance(src_vocab, BaseVocabulary):
            self.src_vocab = src_vocab
        else:
            print(f"\t- [WARNING]: No 'src_vocab_path'. Generating vocabulary from 'src_file'...")
            self.src_vocab = Vocabulary(lang=src_lang).build_from_dataset(filename=src_file_path)

            # Save vocab
            savepath = f"{split_name}_{self.src_lang}.vocabf"
            self.src_vocab.save(filename=savepath)
            print(f"\t- [INFO]: 'src_vocab' save at: {savepath}")

        # Read vocabs: trg
        if trg_vocab and isinstance(trg_vocab, str):
            self.trg_vocab = Vocabulary(lang=trg_lang).build_from_vocab(filename=trg_vocab)
        elif trg_vocab and isinstance(trg_vocab, BaseVocabulary):
            self.trg_vocab = trg_vocab
        else:
            print(f"\t- [WARNING]: No 'trg_vocab_path'. Generating vocabulary from 'trg_file'...")
            self.trg_vocab = Vocabulary(lang=trg_lang).build_from_dataset(filename=trg_file_path)

            # Save vocab
            savepath = f"{split_name}_{self.trg_lang}.vocabf"
            self.trg_vocab.save(filename=savepath)
            print(f"\t- [INFO]: 'trg_vocab' save at: {savepath}")

        # Print info
        if src_vocab == trg_vocab:
            print(f"\t- [INFO]: Loaded shared vocab for '{split_name}' with {len(self.src_vocab):,} tokens")
        else:
            print(f"\t- [INFO]: Loaded src vocab for '{split_name}' with {len(self.src_vocab):,} tokens")
            print(f"\t- [INFO]: Loaded trg vocab for '{split_name}' with {len(self.trg_vocab):,} tokens")

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_line, trg_line = self.src_lines[idx], self.trg_lines[idx]
        return src_line, trg_line

    def collate_fn(self, batch, max_tokens=None):
        x_encoded, y_encoded = [], []
        x_max_len = y_max_len = 0

        # Add elements to batch
        for i, (x, y) in enumerate(batch):
            _x = self.src_vocab.encode(x)
            _y = self.trg_vocab.encode(y)

            # Control tokens in batch
            x_max_len = max(x_max_len, len(_x))
            y_max_len = max(y_max_len, len(_y))

            # Add elements
            if max_tokens is None or (i+1)*(x_max_len+y_max_len) <= max_tokens:  # sample*size
                x_encoded.append(torch.tensor(_x, dtype=torch.long))
                y_encoded.append(torch.tensor(_y, dtype=torch.long))
            else:
                msg = "[WARNING] Dropping {:.2f}% of the batch because the maximum number of tokens ({}) was exceeded"
                drop_ratio = 1 - ((i+1)/len(batch))
                print(msg.format(drop_ratio, max_tokens))
                break

        # Pad sequence
        x_padded = pad_sequence(x_encoded, batch_first=False, padding_value=self.src_vocab.pad_id).T
        y_padded = pad_sequence(y_encoded, batch_first=False, padding_value=self.trg_vocab.pad_id).T

        # Check stuff
        assert x_padded.shape[0] == y_padded.shape[0] == len(x_encoded)  # Control samples
        assert max_tokens is None or (x_padded.numel() + y_padded.numel()) <= max_tokens  # Control max tokens
        return x_padded, y_padded
