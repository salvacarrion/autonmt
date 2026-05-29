"""The torch ``Dataset`` for parallel text.

Reads the already-encoded ``<prefix>.<src>`` / ``<prefix>.<tgt>`` files into
memory as raw strings; ``collate_fn`` does the per-batch vocab encoding and
padding (so vocab choice / max_tokens stay a DataLoader-time concern, not a
read-time one). The matching path engine is
:class:`~autonmt.datasets.dataset.DatasetLayout`.
"""
import functools
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from autonmt.utils.fileio import read_file_lines
from autonmt.utils.logger import get_logger

log = get_logger(__name__)


class TranslationDataset(Dataset):
    def __init__(self, file_prefix, src_lang, tgt_lang, src_vocab=None, tgt_vocab=None, filter_fn=None):
        # Set vocabs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        # Get src/tgt file paths
        src_file_path = file_prefix.strip() + f".{src_lang}"
        tgt_file_path = file_prefix.strip() + f".{tgt_lang}"

        # Read files
        self.src_lines = read_file_lines(filename=src_file_path, autoclean=True)
        self.tgt_lines = read_file_lines(filename=tgt_file_path, autoclean=True)

        # Filter langs
        if filter_fn:
            self.src_lines, self.tgt_lines = filter_fn(self.src_lines, self.tgt_lines)

        assert len(self.src_lines) == len(self.tgt_lines)

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_line, tgt_line = self.src_lines[idx], self.tgt_lines[idx]
        return src_line, tgt_line

    def collate_fn(self, batch, max_tokens=None):
        x_encoded, y_encoded = [], []
        x_max_len = y_max_len = 0

        # Add elements to batch
        for i, (x, y) in enumerate(batch):
            # Encode tokens
            _x = self.src_vocab.encode(x)
            _y = self.tgt_vocab.encode(y)

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
                log.info(msg.format(drop_ratio, max_tokens))
                break

        # Get lengths
        x_len = torch.tensor([len(x) for x in x_encoded], dtype=torch.long)
        y_len = torch.tensor([len(y) for y in y_encoded], dtype=torch.long)

        # Pad sequence
        x_padded = pad_sequence(x_encoded, batch_first=False, padding_value=self.src_vocab.pad_id).T
        y_padded = pad_sequence(y_encoded, batch_first=False, padding_value=self.tgt_vocab.pad_id).T

        assert x_padded.shape[0] == y_padded.shape[0] == len(x_encoded)  # Control samples
        assert max_tokens is None or (x_padded.numel() + y_padded.numel()) <= max_tokens  # Control max tokens
        return (x_padded, y_padded), (x_len, y_len)

    def get_collate_fn(self, max_tokens):
        return functools.partial(self.collate_fn, max_tokens=max_tokens)

