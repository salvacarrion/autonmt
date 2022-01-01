from collections import Counter
from autonmt.bundle.utils import read_file_lines, write_file_lines, flatten

from abc import ABC, abstractmethod


class BaseVocabulary(ABC):
    def __init__(self, sos_id, eos_id, pad_id, sos_piece, eos_piece, pad_piece, lang=None, max_tokens=None):
        # Set IDs
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

        # Set pieces: <s>, </s>, <pad>
        self.sos_piece = sos_piece
        self.eos_piece = eos_piece
        self.pad_piece = pad_piece

        # Set special tokens
        self.special_tokens = [(self.sos_piece, self.sos_id), (self.eos_piece, self.eos_id),
                               (self.pad_piece, self.pad_id)]

        # Other
        self.lang = lang
        self.max_tokens = max_tokens

    @abstractmethod
    def encode(self, *args, **kwargs):
        pass

    @abstractmethod
    def decode(self, *args, **kwargs):
        pass


