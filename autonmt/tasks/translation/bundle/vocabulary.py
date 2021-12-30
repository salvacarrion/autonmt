from autonmt.utils import read_file_lines
from collections import Counter
from autonmt.utils import read_file_lines, write_file_lines, flatten

from abc import ABC, abstractmethod


class BaseVocabulary(ABC):
    def __init__(self, sos_id, eos_id, pad_id, sos_piece, eos_piece, pad_piece):
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

    @abstractmethod
    def encode(self, *args, **kwargs):
        pass

    @abstractmethod
    def decode(self, *args, **kwargs):
        pass


class Vocabulary(BaseVocabulary):

    def __init__(self, lang=None,
                 unk_id=0, sos_id=1, eos_id=2, pad_id=3,
                 unk_piece="<unk>", sos_piece="<s>", eos_piece="</s>", pad_piece="<pad>"):
        super().__init__(sos_id=sos_id, eos_id=eos_id, pad_id=pad_id,
                         sos_piece=sos_piece, eos_piece=eos_piece, pad_piece=pad_piece)

        # Set language
        self.lang = lang

        # Set special tokens
        self.unk_id = unk_id
        self.unk_piece = unk_piece
        self.special_tokens = [(self.unk_piece, self.unk_id), (self.sos_piece, self.sos_id),
                               (self.eos_piece, self.eos_id), (self.pad_piece, self.pad_id)]

        # Build vocab
        self.voc2idx = {}
        self.idx2voc = {}
        self.voc2freq = {}

    def __len__(self):
        return len(self.voc2idx)

    def _assert_vocab(self):
        assert self.idx2voc[self.unk_id] == self.unk_piece
        assert self.idx2voc[self.sos_id] == self.sos_piece
        assert self.idx2voc[self.eos_id] == self.eos_piece
        assert self.idx2voc[self.pad_id] == self.pad_piece

    def _build_from_tokens(self, tokens):
        # Tokens must include the special tokens
        self.voc2idx = {tok: idx for idx, (tok, log_prob) in enumerate(tokens)}
        self.idx2voc = {idx: tok for idx, (tok, log_prob) in enumerate(tokens)}
        self.voc2freq = {tok: log_prob for idx, (tok, log_prob) in enumerate(tokens)}
        self._assert_vocab()
        return self

    def build_from_vocab(self, filename, includes_special_tokes=True):
        # Parse file. Special tokens must appear first in the file
        tokens = [line.split('\t') for line in read_file_lines(filename)]
        special_tokens = [(tok, 0) for tok, tok_id in self.special_tokens] if not includes_special_tokes else []
        tokens = special_tokens + tokens  # Do not sort. It could lead to different idxs
        self._build_from_tokens(tokens)
        self._assert_vocab()
        return self

    def build_from_dataset(self, filename):
        tokens = Counter(flatten([line.strip().split(' ') for line in read_file_lines(filename)]))
        special_tokens = [(tok, 0) for tok, tok_id in self.special_tokens]
        tokens = special_tokens + tokens.most_common()
        self._build_from_tokens(tokens)
        self._assert_vocab()
        return self

    def encode(self, text, add_special_tokens=True):
        tokens = text.strip().split(' ')
        idxs = [self.voc2idx.get(tok, self.unk_id) for tok in tokens]
        idxs = [self.sos_id] + idxs + [self.eos_id] if add_special_tokens else idxs
        return idxs

    def decode(self, idxs, remove_special_tokens=True):
        # Remove special tokens
        if remove_special_tokens:
            try:
                # Remove <sos>
                sos_pos = idxs.index(self.sos_id)
                idxs = idxs[sos_pos+1:]
            except ValueError:
                pass
            try:
                # Remove <eos>
                eos_pos = idxs.index(self.eos_id)
                idxs = idxs[:eos_pos]
            except ValueError:
                pass

        # Decode sentences
        tokens = [self.idx2voc.get(idx, self.unk_piece) for idx in idxs]
        s = ' '.join(tokens)
        return s

    def save(self, filename, include_special_tokens=True):
        lines = []

        # Add special tokens
        if include_special_tokens:
            lines.append((self.unk_piece, 0))
            lines.append((self.sos_piece, 0))
            lines.append((self.eos_piece, 0))
            lines.append((self.pad_piece, 0))

        # Add tokens
        for voc, idx in self.voc2idx.items():
            lines.append(f"{voc}\t{self.voc2freq.get(voc, 0)}")

        # Save file
        write_file_lines(lines=lines, filename=filename)


class VocabularyBytes(BaseVocabulary):
    def __init__(self, hex_input=False, sos_id=256, eos_id=257, pad_id=258,
                 sos_piece="<s>", eos_piece="</s>", pad_piece="<pad>"):
        super().__init__(sos_id=sos_id, eos_id=eos_id, pad_id=pad_id,
                         sos_piece=sos_piece, eos_piece=eos_piece, pad_piece=pad_piece)
        self.hex_input = hex_input

        # Set special tokens
        self._offset = len(self.special_tokens)

    def __len__(self):
        return 256 + len(self.special_tokens)

    def encode(self, text, add_special_tokens=True):
        if self.hex_input:
            b_list = [int(x, base=16) for x in text.split(' ')]
        else:
            s_bytes = text.encode()  # b'Hello world! \xf0\x9f\x8c\xb1'
            b_list = [b for b in s_bytes]  # [72, 101, 108,...]
        idxs = [self.sos_id] + b_list + [self.eos_id] if add_special_tokens else b_list
        return idxs

    def decode(self, idxs, remove_special_tokens=True):
        # Remove special tokens
        if remove_special_tokens:
            try:
                # Remove <sos>
                sos_pos = idxs.index(self.sos_id)
                idxs = idxs[sos_pos+1:]
            except ValueError:
                pass
            try:
                # Remove <eos>
                eos_pos = idxs.index(self.eos_id)
                idxs = idxs[:eos_pos]
            except ValueError:
                pass

        # Decode idxs
        if self.hex_input:
            text = " ".join([hex(x) for x in idxs])
        else:
            b_enc = bytes(idxs)
            text = b_enc.decode()
        return text
