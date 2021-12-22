from autonmt.utils import read_file_lines
from collections import Counter
from autonmt.utils import read_file_lines, flatten


class Vocabulary:
    def __init__(self, lang=None,
                 unk_id=0, sos_id=1, eos_id=2, pad_id=3,
                 unk_piece="<unk>", sos_piece="<s>", eos_piece="</s>", pad_piece="<pad>"):
        # Set language
        self.lang = lang

        # Set special tokens
        self.unk_id = unk_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.unk_piece = unk_piece
        self.sos_piece = sos_piece
        self.eos_piece = eos_piece
        self.pad_piece = pad_piece
        self.special_tokens = [(self.unk_piece, self.unk_id), (self.sos_piece, self.sos_id),
                               (self.eos_piece, self.eos_id), (self.pad_piece, self.pad_id)]

        # Build vocab
        self.voc2idx = {}
        self.idx2voc = {}

    def __len__(self):
        return len(self.voc2idx)

    def _assert_vocab(self):
        assert self.idx2voc[self.unk_id] == self.unk_piece
        assert self.idx2voc[self.sos_id] == self.sos_piece
        assert self.idx2voc[self.eos_id] == self.eos_piece
        assert self.idx2voc[self.pad_id] == self.pad_piece

    def _build_from_tokens(self, tokens):
        self.voc2idx = {tok: idx for idx, (tok, log_prob) in enumerate(tokens)}
        self.idx2voc = {idx: tok for idx, (tok, log_prob) in enumerate(tokens)}
        self._assert_vocab()
        return self

    def build_from_filename(self, filename):
        # Parse file. Special tokens must appear first in the file
        tokens = [line.split('\t') for line in read_file_lines(filename)]
        self._build_from_tokens(tokens)
        self._assert_vocab()
        return self

    def build_from_dataset(self, filename):
        tokens = flatten([line.strip().split(' ') for line in read_file_lines(filename)])
        c = Counter(tokens)
        tokens = [(tok, freq) for tok, freq in c.items()]
        tokens = self.special_tokens + sorted(tokens, key=lambda x: x[1], reverse=True)
        self._build_from_tokens(tokens)
        self._assert_vocab()
        return self

    def encode(self, tokens, add_special_tokens=False):
        idxs = [self.voc2idx.get(tok, self.unk_id) for tok in tokens]
        idxs = [self.sos_id] + idxs + [self.eos_id] if add_special_tokens else idxs
        return idxs

    def decode(self, idxs, remove_special_tokens=False):
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
        return tokens
