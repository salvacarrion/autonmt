"""Whitespace-tokenised vocabulary backed by a ``.vocab`` file.

Handles three encoding regimes via the :class:`~autonmt.utils.enums.SubwordModel`
enum instead of magic strings:

  * SentencePiece-trained models (word/bpe/unigram/char) — encoded ids look up
    pieces in ``voc2idx``; decoding joins pieces with spaces.
  * ``bytes`` — pieces are hex-encoded code points (e.g. ``0x61``); decoding
    converts back via :func:`autonmt.utils.fileio.hex2text`.
  * No subword model (``None``) — vocab loaded but encoding is not used.

The bytes/SPM dispatch is encapsulated in private helpers so callers never see
``if subword_model == "bytes"`` checks.
"""
from typing import List, Optional

from autonmt.utils.enums import SubwordModel, is_bytes_only, is_no_model
from autonmt.utils.fileio import hex2text, read_file_lines, write_file_lines
from autonmt.vocabularies.base_vocab import BaseVocabulary


class Vocabulary(BaseVocabulary):

    def __init__(self,
                 unk_id: int = 0, sos_id: int = 1, eos_id: int = 2, pad_id: int = 3,
                 unk_piece: str = "<unk>", sos_piece: str = "<s>",
                 eos_piece: str = "</s>", pad_piece: str = "<pad>",
                 lang: Optional[str] = None, max_tokens: Optional[int] = None):
        super().__init__(
            unk_id=unk_id, sos_id=sos_id, eos_id=eos_id, pad_id=pad_id,
            unk_piece=unk_piece, sos_piece=sos_piece, eos_piece=eos_piece, pad_piece=pad_piece,
            lang=lang, max_tokens=max_tokens,
        )

        self.voc2idx = {}
        self.idx2voc = {}
        self.voc2freq = {}

        self.vocab_path: Optional[str] = None
        self.pretok_flag: Optional[bool] = None
        self.subword_model: Optional[SubwordModel] = None
        self.model_path: Optional[str] = None
        self.spm_model = None

    def __len__(self) -> int:
        return len(self.voc2idx)

    # --- Loading ---------------------------------------------------------

    def _load_spm_model(self, path: str) -> None:
        import sentencepiece as spm
        self.spm_model = spm.SentencePieceProcessor(model_file=path)

    @staticmethod
    def _parse_vocab_count(value: str):
        """Parse the second (score/count) column of a ``.vocab`` line.

        AutoNMT-written vocabs and SentencePiece ``bpe`` models store an integer
        here, but ``unigram`` / ``char`` / ``word`` models store a float log-prob
        score. Accept both so loading never depends on the subword model (the
        value only feeds :meth:`save`; real token frequencies live in ``.vocabf``).
        """
        value = value.strip()
        try:
            return int(value)
        except ValueError:
            return float(value)

    def _build_from_vocab(self, filename: str, includes_special_tokens: bool = True) -> "Vocabulary":
        tokens = [line.rstrip('\n').split('\t') for line in read_file_lines(filename, autoclean=False)]
        tokens = [(tok, self._parse_vocab_count(freq)) for tok, freq in tokens]
        special = [(tok, 0) for tok, _ in self.special_tokens()] if not includes_special_tokens else []
        tokens = special + tokens  # Special tokens must appear first in the file
        self.voc2idx = {tok: idx for idx, (tok, _) in enumerate(tokens)}
        self.idx2voc = {idx: tok for idx, (tok, _) in enumerate(tokens)}
        self.voc2freq = {tok: freq for _, (tok, freq) in enumerate(tokens)}
        self._assert_vocab()
        return self

    def _assert_vocab(self) -> None:
        assert self.idx2voc[self.unk_id] == self.unk_piece
        assert self.idx2voc[self.sos_id] == self.sos_piece
        assert self.idx2voc[self.eos_id] == self.eos_piece
        assert self.idx2voc[self.pad_id] == self.pad_piece

    def build_from_ds(self, ds, lang: Optional[str] = None) -> "Vocabulary":
        self.lang = ds.dataset_lang_pair if lang is None else lang
        self.pretok_flag = ds.pretok_flag
        self.subword_model = ds.subword_model

        if is_no_model(ds.subword_model):
            raise ValueError("No subword model selected")
        if not is_bytes_only(ds.subword_model):
            self.model_path = ds.get_vocab_path(self.lang) + ".model"
            self._load_spm_model(self.model_path)

        self.vocab_path = ds.get_vocab_path(self.lang) + ".vocab"
        self._build_from_vocab(self.vocab_path)
        return self

    # --- Encode / decode -------------------------------------------------

    def get_tokens(self) -> List[str]:
        return [self.idx2voc[i] for i in range(len(self.idx2voc))]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = text.strip().split(' ')
        idxs = [self.voc2idx.get(tok, self.unk_id) for tok in tokens]
        if self.max_tokens:
            # Reserve room for <sos> and <eos>
            idxs = idxs[:self.max_tokens - 2 * int(add_special_tokens)]
        if add_special_tokens:
            idxs = [self.sos_id] + idxs + [self.eos_id]
        return idxs

    def _strip_special_tokens(self, idxs: List[int]) -> List[int]:
        try:
            idxs = idxs[idxs.index(self.sos_id) + 1:]
        except ValueError:
            pass
        try:
            idxs = idxs[:idxs.index(self.eos_id)]
        except ValueError:
            pass
        return idxs

    def _decode_bytes(self, idxs: List[int]) -> str:
        offset = len(self.special_tokens())
        tokens = [self.idx2voc.get(idx) for idx in idxs if idx >= offset]
        return ''.join(hex2text(tokens))

    def _decode_pieces(self, idxs: List[int]) -> str:
        tokens = [self.idx2voc.get(idx, self.unk_piece) for idx in idxs]
        return ' '.join(tokens)

    def decode(self, idxs: List[int], remove_special_tokens: bool = True) -> str:
        if remove_special_tokens:
            idxs = self._strip_special_tokens(idxs)
        if is_bytes_only(self.subword_model):
            return self._decode_bytes(idxs)
        return self._decode_pieces(idxs)

    # --- Persistence -----------------------------------------------------

    def save(self, filename: str, include_special_tokens: bool = True) -> None:
        lines = []
        if include_special_tokens:
            # Order matters: it must match (unk_id=0, sos_id=1, eos_id=2, pad_id=3).
            for piece, _ in self.special_tokens():
                lines.append(f"{piece}\t0")
        for voc in self.voc2idx:
            lines.append(f"{voc}\t{self.voc2freq.get(voc, 0)}")
        write_file_lines(lines=lines, filename=filename, insert_break_line=True)
