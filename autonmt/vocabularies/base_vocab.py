"""Base class for vocabularies.

All NMT vocabularies share the same four special tokens (unk/sos/eos/pad);
moving them onto the base keeps subclasses consistent and lets the rest of
the framework rely on ``vocab.unk_id`` / ``vocab.special_tokens()`` without
casting to a concrete type.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class BaseVocabulary(ABC):
    """Abstract base for all vocabularies.

    Owns the four special tokens shared by every NMT vocabulary
    (``unk`` / ``sos`` / ``eos`` / ``pad``) and their ids, so the rest of the
    framework can rely on ``vocab.pad_id`` / ``vocab.special_tokens()`` without
    knowing the concrete subclass. Subclasses implement :meth:`encode` /
    :meth:`decode` and the length / lookup protocol.
    """

    def __init__(self,
                 unk_id: int = 0, sos_id: int = 1, eos_id: int = 2, pad_id: int = 3,
                 unk_piece: str = "<unk>", sos_piece: str = "<s>",
                 eos_piece: str = "</s>", pad_piece: str = "<pad>",
                 lang: Optional[str] = None, max_tokens: Optional[int] = None):
        self.unk_id = unk_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

        self.unk_piece = unk_piece
        self.sos_piece = sos_piece
        self.eos_piece = eos_piece
        self.pad_piece = pad_piece

        self.lang = lang
        self.max_tokens = max_tokens

    def special_tokens(self) -> List[Tuple[str, int]]:
        return [
            (self.unk_piece, self.unk_id),
            (self.sos_piece, self.sos_id),
            (self.eos_piece, self.eos_id),
            (self.pad_piece, self.pad_id),
        ]

    @abstractmethod
    def encode(self, *args, **kwargs):
        pass

    @abstractmethod
    def decode(self, *args, **kwargs):
        pass
