"""String-valued Enums for internal flags.

Subclassing ``str`` keeps string comparisons (`x == "all"`) working so existing
user code and persisted configs do not need to change. ``__str__`` is overridden
so that ``str(member)`` returns the value (``"all"``), not the qualified name —
this matters because members get stringified inside path computations.
"""
from enum import Enum
from typing import Optional, Tuple


class _StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class EvalMode(_StrEnum):
    SAME = "same"
    COMPATIBLE = "compatible"
    ALL = "all"

    @classmethod
    def coerce(cls, value):
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError:
            valid = ", ".join(repr(m.value) for m in cls)
            raise ValueError(f"Unknown eval_mode {value!r} (expected one of: {valid})")


class SourceData(_StrEnum):
    RAW = "raw"
    RAW_PREPROCESSED = "raw_preprocessed"
    SPLITS = "splits"


class SubwordModel(_StrEnum):
    """Tokenization scheme. Byte fallback is a separate, orthogonal flag
    (see ``Dataset(byte_fallback=...)``) — not encoded into this enum."""
    NONE = "none"
    WORD = "word"
    CHAR = "char"
    BYTES = "bytes"
    BPE = "bpe"
    UNIGRAM = "unigram"

    @classmethod
    def coerce(cls, value):
        """Accept ``None``, string (case-insensitive), or member. Returns a member or ``None``."""
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.lower())
            except ValueError:
                pass
        valid = ", ".join(repr(m.value) for m in cls)
        raise ValueError(f"Unknown subword_model {value!r} (expected one of: {valid}, or None)")

    @classmethod
    def parse_with_byte_fallback(cls, value, default_byte_fallback: bool = False
                                 ) -> Tuple[Optional["SubwordModel"], bool]:
        """Split ``"<model>+bytes"`` sugar into ``(SubwordModel, byte_fallback=True)``.

        Single source of truth for the ``+bytes`` shorthand used across the
        builder, the dataset, and run-name composition.
        """
        bf = bool(default_byte_fallback)
        if isinstance(value, str) and value.lower().endswith("+bytes"):
            value = value[: -len("+bytes")]
            bf = True
        return cls.coerce(value), bf

    @property
    def needs_pretokenization(self) -> bool:
        return self is SubwordModel.WORD

    @property
    def is_bytes_only(self) -> bool:
        return self is SubwordModel.BYTES

    @property
    def has_vocab(self) -> bool:
        """Whether this subword model produces / requires a learned vocabulary file."""
        return self not in (SubwordModel.NONE, SubwordModel.BYTES)

    @property
    def uses_sentencepiece(self) -> bool:
        """Whether the model is trained / encoded with SentencePiece (word/bpe/unigram/char)."""
        return self in (SubwordModel.WORD, SubwordModel.BPE,
                        SubwordModel.UNIGRAM, SubwordModel.CHAR)


def has_vocab(model: Optional[SubwordModel]) -> bool:
    """``None``-tolerant wrapper for ``SubwordModel.has_vocab``."""
    return model is not None and model.has_vocab


def is_bytes_only(model: Optional[SubwordModel]) -> bool:
    return model is SubwordModel.BYTES


def is_no_model(model: Optional[SubwordModel]) -> bool:
    """True for the 'no subword model' sentinel — ``None`` or ``SubwordModel.NONE``."""
    return model is None or model is SubwordModel.NONE
