"""Lazy re-export of concrete translators.

``from autonmt.backends import AutonmtTranslator`` keeps working, but importing
``autonmt.backends._base.config`` doesn't pull in torch / pytorch_lightning.
"""
__all__ = ["AutonmtTranslator", "FairseqTranslator", "HuggingFaceTranslator"]


def __getattr__(name):
    if name == "AutonmtTranslator":
        from autonmt.backends.autonmt.translation_engine import AutonmtTranslator
        return AutonmtTranslator
    if name == "FairseqTranslator":
        from autonmt.backends.fairseq.translation_engine import FairseqTranslator
        return FairseqTranslator
    if name == "HuggingFaceTranslator":
        from autonmt.backends.huggingface.translation_engine import HuggingFaceTranslator
        return HuggingFaceTranslator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
