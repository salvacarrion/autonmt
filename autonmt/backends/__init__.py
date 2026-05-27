"""Lazy re-export of concrete translators.

``from autonmt.backends import AutonmtTranslator`` keeps working, but importing
``autonmt.backends.base.config`` doesn't pull in torch / pytorch_lightning /
comet_ml.
"""
__all__ = ["AutonmtTranslator", "FairseqTranslator", "HuggingFaceTranslator"]


def __getattr__(name):
    if name == "AutonmtTranslator":
        from autonmt.backends.autonmt.translator import AutonmtTranslator
        return AutonmtTranslator
    if name == "FairseqTranslator":
        from autonmt.backends.fairseq.translator import FairseqTranslator
        return FairseqTranslator
    if name == "HuggingFaceTranslator":
        from autonmt.backends.huggingface.translator import HuggingFaceTranslator
        return HuggingFaceTranslator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
