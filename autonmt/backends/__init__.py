"""Lazy re-export of concrete translators.

``from autonmt.backends import AutonmtTranslator`` keeps working, but importing
``autonmt.backends._base.config`` doesn't pull in torch / pytorch_lightning.
"""
__all__ = ["AutonmtTranslator", "FairseqTranslator", "HuggingFaceTranslator",
           "HuggingFaceCausalLM", "HuggingFaceMaskedLM", "AutonmtCausalLM", "AutonmtMaskedLM"]


def __getattr__(name):
    if name == "AutonmtTranslator":
        from autonmt.backends.autonmt.translation_engine import AutonmtTranslator
        return AutonmtTranslator
    if name == "AutonmtCausalLM":
        from autonmt.backends.autonmt.lm import AutonmtCausalLM
        return AutonmtCausalLM
    if name == "AutonmtMaskedLM":
        from autonmt.backends.autonmt.lm import AutonmtMaskedLM
        return AutonmtMaskedLM
    if name in ("HuggingFaceCausalLM", "HuggingFaceMaskedLM"):
        from autonmt.backends.huggingface import lm as _hf_lm
        return getattr(_hf_lm, name)
    if name == "FairseqTranslator":
        from autonmt.backends.fairseq.translation_engine import FairseqTranslator
        return FairseqTranslator
    if name == "HuggingFaceTranslator":
        from autonmt.backends.huggingface.translation_engine import HuggingFaceTranslator
        return HuggingFaceTranslator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
