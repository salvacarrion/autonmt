"""Lazy re-export of :class:`AutonmtTranslator` so ``from autonmt.toolkits import
AutonmtTranslator`` keeps working, but ``from autonmt.toolkits.config import …``
no longer pulls in torch / pytorch_lightning / comet_ml.
"""
__all__ = ["AutonmtTranslator"]


def __getattr__(name):
    if name == "AutonmtTranslator":
        from autonmt.toolkits.autonmt import AutonmtTranslator
        return AutonmtTranslator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
