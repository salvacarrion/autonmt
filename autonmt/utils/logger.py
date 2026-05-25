import logging
import os

_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_configured = False


def configure(level=None, fmt=_DEFAULT_FORMAT):
    """Attach a single StreamHandler to the autonmt root logger.

    Idempotent. Honours $AUTONMT_LOG_LEVEL when ``level`` is None.
    """
    global _configured
    root = logging.getLogger("autonmt")
    if _configured:
        return root

    if level is None:
        level = os.environ.get("AUTONMT_LOG_LEVEL", "INFO").upper()

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _configured = True
    return root


def get_logger(name):
    """Return a module-level logger and ensure the package handler is set up."""
    configure()
    return logging.getLogger(name)
