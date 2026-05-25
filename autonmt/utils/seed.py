"""Process-wide seeding helper.

Imports torch / pytorch_lightning lazily so the rest of the framework keeps
working on a torch-free install (e.g. preprocessing-only environments).
"""
import time

from autonmt.utils.logger import get_logger

log = get_logger(__name__)


def manual_seed(seed=None, use_deterministic_algorithms: bool = False) -> int:
    """Seed Python ``random``, NumPy, Torch and Lightning together.

    Returns the resolved seed (so callers that pass ``None`` can log it).
    """
    import random
    import numpy as np
    import torch
    import pytorch_lightning as pl

    seed = seed if seed is not None else int(time.time()) % 2**32

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed)

    # See https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    torch.use_deterministic_algorithms(use_deterministic_algorithms)

    log.info(f"\t- [INFO]: Testing random seed ({seed}):")
    log.info(f"\t\t- random: {random.random()}")
    log.info(f"\t\t- numpy: {np.random.rand(1)}")
    log.info(f"\t\t- torch: {torch.rand(1)}")
    return seed
