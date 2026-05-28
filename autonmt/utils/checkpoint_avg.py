"""Checkpoint averaging.

Averaging the parameters of the last N best checkpoints is the standard NMT
trick from Vaswani et al. (2017) §5.4: it typically gains 0.5-2 BLEU over the
single best checkpoint, with no extra training cost. AutoNMT doesn't run it
automatically — you call ``average_checkpoints(...)`` on the files you want
and then load the result like any other Lightning ``.pt``.
"""
from __future__ import annotations

import os
from typing import Iterable, Union

import torch

from autonmt.utils.logger import get_logger

log = get_logger(__name__)

PathLike = Union[str, os.PathLike]


def average_checkpoints(checkpoint_paths: Iterable[PathLike],
                        output_path: PathLike) -> str:
    """Average the ``state_dict`` of N Lightning checkpoints into one ``.pt``.

    Non-state_dict entries (optimizer state, hyper-parameters, epoch...) are
    copied from the *first* checkpoint so the result is loadable by
    :meth:`AutonmtTranslator.load_checkpoint`. Dtypes are preserved.
    """
    paths = [str(p) for p in checkpoint_paths]
    if not paths:
        raise ValueError("checkpoint_paths must contain at least one path")

    log.info(f"=> [Avg]: Averaging {len(paths)} checkpoint(s)")
    base = torch.load(paths[0], map_location="cpu", weights_only=False)
    base_sd = base.get("state_dict", base)
    accum = {k: v.detach().clone().float() for k, v in base_sd.items()}

    for p in paths[1:]:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        missing = set(accum) - set(sd)
        if missing:
            raise ValueError(f"Checkpoint {p} is missing keys: {sorted(missing)[:5]}...")
        for k in accum:
            accum[k] += sd[k].detach().float()

    n = len(paths)
    averaged = {k: (v / n).to(base_sd[k].dtype) for k, v in accum.items()}

    if "state_dict" in base:
        base["state_dict"] = averaged
    else:
        base = averaged

    torch.save(base, output_path)
    log.info(f"=> [Avg]: Wrote {output_path}")
    return str(output_path)
