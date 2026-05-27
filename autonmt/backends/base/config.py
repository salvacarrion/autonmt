"""Typed configuration objects for :meth:`BaseTranslator.fit` / ``predict``.

These are *optional*. The existing keyword-argument call style keeps working —
``fit(train_ds, batch_size=32, ...)`` is equivalent to
``fit(train_ds, config=FitConfig(batch_size=32))``. When both are provided,
the explicit kwargs win on a per-key basis.

The motivation is to (a) document the expected argument shape in one place,
(b) make typos in arg names raise at the call site instead of silently
disappearing into ``**kwargs``, and (c) let users build configs programmatically.
"""
from dataclasses import dataclass, fields, asdict
from typing import Any, Callable, Iterable, List, Optional, Sequence, Union

from autonmt.utils.enums import EvalMode


class _Unset:
    """Sentinel for arguments that were not passed by the caller."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "<UNSET>"

    def __bool__(self):
        return False


UNSET = _Unset()


@dataclass
class FitConfig:
    max_tokens: Optional[int] = None
    batch_size: int = 128
    max_epochs: int = 1
    patience: Optional[int] = None
    optimizer: str = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 0
    gradient_clip_val: float = 0.0
    accumulate_grad_batches: int = 1
    criterion: str = "cross_entropy"
    monitor: str = "val_loss"
    devices: Union[str, int, List[int]] = "auto"
    accelerator: str = "auto"
    num_workers: int = 0
    seed: Optional[int] = None
    force_overwrite: bool = False
    use_bucketing: bool = False
    save_best: bool = True
    save_last: bool = False
    print_samples: int = 0
    strategy: str = "auto"

    def as_kwargs(self) -> dict:
        return asdict(self)


@dataclass
class PredictConfig:
    metrics: Optional[Iterable[str]] = None
    beams: Optional[Sequence[int]] = None
    max_len_a: float = 1.2
    max_len_b: int = 50
    max_tokens: Optional[int] = None
    batch_size: int = 64
    devices: Union[str, int, List[int]] = "auto"
    accelerator: str = "auto"
    num_workers: int = 0
    load_checkpoint: Optional[str] = None
    preprocess_fn: Optional[Callable] = None
    eval_mode: Union[str, EvalMode] = "same"
    force_overwrite: bool = False
    # Optional decoder instance (BaseSearch subclass). When None, the backend
    # picks a default — AutonmtTranslator falls back to GreedySearch / BeamSearch
    # depending on beam_width. Pass an instance of MultinomialSampling /
    # TopKSampling / TopPSampling / custom BaseSearch to override.
    decoder: Optional[Any] = None

    def as_kwargs(self) -> dict:
        return asdict(self)


def merge_config(config, defaults_cls, explicit_kwargs: dict) -> dict:
    """Materialise ``config`` (or a fresh ``defaults_cls()``) into a kwargs dict,
    then overlay ``explicit_kwargs`` so caller-supplied values win.

    Validates that every explicit kwarg matches a field on ``defaults_cls``.
    Extra kwargs (those passed via ``**kwargs`` for the underlying toolkit) are
    returned untouched in a separate dict.
    """
    if config is None:
        config = defaults_cls()
    elif not isinstance(config, defaults_cls):
        raise TypeError(f"config must be a {defaults_cls.__name__} or None, got {type(config).__name__}")

    valid = {f.name for f in fields(defaults_cls)}
    known, extra = {}, {}
    for k, v in explicit_kwargs.items():
        if v is UNSET:
            continue  # caller didn't pass this; let the config value stand
        if k in valid:
            known[k] = v
        else:
            extra[k] = v

    merged = config.as_kwargs()
    merged.update(known)
    return merged, extra
