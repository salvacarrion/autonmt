"""Per-run report *schema*: the dict each backend emits, decoupled from presentation.

``build_run_report`` assembles the per-run dict the translator emits — the schema
lives here (not in ``BaseTranslator``) so changes to the report shape don't drag
the toolkit code along, and not in ``report.py`` so the backends don't import the
plotting/presentation layer just to describe their output.

The :class:`Report` presentation layer (``report.py``) consumes the list of these
dicts; it never builds them.
"""
from __future__ import annotations

import dataclasses
import datetime
from dataclasses import dataclass
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# RunMetadata: backend-supplied keys for build_run_report()
# ---------------------------------------------------------------------------

@dataclass
class RunMetadata:
    """Backend-supplied metadata that feeds into the per-run report dict.

    All fields are optional — a backend can return ``RunMetadata()`` if it
    has nothing to report (the keys will be omitted from the final dict).
    ``vocab__size`` is ``Any`` because some backends report a single int
    (shared vocab) and others a tuple/string (asymmetric src/tgt sizes).
    """
    model__architecture: Optional[str] = None
    model__total_params: Optional[int] = None
    model__trainable_params: Optional[int] = None
    model__no_trainable_params: Optional[int] = None
    model__dtype: Optional[str] = None
    vocab__subword_model: Optional[str] = None
    vocab__size: Optional[Any] = None
    vocab__merged: Optional[bool] = None
    vocab__lang_pair: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        """Drop ``None`` fields so the report dict only carries known keys."""
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}


# ---------------------------------------------------------------------------
# Per-run report dict construction
# ---------------------------------------------------------------------------

def build_run_report(engine: str, run_name: str, eval_ds, config: Dict,
                     translations: Dict, metadata: RunMetadata,
                     train_ds=None) -> Dict:
    """Assemble the dict the translator emits per (run, eval_ds) pair.

    The schema is owned by the reporting layer; the translator only supplies
    a ``RunMetadata`` block (toolkit-specific keys like model architecture,
    param counts, vocab info) plus the cross-cutting dataset / config /
    translations fields. Missing metadata fields are dropped so a minimal
    backend (no model/vocab info) still produces a valid report.

    ``train_dataset`` and ``train__lang_pair`` are taken from ``train_ds``
    when available (the trained model variant), falling back to
    ``metadata.vocab__lang_pair`` if the backend supplied it.
    """
    meta = metadata.as_dict()
    train_dataset = train_ds.dataset_name if train_ds is not None else None
    train_lang_pair = meta.get("vocab__lang_pair")

    base = {
        "engine": engine,
        "run_name": run_name,
        "eval_datetime": str(datetime.datetime.now()),

        # NOTE: train_subsets / test_subsets can override the effective
        # language pair seen by the model. The bare vocab lang_pair is only
        # correct when no subset filter rewrites the pair.
        "train__lang_pair": train_lang_pair,
        "test__lang_pair": f"{eval_ds.src_lang}-{eval_ds.tgt_lang}",

        "train_dataset": train_dataset,
        "test_dataset": eval_ds.dataset_name,
        "test_dataset_full": f"{eval_ds.dataset_name}__{eval_ds.src_lang}-{eval_ds.tgt_lang}",

        "translations": translations,
        "config": config,
    }
    # Backend metadata wins on duplicate keys — there shouldn't be any, but
    # be explicit so future schema growth doesn't silently shadow.
    base.update(meta)
    return base
