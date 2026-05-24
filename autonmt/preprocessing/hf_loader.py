"""Materialise a HuggingFace parallel-corpus dataset on disk in the AutoNMT layout.

Convenience for new users so they can do::

    from autonmt.preprocessing.hf_loader import download_hf_dataset

    download_hf_dataset(
        hf_id="bentrevett/multi30k",
        base_path="datasets",
        dataset_name="multi30k",
        lang_pair="de-en",
        src_field="de",
        trg_field="en",
    )

…and immediately point a ``DatasetBuilder`` at ``datasets/multi30k``.

The ``datasets`` package is an optional dependency; this module imports it
lazily so the rest of the framework keeps working without it installed.
"""
import os
from typing import Optional

from autonmt.bundle.logger import get_logger
from autonmt.bundle.utils import make_dir, write_file_lines

log = get_logger(__name__)


def _require_datasets():
    try:
        import datasets  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required for HF loading. "
            "Install with: pip install datasets"
        ) from e


def download_hf_dataset(
    hf_id: str,
    base_path: str,
    dataset_name: str,
    lang_pair: str,
    src_field: str,
    trg_field: str,
    *,
    size_name: str = "original",
    hf_config: Optional[str] = None,
    split_map: Optional[dict] = None,
    max_train_lines: Optional[int] = None,
    force_overwrite: bool = False,
) -> str:
    """Download a HF dataset and write it as ``train/val/test`` splits on disk.

    Args:
        hf_id: HuggingFace dataset identifier (e.g. ``"bentrevett/multi30k"``).
        base_path: Root where AutoNMT writes datasets.
        dataset_name: Folder name under ``base_path``.
        lang_pair: ``"src-trg"`` (e.g. ``"de-en"``); also drives the file extensions.
        src_field, trg_field: Keys inside each HF example holding the parallel text.
            For nested dicts (e.g. ``{"translation": {"de": ..., "en": ...}}``) pass
            the leaf key — see ``split_map`` for the column path.
        size_name: Sub-folder (AutoNMT supports multiple sizes per dataset).
        hf_config: Optional config name to pass to ``load_dataset``.
        split_map: Maps AutoNMT split names to HF split names, e.g.
            ``{"train": "train", "val": "validation", "test": "test"}``.
            Default tries the canonical names and falls back to whatever
            the dataset exposes.
        max_train_lines: Truncate the train split (useful for smoke tests).
        force_overwrite: Re-download/rewrite files even if they exist.

    Returns:
        Path to the AutoNMT-style ``<base_path>/<dataset_name>/<lang_pair>/<size>/``
        directory that was populated.
    """
    _require_datasets()
    from datasets import load_dataset

    src_lang, trg_lang = lang_pair.split("-")
    split_map = split_map or {"train": "train", "val": "validation", "test": "test"}

    out_dir = os.path.join(base_path, dataset_name, lang_pair, size_name)
    splits_dir = os.path.join(out_dir, "data", "1_splits")
    make_dir(splits_dir)

    log.info(f"=> Loading HF dataset '{hf_id}'"
             + (f" (config={hf_config})" if hf_config else "") + " ...")
    ds = load_dataset(hf_id, hf_config) if hf_config else load_dataset(hf_id)

    for autonmt_split, hf_split in split_map.items():
        if hf_split not in ds:
            log.warning(f"\t- Skipping '{autonmt_split}': HF split '{hf_split}' not found")
            continue

        src_file = os.path.join(splits_dir, f"{autonmt_split}.{src_lang}")
        trg_file = os.path.join(splits_dir, f"{autonmt_split}.{trg_lang}")
        if not force_overwrite and os.path.exists(src_file) and os.path.exists(trg_file):
            log.info(f"\t- '{autonmt_split}' already exists, skipping")
            continue

        log.info(f"\t- Writing '{autonmt_split}' ({src_lang}/{trg_lang})...")
        src_lines = _extract_field(ds[hf_split], src_field)
        trg_lines = _extract_field(ds[hf_split], trg_field)
        if len(src_lines) != len(trg_lines):
            raise ValueError(
                f"Misaligned parallel data in '{hf_split}': "
                f"{len(src_lines)} src vs {len(trg_lines)} trg")

        if autonmt_split == "train" and max_train_lines:
            src_lines = src_lines[:max_train_lines]
            trg_lines = trg_lines[:max_train_lines]

        write_file_lines(src_lines, filename=src_file, insert_break_line=True)
        write_file_lines(trg_lines, filename=trg_file, insert_break_line=True)
        log.info(f"\t  ↳ {len(src_lines):,} pairs written")

    log.info(f"=> Done. AutoNMT layout ready at: {out_dir}")
    return out_dir


def _extract_field(hf_split, field: str):
    """Pull ``field`` from each row. Supports flat columns or ``translation.<lang>``
    nested structures (the convention for most HF MT datasets)."""
    if field in hf_split.column_names:
        return list(hf_split[field])
    if "translation" in hf_split.column_names:
        return [row[field] for row in hf_split["translation"]]
    raise KeyError(
        f"Field {field!r} not found. Available columns: {hf_split.column_names}. "
        f"For nested datasets, ensure the field exists under 'translation'."
    )
