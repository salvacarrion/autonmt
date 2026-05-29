"""Report orchestration: turn raw scores into JSON+CSV artifacts and figures.

This module's job is to wire together :mod:`autonmt.utils.fileio` and
:mod:`autonmt.reporting.figures`. It owns no plotting or schema logic of its own
beyond the very thin ``scores_to_dataframe`` / ``summarize_scores`` transforms.

The ``build_run_report`` helper assembles the per-run dict the translator
emits — the schema lives here (not in ``BaseTranslator``) so changes to the
report shape don't drag the toolkit code along.
"""
from __future__ import annotations

import dataclasses
import datetime
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from autonmt.reporting import figures
from autonmt.utils import fileio
from autonmt.reporting.plots import DEFAULT_STYLE, PlotStyle


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
# Score transforms
# ---------------------------------------------------------------------------

def scores_to_dataframe(scores) -> pd.DataFrame:
    """Flatten the nested ``list[list[dict]]`` shape returned by ``predict()``."""
    rows = [pd.json_normalize(eval_scores)
            for model_scores in scores for eval_scores in model_scores]
    if not rows:
        raise ValueError("=> [Report]: No scores were given")
    return pd.concat(rows)


_DEFAULT_SUMMARY_COLS = (
    "train_dataset", "train__lang_pair", "test_dataset", "test__lang_pair",
    "vocab__subword_model", "vocab__size",
    "model__architecture", "model__total_params",
)


def summarize_scores(df_report: pd.DataFrame,
                     default_cols=_DEFAULT_SUMMARY_COLS,
                     ref_metric: str = "bleu") -> pd.DataFrame:
    """Keep identifying columns plus every column that matches ``ref_metric``."""
    selected = [c for c in df_report.columns if c in default_cols or ref_metric in c]
    return df_report[selected]


def format_summary_table(df_summary: pd.DataFrame) -> str:
    """Render the summary DataFrame for terminal printing.

    Column names are kept verbatim — they're the JSON/CSV keys downstream
    code and users reference, and keeping the ``__`` / ``.`` separators makes
    column boundaries visible without explicit borders. Layout is delegated
    to ``DataFrame.to_string`` (width-aware) with a unicode rule above and
    below.
    """
    if df_summary.empty:
        return "(empty report)"

    df = df_summary.copy().map(_fmt_cell)
    body = df.to_string(index=False, na_rep="-")
    rule = "─" * max(len(line) for line in body.splitlines())
    return f"{rule}\n{body}\n{rule}"


def _fmt_cell(v):
    """Two-decimal floats, thousand-separated ints; everything else unchanged."""
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        return f"{v:,.2f}"
    if isinstance(v, int):
        return f"{v:,}"
    return v


# ---------------------------------------------------------------------------
# Report orchestration
# ---------------------------------------------------------------------------

def _ensure_report_dirs(output_path):
    reports_path = os.path.join(output_path, "reports")
    plots_path = os.path.join(output_path, "plots")
    fileio.make_dir([reports_path, plots_path])
    return reports_path, plots_path


def generate_report(scores, output_path, plot_metric: Optional[str] = None,
                    style: PlotStyle = DEFAULT_STYLE, **figure_kwargs):
    """Write JSON/CSV artifacts under ``<output_path>/reports/`` and, if
    ``plot_metric`` is provided, a model-comparison figure under ``plots/``.

    ``figure_kwargs`` are forwarded to :func:`figures.plot_model_comparison`
    (xlabel, ylabel, title, group_label_fn, legend_label_fn, ...).
    Returns ``(df_report, df_summary)``.
    """
    if not scores:
        raise ValueError("No scores were given")

    reports_path, plots_path = _ensure_report_dirs(output_path)

    df_report = scores_to_dataframe(scores)
    df_summary = summarize_scores(df_report)

    fileio.save_json(scores, os.path.join(reports_path, "report.json"))
    df_report.to_csv(os.path.join(reports_path, "report.csv"), index=False)
    df_summary.to_csv(os.path.join(reports_path, "report_summary.csv"), index=False)

    if plot_metric:
        figures.plot_model_comparison(df_report, plots_path, metric=plot_metric,
                                      style=style, **figure_kwargs)

    return df_report, df_summary


def generate_sweep_report(data: pd.DataFrame, output_path: str, x: str,
                          y_left, y_right=None,
                          prefix: str = "", save_csv: bool = False,
                          style: PlotStyle = DEFAULT_STYLE, **figure_kwargs):
    """Sweep-style report: optional CSV dump + line plot via :func:`figures.plot_metric_sweep`."""
    reports_path, plots_path = _ensure_report_dirs(output_path)

    if save_csv:
        data.to_csv(os.path.join(reports_path, f"{prefix}_sweep.csv"), index=False)

    figures.plot_metric_sweep(data, plots_path, x=x, y_left=y_left, y_right=y_right,
                              prefix=prefix, style=style, **figure_kwargs)


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
