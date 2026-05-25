"""Report orchestration: turn raw scores into JSON+CSV artifacts and figures.

This module's job is to wire together :mod:`autonmt.utils.fileio` and
:mod:`autonmt.reporting.figures`. It owns no plotting or schema logic of its own
beyond the very thin ``scores_to_dataframe`` / ``summarize_scores`` transforms.

The ``build_run_report`` helper assembles the per-run dict the translator
emits — the schema lives here (not in ``BaseTranslator``) so changes to the
report shape don't drag the toolkit code along.
"""
from __future__ import annotations

import datetime
import os
from typing import Dict, Optional

import pandas as pd

from autonmt.reporting import figures
from autonmt.utils import fileio
from autonmt.reporting.plots import DEFAULT_STYLE, PlotStyle


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

def build_run_report(engine: str, run_name: str, model, eval_ds,
                     src_vocab, trg_vocab, config: Dict,
                     translations: Dict) -> Dict:
    """Assemble the dict the translator emits per (run, eval_ds) pair.

    Lives here (not in ``BaseTranslator``) so the report schema is owned by
    the reporting layer. ``translations`` carries the parsed beam → metric
    scores; everything else is metadata the report consumer (CSV / plots /
    leaderboard) joins on.
    """
    total_params, trainable_params, no_trainable_params = model.count_parameters()
    assert src_vocab.subword_model == trg_vocab.subword_model

    if len(src_vocab) != len(trg_vocab):
        vocab_size = f"{len(src_vocab)}/{len(trg_vocab)}"
    else:
        vocab_size = f"{len(src_vocab)}"

    lang_pair = f"{src_vocab.lang}-{trg_vocab.lang}"
    return {
        "engine": engine,
        "run_name": run_name,
        "eval_datetime": str(datetime.datetime.now()),

        "model__architecture": model.architecture,
        "model__trainable_params": trainable_params,
        "model__no_trainable_params": no_trainable_params,
        "model__total_params": total_params,
        "model__dtype": str(model.dtype),

        "vocab__subword_model": src_vocab.subword_model,
        "vocab__size": vocab_size,
        "vocab__merged": "no-specified",
        "vocab__lang_pair": lang_pair,

        # NOTE: train_subsets / test_subsets can override the effective language
        # pair seen by the model. The bare vocab.lang is only correct when no
        # subset filter rewrites the pair.
        "train__lang_pair": lang_pair,
        "test__lang_pair": f"{eval_ds.src_lang}-{eval_ds.trg_lang}",

        "train_dataset": "no-specified",
        "test_dataset": eval_ds.dataset_name,
        "test_dataset_full": f"{eval_ds.dataset_name}__{eval_ds.src_lang}-{eval_ds.trg_lang}",

        "translations": translations,
        "config": config,
    }
