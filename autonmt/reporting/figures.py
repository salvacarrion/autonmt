"""Domain-specific figures for NMT reports.

These are the figures the framework knows how to build by default. Each one
takes already-prepared data (a DataFrame or a Dataset) plus an output directory
and a :class:`PlotStyle`, and emits one figure via :mod:`autonmt.reporting.plots`.

This module is the *only* place where score-schema or dataset-layout knowledge
meets the plot library. ``reporting/plots.py`` stays domain-free.
"""
from __future__ import annotations

import os
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from autonmt.utils import fileio
from autonmt.reporting import plots
from autonmt.datasets import stats
from autonmt.utils.logger import get_logger
from autonmt.reporting.plots import DEFAULT_STYLE, PlotStyle

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def _default_group_label(row) -> str:
    return (f"{row['vocab__subword_model']} - {row['vocab__size']}\n"
            f"Tr: {row['train_dataset'].replace('_', ' ')}").title()


def _prettify(text: str) -> str:
    return text.title().replace('_', ' ')


def plot_model_comparison(df_report: pd.DataFrame, out_dir: str, metric: str,
                          title: str = "Model comparison",
                          xlabel: str = "MT Models", ylabel: str = "BLEU Score",
                          group_label_fn: Optional[Callable[[pd.Series], str]] = None,
                          legend_label_fn: Optional[Callable[[str], str]] = None,
                          style: PlotStyle = DEFAULT_STYLE):
    """Grouped bar plot comparing one metric across runs.

    Expects ``df_report`` produced by :func:`autonmt.reporting.report.scores_to_dataframe`,
    i.e. with columns ``train_dataset``, ``test_dataset``, ``vocab__subword_model``,
    ``vocab__size`` and ``<metric>``.
    """
    required = {"train_dataset", "test_dataset", metric}
    missing = required - set(df_report.columns)
    if missing:
        raise ValueError(f"Missing columns in df_report: {sorted(missing)}")

    log.info("=> Plotting model comparison...")
    log.warning("\t- [WARNING]: Matplotlib might miss some images if the loop is too fast")

    df = df_report.copy()
    df["bar_group_name"] = df.apply(group_label_fn or _default_group_label, axis=1)

    # Only legend on test_dataset when there's actually variation (train != test on
    # at least one row); otherwise the legend would be a single redundant entry.
    same_count = int((df['train_dataset'] == df['test_dataset']).values.sum())
    has_variation = same_count != len(df['train_dataset'])
    legend_col = "test_dataset" if has_variation else None
    if legend_col:
        df[legend_col] = df[legend_col].map(legend_label_fn or _prettify)

    plots.catplot(
        data=df, x="bar_group_name", y=metric, hue=legend_col,
        out_dir=out_dir, fname=f"plot__{metric}",
        title=title, xlabel=xlabel, ylabel=ylabel,
        value_format="{:.2f}", legend_loc="lower right",
        style=style.merge(figsize=(16, 8), font_scale=1.5),
    )


# ---------------------------------------------------------------------------
# Metric sweep
# ---------------------------------------------------------------------------

def plot_metric_sweep(data: pd.DataFrame, out_dir: str, x: str,
                      y_left, y_right=None,
                      xlabel: str = "Vocab sizes",
                      ylabel_left: Optional[str] = None,
                      ylabel_right: Optional[str] = None,
                      title: str = "Vocabularies report",
                      legend_loc: str = "upper left",
                      prefix: str = "",
                      style: PlotStyle = DEFAULT_STYLE):
    """Line plot of a metric across a sweep variable (vocab size, model size, ...).

    ``y_left`` / ``y_right`` may be ``str`` or ``(column, hue_column)`` tuples.
    """
    y_left_col, y_left_hue = y_left if isinstance(y_left, tuple) else (y_left, None)
    y_right_col, y_right_hue = (y_right if isinstance(y_right, tuple) else (y_right, None))

    for col, name in [(y_left_col, "y_left"), (y_right_col, "y_right")]:
        if col is not None and col not in data.columns:
            raise ValueError(f"'{col}' (passed as {name}) was not found in the given dataframe")

    log.info("=> Plotting metric sweep...")

    fname = f"{prefix}sweep__{y_left_col}{'_' + y_right_col if y_right_col else ''}".lower()
    plots.lineplot(
        data=data, x=x,
        y_left=y_left_col, y_left_hue=y_left_hue,
        y_right=y_right_col, y_right_hue=y_right_hue,
        out_dir=out_dir, fname=fname,
        title=title, xlabel=xlabel,
        ylabel_left=ylabel_left or y_left_col,
        ylabel_right=ylabel_right or y_right_col,
        legend_loc=legend_loc,
        style=style.merge(figsize=(8, 6)),
    )


# ---------------------------------------------------------------------------
# Dataset diagnostics
# ---------------------------------------------------------------------------

def _ds_title(ds) -> str:
    ds_name, lang_pair, _ = ds.id()
    return f"{ds_name.title()} ({lang_pair}; {ds.subword_model}; {ds.vocab_size})"


def _ds_fname_suffix(ds) -> str:
    ds_name, lang_pair, ds_size_name = ds.id()
    vocab = f"_{ds.vocab_size}" if ds.vocab_size else ""
    s = f"{ds_name}_{ds_size_name}_{lang_pair}__{ds.subword_model}{vocab}"
    return s.lower().replace('/', '_')


def _maybe_titled(ds_title: str, body: str, add_dataset_title: bool) -> str:
    return f"{ds_title}:\n{body}" if add_dataset_title else body


def _plot_sentence_length_histograms(ds, out_dir, suffix, add_dataset_title, style):
    """One histogram per (split, lang) and a per-split tokens-per-sentence DataFrame."""
    ds_title = _ds_title(ds)
    rows = []
    for fname in ds.get_split_fnames():
        split_name, split_lang = fname.split('.')
        tokens = np.array(stats.count_tokens_per_sentence(filename=ds.get_encoded_path(fname)))

        row = stats.basic_stats(tokens, prefix="")
        row.update({"split": split_name, "lang": split_lang})
        rows.append((fname, row))

        df = pd.DataFrame(tokens, columns=["frequency"])
        title = f"Sentence length distribution ({split_name.title()} - {split_lang})"
        plots.histogram(
            data=df, x="frequency",
            out_dir=out_dir, fname=f"sent_distr_{split_name}_{split_lang}__{suffix}".lower(),
            title=_maybe_titled(ds_title, title, add_dataset_title),
            xlabel="Tokens per sentence", ylabel="Frequency", bins=100,
            style=style.merge(figsize=(6, 4), font_scale=1.5),
        )
    return rows


def _plot_split_sizes(ds, out_dir, suffix, split_stats, add_dataset_title, style):
    df = pd.DataFrame([row for _, row in split_stats])
    ds_title = _ds_title(ds)

    plots.catplot(
        data=df, x="split", y="total_sentences", hue="lang",
        out_dir=out_dir, fname=f"split_size_sent__{suffix}".lower(),
        title=_maybe_titled(ds_title, "Split sizes (by number of sentences)", add_dataset_title),
        xlabel="Dataset partitions", ylabel="Num. of sentences",
        style=style.merge(figsize=(6, 4)),
    )

    if str(ds.subword_model) not in {"None", "none"}:
        body = f"Split sizes (by number of tokens - {str(ds.subword_model).title()})"
        plots.catplot(
            data=df, x="split", y="total_tokens", hue="lang",
            out_dir=out_dir, fname=f"split_size_tok__{suffix}".lower(),
            title=_maybe_titled(ds_title, body, add_dataset_title),
            xlabel="Dataset partitions", ylabel="Num. of tokens",
            style=style.merge(figsize=(6, 4)),
        )


_SUBWORD_DISPLAY = {"word": "Words", "bpe": "BPE", "char": "Chars", "bytes": "Bytes"}


def _plot_vocab_distribution(ds, merge_vocabs, out_dir, suffix,
                             vocab_top_k: Sequence[int], style):
    if str(ds.subword_model) in {"None", "none"}:
        return

    src_lang, trg_lang = ds.id()[1].split('-')
    lang_files = [f"{src_lang}-{trg_lang}"] if merge_vocabs else [src_lang, trg_lang]
    vocab_path = ds.get_vocab_path()

    for lang_file in lang_files:
        vocab_freq_path = os.path.join(vocab_path, lang_file + ".vocabf")
        with open(vocab_freq_path, 'r') as f:
            rows = [line.split('\t') for line in f.readlines()]
        df = pd.DataFrame(rows, columns=["token", "frequency"])
        df["frequency"] = df["frequency"].apply(lambda x: int(x.strip())).astype(int)
        df = df.sort_values(by='frequency', ascending=False, na_position='last')

        for top_k in vocab_top_k:
            df_sample = df.sample(n=top_k, random_state=1).sort_values(
                by='frequency', ascending=False)
            display = _SUBWORD_DISPLAY.get(str(ds.subword_model), str(ds.subword_model))
            plots.barplot(
                data=df_sample, x="token", y="frequency",
                out_dir=out_dir,
                fname=f"vocab_distr_{lang_file}_sampled{top_k}__{suffix}".lower(),
                title=f"Vocabulary distribution ({display} - {len(df):,})",
                xlabel="Tokens", ylabel="Frequency",
                style=style.merge(figsize=(12, 8), font_scale=2.5),
            )


def plot_dataset_diagnostics(ds, *, merge_vocabs: bool,
                             vocab_top_k: Iterable[int] = (256,),
                             add_dataset_title: bool = True,
                             style: PlotStyle = DEFAULT_STYLE):
    """Emit the standard diagnostic figures for one preprocessed dataset.

    Writes the per-split stats JSON alongside the plots, and returns nothing.
    Output directory is ``ds.get_plots_path()``.
    """
    out_dir = ds.get_plots_path()
    fileio.make_dir(out_dir)
    suffix = _ds_fname_suffix(ds)
    log.info(f"\t- Creating plots for: {ds.id2(as_path=True)}")

    log.info("\t\t- Creating 'Sentence length distribution' plots...")
    rows = _plot_sentence_length_histograms(ds, out_dir, suffix, add_dataset_title, style)

    split_stats = {fname: row for fname, row in rows}
    fileio.save_json(split_stats, os.path.join(out_dir, f"stats__{suffix}.json"))

    log.info("\t\t- Creating 'Split sizes' plots...")
    _plot_split_sizes(ds, out_dir, suffix, rows, add_dataset_title, style)

    if str(ds.subword_model) not in {"None", "none"}:
        log.info("\t\t- Creating 'Vocabulary distribution' plots...")
        _plot_vocab_distribution(ds, merge_vocabs, out_dir, suffix,
                                 tuple(vocab_top_k), style)
