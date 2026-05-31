"""Reporting: two data-object reporters over the plot primitives.

- :class:`Report` wraps the nested ``scores`` returned by ``predict()`` and
  presents a *semantic* API — ask for the chart you want (``plot_comparison``,
  ``plot_sweep``, ``plot_matrix``) by the metric you care about, not by the
  seaborn primitive or the long ``translations.beam5.sacrebleu_bleu_score``
  column name. Metric names are resolved generically against the columns that
  actually exist, so HuggingFace (``hg_*``), COMET, BERTScore, etc. all work
  without a hardcoded table.
- :class:`DatasetReport` wraps a :class:`~autonmt.datasets.dataset.Dataset` and
  emits the standard corpus-diagnostics figures (length distributions, split
  sizes, vocabulary distribution).

Both render through the :class:`~autonmt.reporting.plots.BasePlot` hierarchy and
own all the score-schema / dataset-layout knowledge; ``plots.py`` stays
domain-free. The per-run dict *schema* the backends emit lives in
:mod:`autonmt.reporting.schema`, upstream of any ``Report``.
"""
from __future__ import annotations

import os
import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from autonmt.datasets import stats
from autonmt.utils import fileio
from autonmt.utils.logger import get_logger
from autonmt.reporting.plots import (
    DEFAULT_STYLE, PlotStyle, BarPlot, CatPlot, HeatmapPlot, HistogramPlot, LinePlot,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Score transforms (module-level: used internally + by the "manual" example)
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
# Helpers: metric resolution + labelling
# ---------------------------------------------------------------------------

_BEAM_RE = re.compile(r"beam(\d+)", re.IGNORECASE)


def _segment_after_beam(col: str) -> str:
    """``translations.beam5.sacrebleu_bleu_score`` -> ``sacrebleu_bleu_score``."""
    parts = col.split(".")
    for i, p in enumerate(parts):
        if p.lower().startswith("beam"):
            return ".".join(parts[i + 1:])
    return parts[-1]


def _split_of(col: str) -> Optional[str]:
    """Test-subset name when present: ``translations.<split>.beam5.x`` -> ``<split>``.

    Returns ``None`` for the no-subset shape ``translations.beam5.x`` (the
    segment right after ``translations.`` is the beam, not a split name).
    """
    parts = col.split(".")
    if len(parts) >= 2 and not parts[1].lower().startswith("beam"):
        return parts[1]
    return None


def _metric_matches(metric: str, col: str) -> bool:
    """Token-aware substring match: ``metric`` must be bounded by ``.``/``_``/ends.

    Prevents asking for ``"bleu"`` from also matching ``sacrebleu_chrf_score``
    (the *tool* name ``sacrebleu`` contains the substring ``bleu``).
    """
    pat = re.compile(rf"(?:^|[._]){re.escape(str(metric))}(?:[._]|$)")
    return bool(pat.search(col))


def _safe_fname(text: str) -> str:
    return str(text).replace(".", "_").replace("/", "_").lower()


def _default_group_label(row) -> str:
    return (f"{row['vocab__subword_model']} - {row['vocab__size']}\n"
            f"Tr: {row['train_dataset'].replace('_', ' ')}").title()


def _prettify(text: str) -> str:
    return text.title().replace('_', ' ')


# ---------------------------------------------------------------------------
# Report: experiment results
# ---------------------------------------------------------------------------

class Report:
    """Experiment-results reporter over the ``scores`` from ``predict()``.

    ``runs`` is the nested ``list[list[dict]]`` shape (one inner list per trained
    model, one dict per eval dataset). Use :meth:`from_predict` for a single
    model's results, :meth:`from_runs` for a grid, or :meth:`add` to accumulate.

    ``output_path`` is the report's home: :meth:`save` writes under
    ``<output_path>/reports/`` and the ``plot_*`` methods under
    ``<output_path>/plots/``. Every ``save``/``plot_*`` returns ``self`` so calls
    chain.
    """

    def __init__(self, runs: Iterable[List[Dict]], output_path: Optional[str] = None, *,
                 style: PlotStyle = DEFAULT_STYLE,
                 summary_cols: Sequence[str] = _DEFAULT_SUMMARY_COLS,
                 ref_metric: str = "bleu"):
        self._runs: List[List[Dict]] = list(runs)
        self.output_path = output_path
        self.style = style
        self.summary_cols = summary_cols
        self.ref_metric = ref_metric
        self._df: Optional[pd.DataFrame] = None

    # --- constructors --------------------------------------------------------
    @classmethod
    def from_predict(cls, predict_result: List[Dict], **kw) -> "Report":
        """Wrap one model's ``predict()`` result (a ``list[dict]``)."""
        return cls([predict_result], **kw)

    @classmethod
    def from_runs(cls, runs: Iterable[List[Dict]], **kw) -> "Report":
        """Wrap several models' results (a ``list[list[dict]]``, e.g. a grid)."""
        return cls(runs, **kw)

    def add(self, predict_result: List[Dict]) -> "Report":
        """Append one model's ``predict()`` result and invalidate the cache."""
        self._runs.append(predict_result)
        self._df = None
        return self

    # --- data ----------------------------------------------------------------
    @property
    def df(self) -> pd.DataFrame:
        """Wide DataFrame: one row per (model, eval dataset), columns flattened."""
        if self._df is None:
            self._df = scores_to_dataframe(self._runs)
        return self._df

    @property
    def summary(self) -> pd.DataFrame:
        """Identifying columns + every column matching ``ref_metric``."""
        return summarize_scores(self.df, default_cols=self.summary_cols,
                                ref_metric=self.ref_metric)

    def available_metrics(self) -> List[str]:
        """Distinct numeric metric tokens present (e.g. ``sacrebleu_bleu_score``)."""
        return sorted({_segment_after_beam(c) for c in self._metric_columns()})

    def resolve_metric(self, metric: str, *, beam=None, tool: Optional[str] = None,
                       split: Optional[str] = None) -> str:
        """Resolve a short metric name to the full column it refers to.

        Generic, no hardcoded table: substring-matches ``metric`` against the
        numeric metric columns (``translations.[<split>.]beam{N}.{tool}_{metric}_{field}``).
        ``beam`` and ``split`` (the ``test_subsets`` name) are each inferred when
        only one is present and required when several are; ``tool`` disambiguates
        same-metric tools (``sacrebleu`` vs ``hg``). Raises ``ValueError``
        (listing candidates / available metrics) when the match is empty or
        ambiguous.
        """
        candidates = [c for c in self._metric_columns() if _metric_matches(metric, c)]
        if not candidates:
            raise ValueError(f"No metric column matches '{metric}'. "
                             f"Available: {self.available_metrics()}")

        splits = sorted({s for c in candidates if (s := _split_of(c)) is not None})
        if split is not None:
            candidates = [c for c in candidates if _split_of(c) == split]
            if not candidates:
                raise ValueError(f"No '{metric}' column for split='{split}'. "
                                 f"Splits present: {splits}")
        elif len(splits) > 1:
            raise ValueError(f"'{metric}' is ambiguous across splits {splits}; "
                             f"pass split=<name>.")

        beams = sorted({m.group(1) for c in candidates
                        for m in [_BEAM_RE.search(c)] if m})
        if beam is not None:
            tag = f"beam{beam}".lower()
            candidates = [c for c in candidates if tag in c.lower()]
            if not candidates:
                raise ValueError(f"No '{metric}' column for beam={beam}. "
                                 f"Beams present: {beams}")
        elif len(beams) > 1:
            raise ValueError(f"'{metric}' is ambiguous across beams {beams}; "
                             f"pass beam=<N>.")

        if tool is not None:
            candidates = [c for c in candidates
                          if _segment_after_beam(c).startswith(f"{tool}_")]
            if not candidates:
                raise ValueError(f"No '{metric}' column for tool='{tool}'.")

        if len(candidates) > 1:
            raise ValueError(f"'{metric}' is ambiguous; matches {sorted(candidates)}. "
                             f"Narrow with beam=/tool= or a longer metric name.")
        return candidates[0]

    def _metric_columns(self) -> List[str]:
        df = self.df
        return [c for c in df.columns
                if c.startswith("translations.") and pd.api.types.is_numeric_dtype(df[c])]

    # --- artifacts -----------------------------------------------------------
    def save(self, output_path: Optional[str] = None) -> "Report":
        """Write ``report.json`` + ``report.csv`` + ``report_summary.csv``."""
        out = self._require_output_path(output_path)
        reports_path = os.path.join(out, "reports")
        fileio.make_dir(reports_path)
        fileio.save_json(self._runs, os.path.join(reports_path, "report.json"))
        self.df.to_csv(os.path.join(reports_path, "report.csv"), index=False)
        self.summary.to_csv(os.path.join(reports_path, "report_summary.csv"), index=False)
        return self

    def plot_comparison(self, metric: str, *, beam=None, tool: Optional[str] = None,
                        split: Optional[str] = None,
                        title: str = "Model comparison", xlabel: str = "MT Models",
                        ylabel: str = "BLEU Score",
                        group_label_fn: Optional[Callable] = None,
                        legend_label_fn: Optional[Callable] = None,
                        figsize=(16, 8), font_scale: float = 1.5,
                        out_dir: Optional[str] = None,
                        fname: Optional[str] = None) -> "Report":
        """Grouped bar chart comparing one metric across runs."""
        col = self.resolve_metric(metric, beam=beam, tool=tool, split=split)
        df = self.df.copy()
        required = {"train_dataset", "test_dataset", col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in report: {sorted(missing)}")

        df["bar_group_name"] = df.apply(group_label_fn or _default_group_label, axis=1)

        # Legend on test_dataset only when train != test on some row (else redundant).
        same_count = int((df['train_dataset'] == df['test_dataset']).values.sum())
        legend_col = "test_dataset" if same_count != len(df['train_dataset']) else None
        if legend_col:
            df[legend_col] = df[legend_col].map(legend_label_fn or _prettify)

        out = self._plots_dir(out_dir)
        CatPlot(df, x="bar_group_name", y=col, hue=legend_col,
                value_format="{:.2f}", legend_loc="lower right",
                title=title, xlabel=xlabel, ylabel=ylabel,
                style=self.style, figsize=figsize, font_scale=font_scale
                ).render(out, fname or f"plot__{_safe_fname(col)}")
        return self

    def plot_sweep(self, metric: str, x: str, *, beam=None, tool: Optional[str] = None,
                   split: Optional[str] = None, hue: Optional[str] = None,
                   y_right: Optional[str] = None, y_right_hue: Optional[str] = None,
                   xlabel: Optional[str] = None,
                   ylabel_left: Optional[str] = None, ylabel_right: Optional[str] = None,
                   title: str = "Metric sweep", legend_loc: str = "upper left",
                   figsize=(8, 6), font_scale: Optional[float] = None,
                   out_dir: Optional[str] = None,
                   fname: Optional[str] = None) -> "Report":
        """Line plot of a metric across a swept variable (vocab size, model size, ...).

        ``x`` is a report column; ``y_right`` (optional, a raw column) adds a
        dashed secondary axis.
        """
        col = self.resolve_metric(metric, beam=beam, tool=tool, split=split)
        df = self.df
        if x not in df.columns:
            raise ValueError(f"'{x}' not found in report columns")
        if y_right is not None and y_right not in df.columns:
            raise ValueError(f"y_right '{y_right}' not found in report columns")

        out = self._plots_dir(out_dir)
        suffix = f"_{_safe_fname(y_right)}" if y_right else ""
        LinePlot(df, x=x, y_left=col, y_left_hue=hue,
                 y_right=y_right, y_right_hue=y_right_hue,
                 xlabel=xlabel or x, ylabel_left=ylabel_left or col,
                 ylabel_right=ylabel_right or (y_right or ""),
                 legend_loc=legend_loc, title=title,
                 style=self.style, figsize=figsize, font_scale=font_scale
                 ).render(out, fname or f"sweep__{_safe_fname(col)}{suffix}")
        return self

    def plot_matrix(self, metric: str, *, rows: str = "train_dataset",
                    cols: str = "test_dataset", beam=None, tool: Optional[str] = None,
                    split: Optional[str] = None,
                    aggfunc: str = "mean", title: str = "Metric matrix",
                    annot_format: str = ".2f", figsize=(10, 8),
                    font_scale: Optional[float] = None,
                    out_dir: Optional[str] = None,
                    fname: Optional[str] = None) -> "Report":
        """Heatmap of one metric over a ``rows`` × ``cols`` grid (e.g. train × test)."""
        col = self.resolve_metric(metric, beam=beam, tool=tool, split=split)
        df = self.df
        for c in (rows, cols):
            if c not in df.columns:
                raise ValueError(f"'{c}' not found in report columns")

        matrix = df.pivot_table(index=rows, columns=cols, values=col, aggfunc=aggfunc)
        out = self._plots_dir(out_dir)
        HeatmapPlot(matrix, xlabels=list(matrix.columns), ylabels=list(matrix.index),
                    annot_format=annot_format, title=title,
                    style=self.style, figsize=figsize, font_scale=font_scale
                    ).render(out, fname or f"matrix__{_safe_fname(col)}")
        return self

    def __str__(self) -> str:
        return format_summary_table(self.summary)

    # --- internals -----------------------------------------------------------
    def _require_output_path(self, output_path: Optional[str]) -> str:
        out = output_path or self.output_path
        if out is None:
            raise ValueError("No output_path set; pass it to Report(...) or to this method")
        return out

    def _plots_dir(self, out_dir: Optional[str]) -> str:
        if out_dir is None:
            out_dir = os.path.join(self._require_output_path(None), "plots")
        fileio.make_dir(out_dir)
        return out_dir


# ---------------------------------------------------------------------------
# DatasetReport: corpus diagnostics
# ---------------------------------------------------------------------------

_SUBWORD_DISPLAY = {"word": "Words", "bpe": "BPE", "char": "Chars", "bytes": "Bytes"}


def _ds_title(ds) -> str:
    ds_name, lang_pair, _ = ds.base_id()
    return f"{ds_name.title()} ({lang_pair}; {ds.subword_model}; {ds.vocab_size})"


def _ds_fname_suffix(ds) -> str:
    ds_name, lang_pair, ds_size_name = ds.base_id()
    vocab = f"_{ds.vocab_size}" if ds.vocab_size else ""
    s = f"{ds_name}_{ds_size_name}_{lang_pair}__{ds.subword_model}{vocab}"
    return s.lower().replace('/', '_')


def _maybe_titled(ds_title: str, body: str, add_dataset_title: bool) -> str:
    return f"{ds_title}:\n{body}" if add_dataset_title else body


class DatasetReport:
    """Corpus-diagnostics reporter for one preprocessed :class:`Dataset`.

    Figures are written under ``ds.get_plots_path()``. Call :meth:`generate` for
    the full set (and the per-split stats JSON), or the granular ``plot_*``
    methods individually.
    """

    def __init__(self, ds, *, style: PlotStyle = DEFAULT_STYLE):
        self.ds = ds
        self.style = style

    def generate(self, *, merge_vocabs: bool, vocab_top_k: Iterable[int] = (256,),
                 add_dataset_title: bool = True) -> "DatasetReport":
        """Emit all standard diagnostic figures + the per-split stats JSON."""
        ds = self.ds
        out_dir = ds.get_plots_path()
        fileio.make_dir(out_dir)
        suffix = _ds_fname_suffix(ds)
        log.info(f"\t- Creating plots for: {ds.variant_id(as_path=True)}")

        log.info("\t\t- Creating 'Sentence length distribution' plots...")
        rows = self.plot_length_distribution(add_dataset_title=add_dataset_title)
        split_stats = {fname: row for fname, row in rows}
        fileio.save_json(split_stats, os.path.join(out_dir, f"stats__{suffix}.json"))

        log.info("\t\t- Creating 'Split sizes' plots...")
        self.plot_split_sizes(split_stats=rows, add_dataset_title=add_dataset_title)

        if str(ds.subword_model) not in {"None", "none"}:
            log.info("\t\t- Creating 'Vocabulary distribution' plots...")
            self.plot_vocab_distribution(merge_vocabs=merge_vocabs,
                                         vocab_top_k=tuple(vocab_top_k))
        return self

    def plot_length_distribution(self, *, add_dataset_title: bool = True):
        """One histogram per (split, lang). Returns ``[(fname, stats_row), ...]``."""
        ds = self.ds
        out_dir = ds.get_plots_path()
        fileio.make_dir(out_dir)
        suffix = _ds_fname_suffix(ds)
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
            HistogramPlot(df, x="frequency", bins=100,
                          title=_maybe_titled(ds_title, title, add_dataset_title),
                          xlabel="Tokens per sentence", ylabel="Frequency",
                          style=self.style, figsize=(6, 4), font_scale=1.5
                          ).render(out_dir, f"sent_distr_{split_name}_{split_lang}__{suffix}".lower())
        return rows

    def plot_split_sizes(self, *, split_stats=None, add_dataset_title: bool = True):
        """Bar charts of per-split sentence / token counts."""
        ds = self.ds
        out_dir = ds.get_plots_path()
        fileio.make_dir(out_dir)
        suffix = _ds_fname_suffix(ds)
        ds_title = _ds_title(ds)
        rows = split_stats if split_stats is not None else self._collect_split_stats()
        df = pd.DataFrame([row for _, row in rows])

        CatPlot(df, x="split", y="total_sentences", hue="lang",
                title=_maybe_titled(ds_title, "Split sizes (by number of sentences)", add_dataset_title),
                xlabel="Dataset partitions", ylabel="Num. of sentences",
                style=self.style, figsize=(6, 4)
                ).render(out_dir, f"split_size_sent__{suffix}".lower())

        if str(ds.subword_model) not in {"None", "none"}:
            body = f"Split sizes (by number of tokens - {str(ds.subword_model).title()})"
            CatPlot(df, x="split", y="total_tokens", hue="lang",
                    title=_maybe_titled(ds_title, body, add_dataset_title),
                    xlabel="Dataset partitions", ylabel="Num. of tokens",
                    style=self.style, figsize=(6, 4)
                    ).render(out_dir, f"split_size_tok__{suffix}".lower())

    def plot_vocab_distribution(self, *, merge_vocabs: bool,
                                vocab_top_k: Sequence[int] = (256,)):
        """Sampled token-frequency bar charts from the exported ``.vocabf`` files."""
        ds = self.ds
        if str(ds.subword_model) in {"None", "none"}:
            return
        out_dir = ds.get_plots_path()
        fileio.make_dir(out_dir)
        suffix = _ds_fname_suffix(ds)

        src_lang, tgt_lang = ds.base_id()[1].split('-')
        lang_files = [f"{src_lang}-{tgt_lang}"] if merge_vocabs else [src_lang, tgt_lang]
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
                BarPlot(df_sample, x="token", y="frequency",
                        title=f"Vocabulary distribution ({display} - {len(df):,})",
                        xlabel="Tokens", ylabel="Frequency",
                        style=self.style, figsize=(12, 8), font_scale=2.5
                        ).render(out_dir, f"vocab_distr_{lang_file}_sampled{top_k}__{suffix}".lower())

    def _collect_split_stats(self):
        """Per-split basic stats (no figures); used when ``plot_split_sizes`` is called alone."""
        ds = self.ds
        rows = []
        for fname in ds.get_split_fnames():
            split_name, split_lang = fname.split('.')
            tokens = np.array(stats.count_tokens_per_sentence(filename=ds.get_encoded_path(fname)))
            row = stats.basic_stats(tokens, prefix="")
            row.update({"split": split_name, "lang": split_lang})
            rows.append((fname, row))
        return rows
