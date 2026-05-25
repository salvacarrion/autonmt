"""Plot primitives.

Thin, low-level wrappers over seaborn / matplotlib. They take a DataFrame plus
the bare-minimum styling and handle one concern: produce one figure, save it in
one or more formats, and close it. Anything domain-specific (which columns to
read, how to label NMT runs, etc.) belongs in :mod:`autonmt.reporting.figures`.

Two pieces of shared scaffolding:

- :class:`PlotStyle`: default formats, sizes, dpi, font scale, save/show flags.
  All primitives accept ``style=PlotStyle(...)`` and override individual fields
  via kwargs only when needed.
- :func:`save_figure`: writes the active figure to ``<out_dir>/<ext>/<fname>.<ext>``
  for each requested extension and closes it. Used by every primitive.

The overwrite check is also shared (:func:`_all_figs_exist`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from autonmt.utils.formatting import human_format_int
from autonmt.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Backend / global style
# ---------------------------------------------------------------------------

_seaborn_initialized = False


def use_non_gui_backend():
    """Switch matplotlib to the Agg backend, the only safe option inside a loop.

    See https://github.com/matplotlib/matplotlib/issues/8519 — interactive
    backends drop frames when the producer thread out-runs the GUI thread.
    """
    matplotlib.use('agg')


def _ensure_seaborn_defaults():
    global _seaborn_initialized
    if not _seaborn_initialized:
        sns.set_theme()
        _seaborn_initialized = True


# ---------------------------------------------------------------------------
# Shared options
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlotStyle:
    """Shared plot configuration. Pass it explicitly when you need to override
    something; otherwise the primitives use sensible defaults."""
    formats: Tuple[str, ...] = ("png", "pdf")
    figsize: Tuple[float, float] = (12, 8)
    dpi: int = 150
    font_scale: float = 1.0
    overwrite: bool = True
    save: bool = True
    show: bool = False

    def merge(self, **overrides) -> "PlotStyle":
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})


DEFAULT_STYLE = PlotStyle()


# ---------------------------------------------------------------------------
# Shared scaffolding
# ---------------------------------------------------------------------------

def _all_figs_exist(out_dir, fname, formats):
    return all(os.path.exists(os.path.join(out_dir, ext, f"{fname}.{ext}"))
               for ext in formats)


def _should_skip(out_dir, fname, style: PlotStyle) -> bool:
    if not style.save or style.overwrite:
        return False
    if _all_figs_exist(out_dir, fname, style.formats):
        log.info(f"\t\t\t- Skipped (already exists): {fname}")
        return True
    return False


def _validate(style: PlotStyle):
    if style.save and style.show:
        raise ValueError("'save' and 'show' are mutually exclusive")
    if style.save:
        use_non_gui_backend()


def save_figure(fig, out_dir, fname, style: PlotStyle):
    """Persist the active figure under ``<out_dir>/<ext>/<fname>.<ext>`` and close it."""
    if style.save:
        for ext in style.formats:
            save_dir = os.path.join(out_dir, ext)
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            path = os.path.join(save_dir, f"{fname}.{ext}")
            plt.savefig(path, dpi=style.dpi)
            log.info(f"\t\t\t- Figure saved: {path}")

    if style.show:
        plt.show()
    else:
        plt.close(fig) if fig is not None else plt.close()


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def catplot(data, x, y, hue, out_dir, fname,
            title="", xlabel="", ylabel="", legend_title=None,
            value_format: Optional[str] = "{:.0f}",
            rotate_xlabels: float = 0, legend_loc: str = "upper right",
            style: PlotStyle = DEFAULT_STYLE):
    """Grouped bar plot via ``sns.catplot`` (kind='bar')."""
    if _should_skip(out_dir, fname, style):
        return
    _validate(style)
    _ensure_seaborn_defaults()

    # Widen figure for long category labels (rough heuristic, was previously inline).
    max_label_length = max(data[x].astype(str).apply(len))
    w, h = style.figsize
    aspect = (w + max_label_length * 0.2) / h

    sns.set_context("notebook", font_scale=style.font_scale)
    g = sns.catplot(data=data, x=x, y=y, hue=hue, kind="bar",
                    height=h, aspect=aspect, legend_out=False, legend=True)

    for label in g.ax.get_xticklabels():
        label.set_rotation(rotate_xlabels)
        label.set_horizontalalignment('center' if rotate_xlabels == 0 else 'right')

    if value_format:
        for c in g.ax.containers:
            labels = [value_format.format(float(v.get_height())) for v in c]
            g.ax.bar_label(c, labels=labels, label_type='edge')

    g.set(xlabel=xlabel, ylabel=ylabel)
    plt.title(title)
    if hue:
        plt.legend(title=legend_title, loc=legend_loc)
    plt.tight_layout()
    save_figure(g.fig, out_dir, fname, style)


def barplot(data, x, y, out_dir, fname,
            title="", xlabel="x", ylabel="y",
            style: PlotStyle = DEFAULT_STYLE):
    """Single-series bar plot, used for long-tailed vocabulary distributions."""
    if _should_skip(out_dir, fname, style):
        return
    _validate(style)
    _ensure_seaborn_defaults()

    fig, ax = plt.subplots(figsize=style.figsize)
    sns.set_context("notebook", font_scale=style.font_scale)
    sns.barplot(ax=ax, data=data, x=x, y=y, edgecolor="none")

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_xticks([])  # too many bars to label
    ax.tick_params(axis='y', which='major', labelsize=12)
    ax.yaxis.set_major_formatter(human_format_int)

    plt.title(title)
    plt.tight_layout()
    save_figure(fig, out_dir, fname, style)


def histogram(data, x, out_dir, fname,
              title="", xlabel="x", ylabel="y", bins="auto",
              style: PlotStyle = DEFAULT_STYLE):
    if _should_skip(out_dir, fname, style):
        return
    _validate(style)
    _ensure_seaborn_defaults()

    fig, ax = plt.subplots(figsize=style.figsize)
    sns.set_context("notebook", font_scale=style.font_scale)
    sns.histplot(ax=ax, data=data, x=x, bins=bins)

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.tick_params(axis='x', which='major', labelsize=8 * style.font_scale)
    ax.tick_params(axis='y', which='major', labelsize=8 * style.font_scale)
    ax.yaxis.set_major_formatter(human_format_int)

    plt.title(title)
    plt.tight_layout()
    save_figure(fig, out_dir, fname, style)


def heatmap(data, xlabels, ylabels, out_dir, fname,
            title="", annot=True, cbar=False, annot_format=".2f",
            style: PlotStyle = DEFAULT_STYLE):
    if _should_skip(out_dir, fname, style):
        return
    _validate(style)
    _ensure_seaborn_defaults()

    fig, ax = plt.subplots(figsize=style.figsize)
    sns.set_context("notebook", font_scale=style.font_scale)
    sns.heatmap(data, ax=ax, annot=annot, cbar=cbar, fmt=annot_format)
    ax.set_xticklabels([x.title() for x in xlabels], ha='center', minor=False)
    ax.set_yticklabels([y.title() for y in ylabels], va='center', minor=False)

    if title:
        plt.title(title, y=1.01)
    plt.tight_layout()
    save_figure(fig, out_dir, fname, style)


def lineplot(data, x, y_left, out_dir, fname,
             y_left_hue=None, y_right=None, y_right_hue=None,
             title="", xlabel="", ylabel_left="", ylabel_right="",
             legend_loc: str = "upper left",
             style: PlotStyle = DEFAULT_STYLE):
    """Line plot with an optional secondary axis on the right."""
    if _should_skip(out_dir, fname, style):
        return
    _validate(style)
    _ensure_seaborn_defaults()

    fig, ax = plt.subplots(figsize=style.figsize)
    sns.set_context("notebook", font_scale=style.font_scale)

    g1 = sns.lineplot(data=data, x=x, y=y_left, hue=y_left_hue, ax=ax,
                      marker="o", legend=True)
    g1.set(ylim=(0, None), xlabel=xlabel, ylabel=ylabel_left)
    h1, l1 = g1.get_legend_handles_labels()

    h2, l2 = [], []
    legend_target = g1
    if y_right:
        ax2 = ax.twinx()
        ax2.grid(False)
        g2 = sns.lineplot(data=data, x=x, y=y_right, hue=y_right_hue, ax=ax2,
                          color="grey", linestyle="dashed",
                          label=ylabel_right, legend=False)
        g2.set(xlabel=xlabel, ylabel=ylabel_right)
        h2, l2 = g2.get_legend_handles_labels()
        ax.get_legend().remove()
        legend_target = g2

    legend_target.legend(loc=legend_loc, handles=h1 + h2, labels=l1 + l2)

    plt.title(title)
    plt.tight_layout()
    save_figure(fig, out_dir, fname, style)
