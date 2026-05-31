"""Plot primitives: a small :class:`BasePlot` template hierarchy.

Thin, low-level wrappers over seaborn / matplotlib. Each concrete plot takes a
DataFrame plus the bare-minimum styling and handles one concern: produce one
figure, save it in one or more formats, and close it. Anything domain-specific
(which columns to read, how to label NMT runs, etc.) belongs in the report
classes (:class:`autonmt.reporting.report.Report` /
:class:`~autonmt.reporting.report.DatasetReport`), not here.

Design — template method:

- :class:`BasePlot` owns the shared lifecycle (skip-if-exists, backend
  validation, seaborn context, figure creation, title/labels, save+close).
- Each subclass implements only :meth:`BasePlot._draw` (the seaborn call) and,
  when its labelling differs, overrides :meth:`BasePlot._finish` / ``_make_fig``.

Shared scaffolding:

- :class:`PlotStyle`: default formats, sizes, dpi, font scale, save/show flags.
  Pass ``style=PlotStyle(...)`` (or per-instance ``figsize`` / ``font_scale``).
- :func:`save_figure`: writes the active figure to ``<out_dir>/<ext>/<fname>.<ext>``
  for each requested extension and closes it.
- :func:`_should_skip`: the shared overwrite check.
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
# Base plot (template method)
# ---------------------------------------------------------------------------

class BasePlot:
    """One figure, one lifecycle. Subclasses implement :meth:`_draw`.

    The lifecycle (see :meth:`render`): skip-if-exists → validate backend →
    seaborn context → create figure → ``_draw(ax)`` → ``_finish(fig, ax)`` →
    save + close. ``figsize`` / ``font_scale`` override the ``style`` fields for
    this single plot without mutating the shared :class:`PlotStyle`.
    """

    def __init__(self, data, *, title: str = "", xlabel: str = "", ylabel: str = "",
                 style: PlotStyle = DEFAULT_STYLE,
                 figsize: Optional[Tuple[float, float]] = None,
                 font_scale: Optional[float] = None):
        self.data = data
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.style = style.merge(figsize=figsize, font_scale=font_scale)

    def render(self, out_dir, fname) -> "BasePlot":
        if _should_skip(out_dir, fname, self.style):
            return self
        _validate(self.style)
        _ensure_seaborn_defaults()
        sns.set_context("notebook", font_scale=self.style.font_scale)

        fig, ax = self._make_fig()
        self._draw(ax)
        self._finish(fig, ax)
        save_figure(fig, out_dir, fname, self.style)
        return self

    # --- hooks ---------------------------------------------------------------
    def _make_fig(self):
        return plt.subplots(figsize=self.style.figsize)

    def _draw(self, ax):
        raise NotImplementedError

    def _finish(self, fig, ax):
        ax.set(xlabel=self.xlabel, ylabel=self.ylabel)
        if self.title:
            ax.set_title(self.title)
        fig.tight_layout()


# ---------------------------------------------------------------------------
# Concrete plots
# ---------------------------------------------------------------------------

class CatPlot(BasePlot):
    """Grouped bar plot (categorical comparison).

    Drawn with ``sns.barplot`` on a single axes (not ``sns.catplot``, whose
    FacetGrid owns its own figure and wouldn't fit the shared lifecycle). The
    figure is widened to fit long category labels.
    """

    def __init__(self, data, *, x, y, hue=None, legend_title=None,
                 value_format: Optional[str] = "{:.0f}",
                 rotate_xlabels: float = 0, legend_loc: str = "upper right", **kw):
        super().__init__(data, **kw)
        self.x, self.y, self.hue = x, y, hue
        self.legend_title = legend_title
        self.value_format = value_format
        self.rotate_xlabels = rotate_xlabels
        self.legend_loc = legend_loc

    def _make_fig(self):
        # Widen for long category labels (rough heuristic; matches the old aspect calc).
        max_label_length = max(self.data[self.x].astype(str).apply(len))
        w, h = self.style.figsize
        return plt.subplots(figsize=(w + max_label_length * 0.2, h))

    def _draw(self, ax):
        sns.barplot(data=self.data, x=self.x, y=self.y, hue=self.hue, ax=ax)

        for label in ax.get_xticklabels():
            label.set_rotation(self.rotate_xlabels)
            label.set_horizontalalignment('center' if self.rotate_xlabels == 0 else 'right')

        if self.value_format:
            for c in ax.containers:
                labels = [self.value_format.format(float(v.get_height())) for v in c]
                ax.bar_label(c, labels=labels, label_type='edge')

    def _finish(self, fig, ax):
        ax.set(xlabel=self.xlabel, ylabel=self.ylabel)
        if self.title:
            ax.set_title(self.title)
        if self.hue:
            ax.legend(title=self.legend_title, loc=self.legend_loc)
        fig.tight_layout()


class BarPlot(BasePlot):
    """Single-series bar plot, used for long-tailed vocabulary distributions."""

    def __init__(self, data, *, x, y, **kw):
        super().__init__(data, **kw)
        self.x, self.y = x, y

    def _draw(self, ax):
        sns.barplot(ax=ax, data=self.data, x=self.x, y=self.y, edgecolor="none")
        ax.set_xticks([])  # too many bars to label
        ax.tick_params(axis='y', which='major', labelsize=12)
        ax.yaxis.set_major_formatter(human_format_int)


class HistogramPlot(BasePlot):
    """Distribution histogram (e.g. tokens-per-sentence)."""

    def __init__(self, data, *, x, bins="auto", **kw):
        super().__init__(data, **kw)
        self.x, self.bins = x, bins

    def _draw(self, ax):
        sns.histplot(ax=ax, data=self.data, x=self.x, bins=self.bins)
        ax.tick_params(axis='x', which='major', labelsize=8 * self.style.font_scale)
        ax.tick_params(axis='y', which='major', labelsize=8 * self.style.font_scale)
        ax.yaxis.set_major_formatter(human_format_int)


class HeatmapPlot(BasePlot):
    """Annotated matrix heatmap (e.g. a train × test metric grid)."""

    def __init__(self, data, *, xlabels, ylabels, annot=True, cbar=False,
                 annot_format=".2f", **kw):
        super().__init__(data, **kw)
        self.xlabels, self.ylabels = xlabels, ylabels
        self.annot, self.cbar, self.annot_format = annot, cbar, annot_format

    def _draw(self, ax):
        sns.heatmap(self.data, ax=ax, annot=self.annot, cbar=self.cbar, fmt=self.annot_format)
        ax.set_xticklabels([str(x).title() for x in self.xlabels], ha='center', minor=False)
        ax.set_yticklabels([str(y).title() for y in self.ylabels], va='center', minor=False)

    def _finish(self, fig, ax):
        # Heatmaps label via ticks, not axis labels.
        if self.title:
            ax.set_title(self.title, y=1.01)
        fig.tight_layout()


class LinePlot(BasePlot):
    """Line plot with an optional secondary (right) y-axis.

    ``y_left`` / ``y_right`` are column names; ``*_hue`` optionally split each
    into multiple lines. The right axis is dashed/grey and shares the legend.
    """

    def __init__(self, data, *, x, y_left, y_left_hue=None,
                 y_right=None, y_right_hue=None,
                 ylabel_left: str = "", ylabel_right: str = "",
                 legend_loc: str = "upper left", **kw):
        super().__init__(data, **kw)
        self.x = x
        self.y_left, self.y_left_hue = y_left, y_left_hue
        self.y_right, self.y_right_hue = y_right, y_right_hue
        self.ylabel_left, self.ylabel_right = ylabel_left, ylabel_right
        self.legend_loc = legend_loc

    def _draw(self, ax):
        g1 = sns.lineplot(data=self.data, x=self.x, y=self.y_left,
                          hue=self.y_left_hue, ax=ax, marker="o", legend=True)
        g1.set(ylim=(0, None), xlabel=self.xlabel, ylabel=self.ylabel_left)
        h1, l1 = g1.get_legend_handles_labels()

        h2, l2 = [], []
        legend_target = g1
        if self.y_right:
            ax2 = ax.twinx()
            ax2.grid(False)
            g2 = sns.lineplot(data=self.data, x=self.x, y=self.y_right,
                              hue=self.y_right_hue, ax=ax2,
                              color="grey", linestyle="dashed",
                              label=self.ylabel_right, legend=False)
            g2.set(xlabel=self.xlabel, ylabel=self.ylabel_right)
            h2, l2 = g2.get_legend_handles_labels()
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
            legend_target = g2

        if h1 + h2:  # nothing to label for a single unhued line
            legend_target.legend(loc=self.legend_loc, handles=h1 + h2, labels=l1 + l2)

    def _finish(self, fig, ax):
        # Axis labels are set inside _draw (left/right differ).
        if self.title:
            ax.set_title(self.title)
        fig.tight_layout()
