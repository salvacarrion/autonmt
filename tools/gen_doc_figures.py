"""
Generate the example figures + report artifacts embedded in the documentation.

These are produced from *synthetic but plausible* scores — no models are
trained. The point is to render the exact figures AutoNMT's reporting layer
emits (`Report.plot_comparison` / `plot_sweep` / `plot_matrix` and the
`DatasetReport` diagnostics), so the docs can show real outputs without a GPU.

The numbers are hand-picked to tell a believable story (subword beats word-level,
the Transformer beats RNN/Conv, BLEU plateaus as the vocab grows, in-domain beats
cross-domain, sentence lengths are right-skewed, token frequencies are Zipfian) —
they are illustrative, not measured.

Design notes
------------
- All figures share one style: SVG (vector — scales without quality loss),
  a single `FONT_SCALE`, and a consistent figure height, so text size is
  homogeneous across the gallery.
- The sentence-length histogram caps its bin count at the token-count range, so
  integer counts never produce empty sub-unit bins (the "gaps" artifact). This
  mirrors the fix in `DatasetReport.plot_length_distribution`.

Run:
    python tools/gen_doc_figures.py

Writes SVGs into  docs/images/reports/  and prints the terminal summary tables
(captured into the docs by hand).
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pandas as pd

from autonmt.reporting.report import Report
from autonmt.reporting.plots import PlotStyle, HistogramPlot, CatPlot, BarPlot

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMG_DIR = os.path.join(ROOT, "docs", "images", "reports")
BUILD_DIR = os.path.join(IMG_DIR, "_build")

# One shared style for the whole gallery: vector output + a single font scale, so
# every figure reads at the same text size. Height is kept ~constant per figure
# below for the same reason (font size is in points; ratio-to-figure stays even).
FONT_SCALE = 1.4
STYLE = PlotStyle(formats=("svg",), font_scale=FONT_SCALE)

_SW_DISPLAY = {"word": "Word", "char": "Char", "bpe": "BPE", "unigram": "Unigram"}


# ---------------------------------------------------------------------------
# Synthetic score builder (mirrors the dict shape predict() returns)
# ---------------------------------------------------------------------------

def run_dict(*, train_dataset, test_dataset, lang_pair, subword, vocab_size,
             bleu, chrf=None, total_params=None, beam=5, arch="transformer"):
    """One (run, eval_ds) report dict, matching autonmt.reporting.schema."""
    translations = {f"beam{beam}": {"sacrebleu_bleu_score": float(bleu)}}
    if chrf is not None:
        translations[f"beam{beam}"]["sacrebleu_chrf_score"] = float(chrf)
    d = {
        "engine": "autonmt",
        "run_name": f"{train_dataset}_{subword}_{vocab_size}",
        "model__architecture": arch,
        "vocab__subword_model": subword,
        "vocab__size": vocab_size,
        "vocab__merged": False,
        "train__lang_pair": lang_pair,
        "test__lang_pair": lang_pair,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "test_dataset_full": f"{test_dataset}__{lang_pair}",
        "translations": translations,
    }
    if total_params is not None:
        d["model__total_params"] = int(total_params)
    return d


# ---------------------------------------------------------------------------
# Figure — tokenization comparison (grouped bar chart)
# ---------------------------------------------------------------------------

def fig_comparison_tokenization():
    # Same multi30k de-en test set; only the tokenization changes. Subword
    # models (bpe/unigram) beat word-level; char is competitive but behind.
    runs = [
        [run_dict(train_dataset="multi30k", test_dataset="multi30k", lang_pair="de-en",
                  subword="word", vocab_size=16000, bleu=31.2, chrf=53.4)],
        [run_dict(train_dataset="multi30k", test_dataset="multi30k", lang_pair="de-en",
                  subword="char", vocab_size=256, bleu=33.1, chrf=55.6)],
        [run_dict(train_dataset="multi30k", test_dataset="multi30k", lang_pair="de-en",
                  subword="bpe", vocab_size=8000, bleu=36.0, chrf=57.4)],
        [run_dict(train_dataset="multi30k", test_dataset="multi30k", lang_pair="de-en",
                  subword="unigram", vocab_size=8000, bleu=36.4, chrf=57.8)],
    ]
    report = Report.from_runs(runs, output_path=BUILD_DIR, style=STYLE)
    report.plot_comparison(
        "bleu", beam=5,
        xlabel="Tokenization", ylabel="BLEU",
        title="Tokenization comparison — multi30k (de→en)",
        group_label_fn=lambda r: f"{_SW_DISPLAY[r['vocab__subword_model']]}\n{r['vocab__size']:,}",
        figsize=(10, 5.2), font_scale=FONT_SCALE,
        fname="report_comparison",
    )
    return report


# ---------------------------------------------------------------------------
# Figure — architecture comparison (grouped bar chart)
# ---------------------------------------------------------------------------

def fig_comparison_architecture():
    # One fixed dataset cell; only the model changes. Transformer leads, the conv
    # model and the attention RNN follow, and a plain LSTM (no attention) trails —
    # the usual small-corpus ordering. The first field is the string each model
    # reports as `model__architecture` (SimpleRNN with an LSTM cell -> "SimpleRNN-LSTM").
    specs = [
        ("transformer",    "Transformer", 36.0, 57.4),
        ("convs2s",        "ConvS2S",     34.9, 56.5),
        ("bahdanau",       "Bahdanau",    33.8, 55.7),
        ("SimpleRNN-LSTM", "LSTM",        30.6, 53.0),
    ]
    labels = {arch: label for arch, label, *_ in specs}
    runs = [[run_dict(train_dataset="multi30k", test_dataset="multi30k", lang_pair="de-en",
                      subword="bpe", vocab_size=8000, bleu=b, chrf=c, arch=arch)]
            for arch, _label, b, c in specs]
    report = Report.from_runs(runs, output_path=BUILD_DIR, style=STYLE)
    report.plot_comparison(
        "bleu", beam=5,
        xlabel="Model architecture", ylabel="BLEU",
        title="Architecture comparison — multi30k (de→en)",
        group_label_fn=lambda r: labels[r["model__architecture"]],
        figsize=(9.5, 5.2), font_scale=FONT_SCALE,
        fname="report_comparison_arch",
    )
    return report


# ---------------------------------------------------------------------------
# Figure — vocab-size sweep (line + secondary axis)
# ---------------------------------------------------------------------------

def fig_sweep():
    # BPE vocab sweep on multi30k de-en. BLEU climbs then plateaus; the model's
    # parameter count grows ~linearly with the vocab (the embedding tables).
    base = 5_000_000
    sizes = [1000, 2000, 4000, 8000, 16000]
    bleu = [30.1, 33.4, 35.2, 36.0, 35.6]
    chrf = [52.0, 55.1, 56.8, 57.4, 57.1]
    runs = [[run_dict(train_dataset="multi30k", test_dataset="multi30k", lang_pair="de-en",
                      subword="bpe", vocab_size=v, bleu=b, chrf=c,
                      total_params=base + v * 256 * 2)]
            for v, b, c in zip(sizes, bleu, chrf)]
    report = Report.from_runs(runs, output_path=BUILD_DIR, style=STYLE)
    report.plot_sweep(
        "bleu", x="vocab__size", y_right="model__total_params",
        xlabel="BPE vocabulary size", ylabel_left="BLEU",
        ylabel_right="Model parameters",
        title="Vocabulary-size sweep — multi30k (de→en)",
        legend_loc="lower right",   # keep it off the BLEU curve (top) and params line
        figsize=(9, 5.2), font_scale=FONT_SCALE,
        fname="report_sweep",
    )
    return report


# ---------------------------------------------------------------------------
# Figure — cross-evaluation matrix (heatmap)
# ---------------------------------------------------------------------------

def fig_matrix():
    # Three de-en domains, each model scored on every test set (eval_mode="all").
    # In-domain (diagonal) dominates; transfer drops with domain distance.
    domains = ["europarl", "iwslt", "multi30k"]
    bleu = {
        "europarl": {"europarl": 34.2, "iwslt": 22.1, "multi30k": 12.4},
        "iwslt":    {"europarl": 19.8, "iwslt": 31.5, "multi30k": 16.7},
        "multi30k": {"europarl":  9.3, "iwslt": 14.2, "multi30k": 36.8},
    }
    runs = []
    for train in domains:
        evals = [run_dict(train_dataset=train, test_dataset=test, lang_pair="de-en",
                          subword="bpe", vocab_size=8000, bleu=bleu[train][test])
                 for test in domains]
        runs.append(evals)
    report = Report.from_runs(runs, output_path=BUILD_DIR, style=STYLE)
    report.plot_matrix(
        "bleu", rows="train_dataset", cols="test_dataset",
        title="Cross-domain transfer — BLEU (de→en)",
        figsize=(7.5, 6), font_scale=FONT_SCALE,
        fname="report_matrix",
    )
    return report


# ---------------------------------------------------------------------------
# Figure — dataset diagnostics (same primitives DatasetReport uses)
# ---------------------------------------------------------------------------

def fig_dataset_diagnostics():
    rng = np.random.default_rng(42)

    # (1) Sentence-length distribution: right-skewed, ~13 tokens/sentence mean.
    # Token counts are integers, so cap the bins at the value range — otherwise
    # 100 bins over a ~50-token range leave empty sub-unit bins ("gaps"). This
    # mirrors DatasetReport.plot_length_distribution.
    lengths = np.clip(rng.lognormal(mean=2.5, sigma=0.45, size=29_000), 1, None).astype(int)
    n_bins = min(100, int(lengths.max() - lengths.min())) or 1
    df_len = pd.DataFrame(lengths, columns=["frequency"])
    HistogramPlot(
        df_len, x="frequency", bins=n_bins,
        title="Sentence length distribution (Train - en)",
        xlabel="Tokens per sentence", ylabel="Frequency",
        style=STYLE, figsize=(7, 4.6), font_scale=FONT_SCALE,
    ).render(BUILD_DIR, "dataset_length_distribution")

    # (2) Split sizes: multi30k's canonical train/val/test, two languages.
    split_rows = []
    for split, n_sent, tok_de, tok_en in [
        ("train", 29_000, 377_000, 358_000),
        ("val",    1_014,  13_200,  12_500),
        ("test",   1_000,  12_900,  12_100),
    ]:
        split_rows.append({"split": split, "lang": "de",
                           "total_sentences": n_sent, "total_tokens": tok_de})
        split_rows.append({"split": split, "lang": "en",
                           "total_sentences": n_sent, "total_tokens": tok_en})
    df_split = pd.DataFrame(split_rows)
    CatPlot(
        df_split, x="split", y="total_sentences", hue="lang",
        title="Split sizes (by number of sentences)",
        xlabel="Dataset partitions", ylabel="Num. of sentences",
        style=STYLE, figsize=(7.5, 4.6), font_scale=FONT_SCALE,
    ).render(BUILD_DIR, "dataset_split_sizes")

    # (3) Vocabulary distribution: Zipfian frequencies, sampled like DatasetReport.
    vocab_n = 8000
    ranks = np.arange(1, vocab_n + 1)
    freqs = (3_000_000 / ranks ** 1.07).astype(int) + 1
    df_vocab = pd.DataFrame({"token": [f"tok_{i}" for i in ranks], "frequency": freqs})
    df_sample = df_vocab.sample(n=256, random_state=1).sort_values(
        by="frequency", ascending=False)
    BarPlot(
        df_sample, x="token", y="frequency",
        title=f"Vocabulary distribution (BPE - {vocab_n:,})",
        xlabel="Tokens", ylabel="Frequency",
        style=STYLE, figsize=(10, 5.2), font_scale=FONT_SCALE,
    ).render(BUILD_DIR, "dataset_vocab_distribution")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _flatten_svgs():
    """Move every <build>/**/svg/*.svg up into docs/images/reports/ and drop the scratch dir.

    `Report.plot_*` writes under `<output_path>/plots/svg/`, while the bare
    primitives write under `<output_path>/svg/`; collect from both.
    """
    for dirpath, _dirs, files in os.walk(BUILD_DIR):
        if os.path.basename(dirpath) != "svg":
            continue
        for name in sorted(files):
            if name.endswith(".svg"):
                shutil.copy2(os.path.join(dirpath, name), os.path.join(IMG_DIR, name))
                print(f"  - docs/images/reports/{name}")
    shutil.rmtree(BUILD_DIR)


def _remove_stale_pngs():
    """Drop the earlier PNG renders now that the gallery is vector-only."""
    for name in os.listdir(IMG_DIR):
        if name.endswith(".png"):
            os.remove(os.path.join(IMG_DIR, name))


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    if os.path.exists(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)

    print("[gen] rendering figures...")
    rep_tok = fig_comparison_tokenization()
    fig_comparison_architecture()
    fig_sweep()
    rep_mat = fig_matrix()
    fig_dataset_diagnostics()

    print("[gen] figures written:")
    _flatten_svgs()
    _remove_stale_pngs()

    print("\n[gen] terminal summary — tokenization comparison:")
    print(rep_tok)
    print("\n[gen] terminal summary — cross-domain matrix:")
    print(rep_mat)


if __name__ == "__main__":
    main()
