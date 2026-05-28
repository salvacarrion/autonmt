"""
============================================================================
 Tutorial 07 — Under the hood: full manual control
============================================================================

What you'll learn
-----------------
Every shortcut you've used in tutorials 01–06 has a manual equivalent. This
script does the same end-to-end run as tutorial 01 (multi30k de→en, one
Transformer, BLEU + chrF on the test split) but EXPANDS every shortcut into
its lower-level form. Use it as a reference when you need to:

    - Swap a piece (custom decoder, custom Transformer dims, custom callbacks).
    - Resume from a saved checkpoint instead of re-training.
    - Run translate / score / parse on their own (e.g. re-score with a new
      metric without re-translating).
    - Assemble the report manually from the score dicts (e.g. before adding
      your own columns or merging with another experiment).

What's new vs tutorial 06
-------------------------
Nothing in terms of features — this script *exposes* the same pipeline you've
already used. Where earlier tutorials called one-liner helpers like
`AutonmtTranslator.from_dataset(...)`, `ds.build_vocabs(...)`,
`Transformer.from_vocabs(...)`, `FitConfig(...)`, `trainer.predict(...)`, and
`generate_report(...)`, this script unpacks each of them.

Run
---
    pip install -e '.[hf]'
    python examples/07_under_the_hood.py
"""
import datetime
import os

from tokenizers.normalizers import NFKC, Strip

from autonmt.backends import AutonmtTranslator
from autonmt.core.decoding import BeamSearch
from autonmt.core.models import Transformer
from autonmt.datasets import DatasetBuilder
from autonmt.datasets.hf_loader import download_hf_dataset
from autonmt.datasets.processors import normalize_lines, preprocess_lines, preprocess_pairs
from autonmt.reporting.figures import plot_model_comparison
from autonmt.reporting.report import (
    format_summary_table,
    scores_to_dataframe,
    summarize_scores,
)
from autonmt.utils import fileio
from autonmt.utils.seed import manual_seed
from autonmt.vocabularies import Vocabulary

BASE_PATH = "datasets/07_manual"
DATASET = "multi30k"
LANG_PAIR = "de-en"


def normalize(lines):
    return normalize_lines(lines, seq=[NFKC(), Strip()])


def preprocess_train(data, ds):
    return preprocess_pairs(data["src"]["lines"], data["trg"]["lines"], normalize_fn=normalize)


def preprocess_predict(data, ds):
    return preprocess_lines(data["lines"], normalize_fn=normalize)


# ---------------------------------------------------------------------------
# (1) Reproducibility: call manual_seed yourself instead of relying on FitConfig
# ---------------------------------------------------------------------------
# `fit()` internally calls `manual_seed(seed=cfg.seed)` when you pass a seed in
# FitConfig. If you want everything BEFORE fit() to also be deterministic (vocab
# construction, dataloader workers, custom data prep), call it up-front.
manual_seed(seed=42)


def main():
    # -----------------------------------------------------------------------
    # (2) Dataset on disk (same as tutorial 01)
    # -----------------------------------------------------------------------
    download_hf_dataset(
        hf_id="bentrevett/multi30k", base_path=BASE_PATH,
        dataset_name=DATASET, lang_pair=LANG_PAIR,
        src_field="de", trg_field="en",
    )

    builder = DatasetBuilder(
        base_path=BASE_PATH,
        datasets=[{
            "name": DATASET,
            "languages": [LANG_PAIR],
            "sizes": [("original", None)],
        }],
        encoding=[{"subword_models": ["bpe"], "vocab_sizes": [4000]}],
        preprocess_raw_fn=preprocess_train,
        preprocess_splits_fn=preprocess_train,
        merge_vocabs=False,
    ).build(force_overwrite=False)

    train_ds = builder.get_train_ds()[0]
    test_datasets = builder.get_test_ds()

    # -----------------------------------------------------------------------
    # (3) Manual vocab construction
    # -----------------------------------------------------------------------
    # Shortcut:  src_vocab, trg_vocab = train_ds.build_vocabs(max_tokens=150)
    # Manual:    build each vocab yourself, e.g. so you can use different
    #            max_tokens per side, or swap in a custom Vocabulary subclass.
    src_vocab = Vocabulary(max_tokens=150).build_from_ds(
        ds=train_ds, lang=train_ds.src_lang)
    trg_vocab = Vocabulary(max_tokens=150).build_from_ds(
        ds=train_ds, lang=train_ds.trg_lang)
    print(f"[vocab] src={len(src_vocab)} tokens, trg={len(trg_vocab)} tokens")

    # -----------------------------------------------------------------------
    # (4) Manual model construction
    # -----------------------------------------------------------------------
    # Shortcut:  model = Transformer.from_vocabs(src_vocab, trg_vocab)
    # Manual:    explicit dims so you can override anything (depth, width,
    #            attention heads, dropout, max positions, ...). The defaults
    #            below match the ones `Transformer.from_vocabs` would have used.
    model = Transformer(
        src_vocab_size=len(src_vocab),
        trg_vocab_size=len(trg_vocab),
        padding_idx=src_vocab.pad_id,
        encoder_embed_dim=256, decoder_embed_dim=256,
        encoder_layers=3, decoder_layers=3,
        encoder_attention_heads=8, decoder_attention_heads=8,
        encoder_ffn_embed_dim=512, decoder_ffn_embed_dim=512,
        dropout=0.1,
        max_src_positions=1024, max_trg_positions=1024,
    )

    # -----------------------------------------------------------------------
    # (5) Manual translator construction
    # -----------------------------------------------------------------------
    # Shortcut:  AutonmtTranslator.from_dataset(train_ds, ..., run_prefix="manual")
    # Manual:    compute runs_dir / run_name yourself. This is where you'd
    #            override the path scheme (e.g. flat directory of runs) or
    #            tag a run with a custom name like "ablation-v2-seed42".
    runs_dir = train_ds.get_runs_path(toolkit="autonmt")
    run_name = train_ds.get_run_name(run_prefix="manual")  # or any string
    print(f"[run]   runs_dir={runs_dir}")
    print(f"[run]   run_name={run_name}")

    trainer = AutonmtTranslator(
        model=model,
        src_vocab=src_vocab, trg_vocab=trg_vocab,
        runs_dir=runs_dir, run_name=run_name,
    )

    # -----------------------------------------------------------------------
    # (6) fit() with raw kwargs instead of FitConfig
    # -----------------------------------------------------------------------
    # Both call styles are equivalent. Explicit kwargs are handy when you're
    # iterating in a notebook and don't want to instantiate a dataclass each
    # time; FitConfig is handy when you want type-checked configs and to share
    # the same object across runs.
    #
    # Toolkit-specific kwargs (`strategy`, `wandb_params`, `use_bucketing`) ride
    # along through **kwargs whether you pass a FitConfig or not.
    trainer.fit(
        train_ds,
        max_epochs=1, batch_size=128, learning_rate=1e-3,
        optimizer="adam", weight_decay=0.0,
        monitor="val_loss", save_best=True, save_last=False,
        num_workers=0, devices="auto", accelerator="auto",
        seed=42, force_overwrite=False,
    )

    # -----------------------------------------------------------------------
    # (7) translate() → score_translations() → parse_metrics(), separately
    # -----------------------------------------------------------------------
    # Shortcut:  scores = trainer.predict(test_datasets, config=PredictConfig(...))
    # Manual:    the three stages predict() runs in sequence. Splitting them is
    #            useful when you want to (a) run translate once and then
    #            re-score with new metrics without re-decoding, (b) plug a
    #            custom decoder instance, or (c) inspect intermediate artifacts
    #            (`src.txt` / `ref.txt` / `hyp.txt` under `eval/<eval_ds>/beam<n>/`).
    metrics = {"bleu", "chrf"}
    beams = [5]

    # Custom decoder: bias beam search toward longer hypotheses. Try
    # TopPSampling / TopKSampling / MultinomialSampling for stochastic decoding.
    decoder = BeamSearch(length_penalty=1.2)

    # `filter_eval_datasets` runs internally inside predict(); doing it manually
    # here lets you log / skip / reorder eval datasets before translating.
    eval_datasets = trainer.filter_eval_datasets(test_datasets, eval_mode="compatible")

    scores = []  # one entry per eval_ds → matches the shape predict() returns
    for eval_ds in eval_datasets:
        # 7.a Decode hypotheses (writes hyp.tok → hyp.txt, src.txt, ref.txt).
        trainer.translate(
            eval_ds, beams=beams,
            preprocess_fn=preprocess_predict, force_overwrite=False,
            max_len_a=1.2, max_len_b=50,
            batch_size=64, max_tokens=None,
            devices="auto", accelerator="auto", num_workers=0,
            checkpoint="best",        # or "last", or a path/filename
            decoder=decoder,
        )

        # 7.b Score the existing hypotheses. Re-run this alone after the
        #     translate step if you only want to add a new metric.
        trainer.score_translations(
            eval_ds, beams=beams, metrics=metrics, force_overwrite=False,
        )

        # 7.c Parse the per-metric files on disk back into a dict matching
        #     the schema `generate_report` expects.
        run_scores = trainer.parse_metrics(
            eval_ds, beams=beams, metrics=metrics,
        )
        scores.append(run_scores)

    # -----------------------------------------------------------------------
    # (8) Build the report manually
    # -----------------------------------------------------------------------
    # Shortcut:  df_report, df_summary = generate_report(scores=[scores], output_path=out)
    # Manual:    transform → save → plot, step by step. Useful when you want to
    #            merge `df_report` with another experiment, add custom columns,
    #            or skip the disk artifacts entirely.
    out = f".outputs/07_manual/{datetime.datetime.now():%Y%m%d_%H%M%S}"
    reports_dir = os.path.join(out, "reports")
    plots_dir = os.path.join(out, "plots")
    fileio.make_dir([reports_dir, plots_dir])

    df_report = scores_to_dataframe([scores])
    df_summary = summarize_scores(df_report)

    fileio.save_json([scores], os.path.join(reports_dir, "report.json"))
    df_report.to_csv(os.path.join(reports_dir, "report.csv"), index=False)
    df_summary.to_csv(os.path.join(reports_dir, "report_summary.csv"), index=False)

    plot_model_comparison(
        df_report=df_report, out_dir=plots_dir,
        metric="translations.beam5.sacrebleu_bleu_score",
        xlabel="Run", ylabel="BLEU", title="Manual run (de→en)",
    )

    # -----------------------------------------------------------------------
    # (9) Inspect the checkpoint we just trained (handy for follow-ups)
    # -----------------------------------------------------------------------
    # `load_checkpoint(...)` accepts "best" / "last" / filename / absolute path.
    # `get_checkpoint_path("best")` returns the on-disk path without loading.
    ckpt_path = trainer.get_checkpoint_path(mode="best")
    print(f"\n[ckpt]  best checkpoint: {ckpt_path}")

    print(f"\nReport + plots saved to: {os.path.abspath(out)}\n")
    print(format_summary_table(df_summary))


if __name__ == "__main__":
    main()
