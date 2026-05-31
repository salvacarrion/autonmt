"""
============================================================================
 Advanced 05 — Cross-evaluation matrix (domain transfer heatmap)
============================================================================

What you'll learn
-----------------
How to turn a grid into a **train × test matrix** and visualise it as a heatmap
with `Report.plot_matrix(...)`. The canonical use is *domain transfer*: train one
model per domain, evaluate **every** model on **every** test set, and read the
off-diagonal cells as "how well does a model trained on A do on B?".

The mechanics that make it a matrix
-----------------------------------
- Two datasets with the **same language pair** (so the models are mutually
  evaluable): here two toy es-en domains, "weather" and "tech", with
  deliberately disjoint vocabulary.
- `eval_mode="all"` — each trained model is scored on *all* test sets, not just
  its own. That's what fills the off-diagonal.
- `Report.plot_matrix("bleu", rows="train_dataset", cols="test_dataset")` pivots
  the flat report into a `train_dataset × test_dataset` grid and draws the
  heatmap. `train_dataset` is the domain a model trained on; `test_dataset` is
  the domain it was scored on.

What's new vs the basics grid (04/05)
-------------------------------------
Those sweep tokenization/size on **one** dataset and compare cells with a bar
chart. Here the grid axis is the **dataset/domain itself**, and the right view is
a matrix, not bars.

Expected output
---------------
Like every example here, this proves the *recipe*, not a publishable result:
toy corpora + 2 epochs give low, noisy BLEU. With real corpora and adequate
training you'd expect the **diagonal to dominate** (in-domain > cross-domain).

Run
---
    python examples/advanced/05_cross_eval_matrix.py
"""
import datetime
import os

from tokenizers.normalizers import NFKC, Strip

from autonmt.backends import AutonmtTranslator
from autonmt.backends._base.config import FitConfig, PredictConfig
from autonmt.core.nn.models import Transformer
from autonmt.datasets import DatasetBuilder
from autonmt.datasets.preprocessing import normalize_lines, preprocess_lines, preprocess_pairs
from autonmt.reporting.report import Report

BASE_PATH = "datasets/adv_05_xeval"
LANG_PAIR = "es-en"
SIZE = "original"

# Two domains with deliberately disjoint vocabulary, so cross-domain transfer is
# visible. In a real project these would be two genuine corpora (e.g. news vs
# medical) materialised the same way — see Tutorial 02 for the on-disk layout.
DOMAINS = {
    "weather": [
        ("Hoy hace sol.", "It is sunny today."),
        ("Mañana lloverá.", "It will rain tomorrow."),
        ("El cielo está nublado.", "The sky is cloudy."),
        ("Hace mucho viento.", "It is very windy."),
        ("La temperatura baja por la noche.", "The temperature drops at night."),
        ("Hay niebla en la montaña.", "There is fog in the mountains."),
        ("El verano es caluroso.", "The summer is hot."),
        ("En invierno nieva.", "In winter it snows."),
    ],
    "tech": [
        ("El ordenador no enciende.", "The computer does not turn on."),
        ("Reinicia el servidor.", "Restart the server."),
        ("La red está caída.", "The network is down."),
        ("Actualiza el software.", "Update the software."),
        ("El disco está lleno.", "The disk is full."),
        ("Instala la nueva versión.", "Install the new version."),
        ("El código tiene un error.", "The code has a bug."),
        ("Guarda el archivo.", "Save the file."),
    ],
}


def bootstrap_domains():
    """Write each domain's train/val/test splits into the expected layout.

    Mirrors Tutorial 02's `bootstrap_local_data`, once per domain. Real users
    replace this with their own parallel corpora.
    """
    for domain, base_pairs in DOMAINS.items():
        splits_dir = os.path.join(BASE_PATH, domain, LANG_PAIR, SIZE, "data", "1_splits")
        if os.path.exists(os.path.join(splits_dir, "train.es")):
            continue  # already bootstrapped on a previous run.

        os.makedirs(splits_dir, exist_ok=True)
        pairs = base_pairs * 250  # repeat so SPM has enough material
        train, val, test = pairs[:1600], pairs[1600:1800], pairs[1800:2000]
        for split_name, split in [("train", train), ("val", val), ("test", test)]:
            for lang_idx, lang in enumerate(["es", "en"]):
                path = os.path.join(splits_dir, f"{split_name}.{lang}")
                with open(path, "w", encoding="utf-8") as f:
                    f.write("\n".join(row[lang_idx] for row in split) + "\n")
        print(f"[bootstrap] '{domain}' corpus written to: {splits_dir}")


def normalize(lines):
    return normalize_lines(lines, seq=[NFKC(), Strip()])


def preprocess_train(data, ds):
    return preprocess_pairs(data["src"]["lines"], data["tgt"]["lines"], normalize_fn=normalize)


def preprocess_predict(data, ds):
    return preprocess_lines(data["lines"], normalize_fn=normalize)


def main():
    bootstrap_domains()

    # One builder, two datasets (the two domains). Same encoding for both so the
    # only axis that varies is the domain.
    builder = DatasetBuilder(
        base_path=BASE_PATH,
        datasets=[
            {"name": "weather", "languages": [LANG_PAIR], "sizes": [(SIZE, None)]},
            {"name": "tech", "languages": [LANG_PAIR], "sizes": [(SIZE, None)]},
        ],
        encoding=[{"subword_models": ["bpe"], "vocab_sizes": [500]}],
        preprocess_raw_fn=preprocess_train,
        preprocess_splits_fn=preprocess_train,
        merge_vocabs=False,
    ).build(force_overwrite=False)

    train_datasets = builder.get_train_ds()
    test_datasets = builder.get_test_ds()      # BOTH domains' test sets

    # Train one model per domain; score each on EVERY test set (eval_mode="all").
    scores = []
    for train_ds in train_datasets:
        src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=150)
        model = Transformer.from_vocabs(src_vocab, tgt_vocab)

        trainer = AutonmtTranslator.from_dataset(
            train_ds, model=model,
            src_vocab=src_vocab, tgt_vocab=tgt_vocab,
            run_prefix="xeval",
        )
        trainer.fit(train_ds, config=FitConfig(max_epochs=2, batch_size=32,
                                               learning_rate=1e-3, seed=42))
        scores.append(trainer.predict(
            test_datasets,
            config=PredictConfig(
                metrics={"bleu"}, beams=[5],
                load_checkpoint="best",
                preprocess_fn=preprocess_predict,
                eval_mode="all",          # <- score every model on every domain
            ),
        ))

    # The matrix: rows = the domain a model trained on, cols = the domain it was
    # scored on. The single beam is inferred (no beam= needed).
    out = f".outputs/adv_05_xeval/{datetime.datetime.now():%Y%m%d_%H%M%S}"
    report = Report.from_runs(scores, output_path=out).save()
    report.plot_matrix(
        "bleu", rows="train_dataset", cols="test_dataset",
        title="Cross-domain transfer (BLEU)",
    )

    print(f"\nReport + matrix saved to: {os.path.abspath(out)}\n")
    print(report)
    print(
        "\nRead the heatmap: diagonal = in-domain, off-diagonal = transfer. With\n"
        "real corpora the diagonal should dominate; a high off-diagonal means the\n"
        "two domains share enough that one model generalises to the other."
    )


if __name__ == "__main__":
    main()
