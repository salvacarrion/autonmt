import datetime

from tokenizers import normalizers
from tokenizers.normalizers import NFKC, Strip, Lowercase

from autonmt.bundle.report import generate_report
from autonmt.modules.models import Transformer
from autonmt.preprocessing import DatasetBuilder
from autonmt.toolkits import AutonmtTranslator
from autonmt.vocabularies import Vocabulary
from autonmt.toolkits.fairseq import FairseqTranslator


def main(fairseq_args=None, venv_path=None):
    # Create preprocessing for training
    builder = DatasetBuilder(
        base_path="/home/scarrion/datasets/nn/translation",
        datasets=[
            # {"name": "cf", "languages": ["es-en", "fr-en", "de-en"], "sizes": [("100k", 100000)]},
            # {"name": "cf", "languages": ["es-en", "fr-en", "de-en"], "sizes": [("100k", 100000)]},
            {"name": "cf/multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
            # {"name": "cf/euro30k", "languages": ["es-en"], "sizes": [("30k", 30000)], "split_sizes": (None, 1000, 1000)},
        ],
        encoding=[
            # {"subword_models": ["word", "unigram+bytes", "char+bytes"], "vocab_sizes": [8000, 16000]},
            # {"subword_models": ["word", "unigram+bytes"], "vocab_sizes": [8000, 16000, 32000]},
            # {"subword_models": ["char", "unigram+bytes"], "vocab_sizes": [8000]},
            {"subword_models": ["unigram"], "vocab_sizes": [4000]},
        ],
        normalizer=normalizers.Sequence([NFKC(), Strip(), Lowercase()]).normalize_str,
        merge_vocabs=False,
        eval_mode="same",
    ).build(make_plots=False, force_overwrite=False)

    # Create preprocessing for training and testing
    tr_datasets = builder.get_train_ds()
    ts_datasets = builder.get_test_ds()

    # Train & Score a model for each dataset
    scores = []
    use_custom = False
    for train_ds in tr_datasets:
        if use_custom:
            # Instantiate vocabs and model
            src_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=train_ds, lang=train_ds.src_lang)
            trg_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=train_ds, lang=train_ds.trg_lang)
            model = Transformer(src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab), padding_idx=src_vocab.pad_id)
            model = AutonmtTranslator(model=model, src_vocab=src_vocab, trg_vocab=trg_vocab)
            model.fit(train_ds, max_epochs=10, learning_rate=0.001, optimizer="adam", batch_size=128, max_tokens=None, seed=1234, patience=10, num_workers=12, strategy="dp", force_overwrite=True)
        else:
            model = FairseqTranslator(venv_path=venv_path)
            model.fit(train_ds, max_epochs=10, learning_rate=0.001, optimizer="adam", batch_size=128, max_tokens=None, seed=1234, patience=10, num_workers=12, fairseq_args=fairseq_args, force_overwrite=True)

        m_scores = model.predict(ts_datasets, metrics={"bleu"}, beams=[1], model_ds=train_ds, force_overwrite=True)
        scores.append(m_scores)

    # Make report and print it
    output_path = f".outputs/cf/fairseq"
    df_report, df_summary = generate_report(scores=scores, output_path=output_path, plot_metric="beam1__sacrebleu_bleu_score")
    print("Summary:")
    print(df_summary.to_string(index=False))



if __name__ == "__main__":
    # These args are pass to fairseq using our pipeline
    # Fairseq Command-line tools: https://fairseq.readthedocs.io/en/latest/command_line_tools.html
    fairseq_cmd_args = [
        "--arch transformer",
        "--encoder-embed-dim 256",
        "--decoder-embed-dim 256",
        "--encoder-layers 3",
        "--decoder-layers 3",
        "--encoder-attention-heads 8",
        "--decoder-attention-heads 8",
        "--encoder-ffn-embed-dim 512",
        "--decoder-ffn-embed-dim 512",
        "--dropout 0.1",
        "--no-epoch-checkpoints",
        "--maximize-best-checkpoint-metric",
        "--best-checkpoint-metric bleu",
        "--eval-bleu",
        "--eval-bleu-print-samples",
        "--scoring sacrebleu",
        "--log-format simple",
        "--task translation",
    ]

    venv_path = "/home/scarrion/.venvs/fairseq/bin/activate"  # To speed-up training

    # Run grid
    main(fairseq_args=fairseq_cmd_args, venv_path=None)

