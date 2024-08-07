import datetime
import time
import os
import torch
torch.set_float32_matmul_precision("high")

from autonmt.modules.models import *
from autonmt.preprocessing import DatasetBuilder
from autonmt.toolkits import AutonmtTranslator
from autonmt.vocabularies import Vocabulary

from autonmt.bundle.report import generate_report
from autonmt.bundle.plots import plot_metrics

from autonmt.preprocessing.processors import preprocess_pairs, preprocess_lines, normalize_lines
from tokenizers.normalizers import NFKC, Strip, Lowercase

# Preprocess functions
normalize_fn = lambda x: normalize_lines(x, seq=[NFKC(), Strip()])
preprocess_raw_fn = lambda data, ds: preprocess_pairs(data["src"]["lines"], data["trg"]["lines"], normalize_fn=normalize_fn, min_len=1, max_len=None, remove_duplicates=False, shuffle_lines=False)
preprocess_splits_fn = lambda data, ds: preprocess_pairs(data["src"]["lines"], data["trg"]["lines"], normalize_fn=normalize_fn, shuffle_lines=False)
preprocess_predict_fn = lambda data, ds: preprocess_lines(data["lines"], normalize_fn=normalize_fn)

BASE_PATH = "/home/scarrion/datasets/translate"  # Remote
# BASE_PATH = "/Users/salvacarrion/Documents/Programming/datasets/translate"  # Remote

def main():
    # Create preprocessing for training
    builder = DatasetBuilder(
        # Root folder for datasets
        base_path=BASE_PATH,

        # Set of datasets, languages, training sizes to try
        datasets=[
            {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
            # {"name": "europarl", "languages": ["en-es"], "sizes": [("50k", 50000), ("100k", 100000), ("original", None)]},
        ],

        # Set of subword models and vocab sizes to try
        encoding=[
            # {"subword_models": ["bytes"], "vocab_sizes": [1000]},
            # {"subword_models": ["char"], "vocab_sizes": [1000]},
            # {"subword_models": ["bpe"], "vocab_sizes": [4000]},
            {"subword_models": ["word"], "vocab_sizes": [4000]},
        ],

        # Preprocessing functions
        preprocess_raw_fn=preprocess_raw_fn,
        preprocess_splits_fn=preprocess_splits_fn,

        # Additional args
        merge_vocabs=False,
    ).build(make_plots=False, force_overwrite=False, verbose=True)

    # Create preprocessing for training and testing
    tr_datasets = builder.get_train_ds()
    ts_datasets = builder.get_test_ds()

    # Train & Score a model for each dataset
    scores = []
    for i, train_ds in enumerate(tr_datasets, 1):
        for model_class in [SimpleRNN, ContextRNN, AttentionRNN, Conv, Transformer]:
            # Define max tokens (99.96% of the data)
            if train_ds.subword_model == "bytes":
                max_tokens_src, max_tokens_tgt = 539, 598
            elif train_ds.subword_model == "char":
                max_tokens_src, max_tokens_tgt = 540, 588
            elif train_ds.subword_model == "bpe":
                max_tokens_src, max_tokens_tgt = 106, 115
            elif train_ds.subword_model == "word":
                max_tokens_src, max_tokens_tgt = 99, 106
            else:
                raise ValueError(f"Unknown subword model: {train_ds.subword_model}")

            for iters in [1]:
                # Instantiate vocabs and model
                src_vocab = Vocabulary(max_tokens=max_tokens_src).build_from_ds(ds=train_ds, lang=train_ds.src_lang)
                trg_vocab = Vocabulary(max_tokens=max_tokens_tgt).build_from_ds(ds=train_ds, lang=train_ds.trg_lang)
                model = model_class(src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab), padding_idx=src_vocab.pad_id)

                # Define trainer
                runs_dir = train_ds.get_runs_path(toolkit="autonmt")
                run_prefix = f"{model.architecture}-{iters}ep__" + '_'.join(train_ds.id()[:2]).replace('/', '-')
                run_name = train_ds.get_run_name(run_prefix=run_prefix)  #+ f"__{int(time.time())}"
                trainer = AutonmtTranslator(model=model, src_vocab=src_vocab, trg_vocab=trg_vocab,
                                            runs_dir=runs_dir, run_name=run_name)

                # Print info
                total_params, trainable_params, no_trainable_params = model.count_parameters()
                print(f"=> Training model...")
                print(f"\t- TRAINING ({i}/{len(tr_datasets)}): {str(train_ds)}")
                print(f"\t- TESTING ({len(ts_datasets)}): {', '.join([str(x) for x in ts_datasets])}")
                print(f"\t- MODEL PREFIX: {run_prefix}")
                print(f"\t- MODEL ARCHITECTURE: {model.architecture}")
                print(f"\t- MODEL PARAMETERS:")
                print(f"\t\t- Total parameters: {total_params/10**6:,.2f}M")
                print(f"\t\t- Trainable parameters: {trainable_params/10**6:,.2f}M")
                print(f"\t\t- No trainable parameters: {no_trainable_params/10**6:,.2f}M")

                # Train model
                wandb_params = None  #dict(project="architecture", entity="salvacarrion", reinit=True)
                # trainer.fit(train_ds, max_epochs=iters, learning_rate=0.001, optimizer="adam", batch_size=128, seed=1234,
                #             patience=10, num_workers=0, accelerator="auto", strategy="auto", save_best=True, save_last=True, print_samples=1,
                #             wandb_params=wandb_params, use_bucketing=False)

                # Test model
                m_scores = trainer.predict(ts_datasets, metrics={"bleu"}, beams=[1], load_checkpoint="best",
                                           preprocess_fn=preprocess_predict_fn, eval_mode="compatible", force_overwrite=True)
                for ms in m_scores:
                    ms['train_dataset'] = train_ds.dataset_name
                    ms['vocab__merged'] = train_ds.merge_vocabs
                    ms['max_iters'] = str(iters)
                    ms['train_dataset'] = str(train_ds)
                    ms['architecture'] = model.architecture
                scores.append(m_scores)

    # Make report
    output_path = os.path.join(BASE_PATH, f".outputs/autonmt/multi30k")
    df_report, df_summary = generate_report(scores=scores, output_path=output_path)

    # Print summary
    print("Summary:")
    print(df_summary.to_string(index=False))

    # Plot metrics
    plots_path = os.path.join(output_path, "plots")
    plot_metrics(output_path=plots_path, df_report=df_report, plot_metric="translations.beam1.sacrebleu_bleu_score",
                 xlabel="MT Models", ylabel="BLEU Score", title="Model comparison")


if __name__ == "__main__":
    main()
