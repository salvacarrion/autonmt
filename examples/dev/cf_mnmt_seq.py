import datetime
import os.path
import random

from tokenizers import normalizers
from tokenizers.normalizers import NFKC, Strip, Lowercase

from autonmt.bundle.report import generate_report
from autonmt.preprocessing import DatasetBuilder
from autonmt.modules.models import Transformer
from autonmt.modules.seq2seq import LitSeq2Seq
from autonmt.toolkits.autonmt_v2 import AutonmtTranslatorV2
from autonmt.vocabularies import Vocabulary
from autonmt.bundle.utils import make_dir
import torch
import numpy as np

from collections import defaultdict
# class LitSeq2SeqV2(LitSeq2Seq):
#     pass
#
#
# class TransformerV2(LitSeq2SeqV2):
#     pass



def main(prob_old_tr):
    DS_LANGS = ["en-es", "en-fr", "en-de"]

    def filter_train_fn(src_lines, trg_lines, filter_langs):
        filter_langs = [None] if filter_langs is None else filter_langs  # Fix non-lists

        tr_lang = filter_langs[0]  # Trick
        filter_langs = DS_LANGS[:DS_LANGS.index(tr_lang)+1]
        filter_probs = {lang: 1.0 if lang == tr_lang else prob_old_tr for lang in filter_langs}

        # Filter by language and split
        datasets = defaultdict(list)
        for src_line, trg_line in zip(src_lines, trg_lines):
            lang_code = src_line[:20].replace(' ', '').replace('▁', '').strip().lower()[1:6]
            if None in filter_langs or lang_code in filter_langs:
                if random.random() < filter_probs[lang_code]:  # Filter % samples
                    datasets[lang_code].append((src_line, trg_line))

        # Compute coefficients
        samples_by_lang = [len(datasets[lang]) for lang in datasets.keys()]
        ratios_per_batch = [len(datasets[lang])/max(samples_by_lang) for lang in datasets.keys()]
        coeff_by_lang = [1/ratio for ratio in ratios_per_batch]

        # Get oversampled dataset
        new_samples_by_lang = {lang: round(count*coeff) for lang, count, coeff, in zip(datasets.keys(), samples_by_lang, coeff_by_lang)}
        lines = []
        for lang in datasets.keys():
            lines += datasets[lang]  # Add all

            # Oversample rest
            target_lines = new_samples_by_lang[lang]
            missing = max(target_lines - len(datasets[lang]), 0)  # Just in case
            lines += random.choices(datasets[lang], k=missing)

        # Shuffle lines
        random.shuffle(lines)

        # Split src and trt lines
        src_lines, trg_lines = list(zip(*lines))
        return src_lines, trg_lines

    def filter_eval_fn(src_lines, trg_lines, filter_langs):
        filter_langs = [None] if filter_langs is None else filter_langs  # Fix non-lists

        # Filter by language and split
        datasets = defaultdict(list)
        filter_langs = set(filter_langs)
        for src_line, trg_line in zip(src_lines, trg_lines):
            lang_code = src_line[:20].replace(' ', '').replace('▁', '').strip().lower()[1:6]
            if None in filter_langs or lang_code in filter_langs:
                datasets[lang_code].append((src_line, trg_line))

        # Get lines
        lines = []
        for lang in datasets.keys():
            lines += datasets[lang]  # Add all

        # Shuffle lines
        random.shuffle(lines)

        # Split src and trt lines
        src_lines, trg_lines = list(zip(*lines))
        return src_lines, trg_lines

    # Create preprocessing for training
    builder = DatasetBuilder(
        base_path="/home/scarrion/datasets/nn/translation",
        datasets=[
            # {"name": "europarl_cf", "languages": ["es-en", "fr-en", "de-en"], "sizes": [("500k", 500000), ("100k", 100000), ("10k", 10000)], "split_sizes": (None, 1000, 1000)},
            #{"name": "europarl_cf", "languages": ["xx-yy"], "sizes": [("1500k", 1500000), ("300k", 300000), ("30k", 30000)], "split_sizes": (None, 1000, 1000)},  #("1500k", 1500000), ("300k", 300000), ("30k", 30000)

            {"name": "europarl_cf", "languages": ["en-xx"], "sizes": [("300k", 300000)], "split_sizes": (None, 1500, 1500)},  #("1500k", 1500000), ("300k", 300000), ("30k", 30000)
        ],
        encoding=[
            # {"subword_models": ["bytes", "char+bytes"], "vocab_sizes": [1000]},
            # {"subword_models": ["unigram+bytes", "word+bytes"], "vocab_sizes": [8000, 16000, 24000]},

            {"subword_models": ["unigram+bytes"], "vocab_sizes": [16000]},
        ],
        normalizer=lambda x: normalizers.Sequence([NFKC(), Strip(), Lowercase()]).normalize_str(x),
        merge_vocabs=False,
        eval_mode="same",
    ).build(make_plots=False, force_overwrite=False)

    # Create preprocessing for training and testing
    tr_datasets = builder.get_train_ds()
    ts_datasets = builder.get_test_ds()

    # Instantiate vocabs and model
    default_ds = tr_datasets[0]
    src_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=default_ds, lang=default_ds.src_lang)
    trg_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=default_ds, lang=default_ds.trg_lang)
    # checkpoint_path = "mymodels/baselines/2_en-es/2_tr(en-es)_last.pt"  # local
    checkpoint_path = "/home/scarrion/projects/autonmt/mymodels/baselines/2_en-es/2_tr(en-es)_last.pt"  # remote
    # checkpoint_path = "/home/scarrion/projects/autonmt/mymodels/baselines/3_en-fr/3_tr(en-fr)_last.pt"  # remote

    # Train & Score a model for each dataset
    scores = []
    tr_langs_acc = ["en-es"]

    # Filter languages
    tr_pairs_seq = [["en-fr"], ["en-de"]]
    ts_pairs = [None, ["en-es"], ["en-fr"], ["en-de"]]

    alias = "seq"
    for i, tr_pairs in enumerate(tr_pairs_seq, 1):
        tr_pairs_i_str = 'all' if tr_pairs is None else '+'.join(tr_pairs)
        ts_pairs_str = '|'.join(["all" if x is None else '+'.join(x) for x in ts_pairs])
        tr_langs_acc.append(tr_pairs_i_str)

        # Create path
        prefix = f"{str(i)}_{','.join(tr_langs_acc)}_pp{prob_old_tr}_physically_oversampled"
        m_path = os.path.join("mymodels", alias)
        make_dir([m_path])

        print(f"=> Training model... (ID={prefix})")
        print(f"\t- TRAINING ({i}/{len(tr_pairs_seq)}): {tr_pairs_i_str} (hist.: {','.join(tr_langs_acc)})")
        print(f"\t- TESTING ({len(ts_pairs)}): {ts_pairs_str}")
        print(f"\t- MODEL PREFIX: {prefix}")
        print(f"\t- MODEL PATH: {m_path}")
        print(f"\t- PREV. PROB: {prob_old_tr}")

        # Set model
        t_model = Transformer(src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab), padding_idx=src_vocab.pad_id)
        if checkpoint_path:
            print(f"\t- Loading checkpoint: {checkpoint_path}")
            model_state_dict = torch.load(checkpoint_path)
            model_state_dict = model_state_dict.get("state_dict", model_state_dict)
            t_model.load_state_dict(model_state_dict)

        wandb_params = dict(project="autonmt", entity="salvacarrion")
        model = AutonmtTranslatorV2(model=t_model, src_vocab=src_vocab, trg_vocab=trg_vocab,
                                    wandb_params=wandb_params, run_prefix=prefix,
                                    load_best_checkpoint=False, print_samples=3)

        # Set filters for multilingual/continual learning (sequential tasks)
        model.filter_train = tr_pairs
        model.filter_eval = ts_pairs  # Independent evaluation at log
        model.filter_train_fn = filter_train_fn
        model.filter_eval_fn = filter_eval_fn

        # Use multilingual val/test and then filter
        if len(ts_pairs) <= 1:
            monitor = "val_loss"
        else:
            dataloader_idx = ts_pairs.index(tr_pairs)  # 0 for global
            monitor = "val_"
            monitor += 'all' if ts_pairs[dataloader_idx] is None else '+'.join(ts_pairs[dataloader_idx])
            monitor += f"_loss/dataloader_idx_{dataloader_idx}"
        print(f"\t- MONITOR: {monitor}")

        # Train
        model.fit(default_ds, max_epochs=25, learning_rate=0.0001, optimizer="adam", batch_size=96, seed=1234,
                  patience=5, num_workers=12,  monitor=monitor, devices="auto", accelerator="auto", strategy="ddp")  #val_loss, 'val_all_loss/dataloader_idx_0'

        # Save model
        checkpoint_path = os.path.join(m_path, prefix + "_last.pt")
        print(f"\t- Saving current model at: {checkpoint_path}")
        torch.save(t_model.state_dict(), checkpoint_path)
        # checkpoint_path = None
        # tr_langs_acc = []
        asd = 3

        # # Get predictions
        # m_scores = model.predict(ts_datasets, model_ds=default_ds, metrics={"bleu"}, beams=[1], load_best_checkpoint=False)  # model_ds=train_ds => if fit() was not used before
        # scores.append(m_scores)
        asd = 3

    # # Make report and print it
    # output_path = f".outputs/autonmt/{str(datetime.datetime.now())}"
    # for i in range(len(ts_pairs)):
    #     print(f"EVAL PAIR: {ts_pairs[i]} ({i})")
    #     df_report, df_summary = generate_report(scores=scores, output_path=output_path, plot_metric=f"beam1-{i}__sacrebleu_bleu_score")
    #     print("Summary:")
    #     print(df_summary.to_string(index=False))
    #     print("-"*100)


if __name__ == "__main__":
    for p in [0.01, 0.05, 0.1, 0.25]:
        main(p)
    print("Done!")