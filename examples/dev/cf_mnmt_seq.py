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

# class LitSeq2SeqV2(LitSeq2Seq):
#     pass
#
#
# class TransformerV2(LitSeq2SeqV2):
#     pass


def filter_fn(line, lang_pair):
    if lang_pair is None:
        return True
    else:
        tmp = line[:20].replace(' ', '').replace('▁', '').strip().lower()
        return tmp.startswith(f"<{lang_pair.lower().strip()}>")

def main():
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
    checkpoint_path = None #'mymodels/seq/2022-05-25 19:12:18.696802/1_en-es/1_tr(en-es)_last.pt'

    # Train & Score a model for each dataset
    scores = []
    tr_langs_acc = []

    # Filter languages
    tr_pairs_seq = [None, ["en-es"], ["en-fr"], ["en-de"]]
    ts_pairs = [None, ["en-es"], ["en-fr"], ["en-de"]]

    mid = str(datetime.datetime.now())
    alias = "baselines"
    for i, tr_pairs in enumerate(tr_pairs_seq, 1):
        tr_pairs_i_str = 'all' if tr_pairs is None else '+'.join(tr_pairs)
        ts_pairs_str = '|'.join(["all" if x is None else '+'.join(x) for x in ts_pairs])
        tr_langs_acc.append(tr_pairs_i_str)

        # Create path
        m_path = os.path.join("mymodels", alias, str(mid), f"{str(i)}_{','.join(tr_langs_acc)}")
        make_dir([m_path])

        prefix = f"{i}_tr({tr_pairs_i_str})"
        print(f"=> Training model... (ID={mid}-{i})")
        print(f"\t- TRAINING ({i}/{len(tr_pairs_seq)}): {tr_pairs_i_str} (hist.: {','.join(tr_langs_acc)})")
        print(f"\t- TESTING ({len(ts_pairs)}): {ts_pairs_str}")
        print(f"\t- MODEL PREFIX: {prefix}")
        print(f"\t- MODEL PATH: {m_path}")

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
        model.filter_fn = filter_fn

        # Use multilingual val/test and then filter
        if len(ts_pairs) <= 1:
            monitor = "val_loss"
        else:
            dataloader_idx = i-1
            monitor = "val_"
            monitor += 'all' if ts_pairs[dataloader_idx] is None else '+'.join(ts_pairs[dataloader_idx])
            monitor += f"_loss/dataloader_idx_{dataloader_idx}"

        # Train
        model.fit(default_ds, max_epochs=100, learning_rate=0.0001, optimizer="adam", batch_size=96, seed=1234,
                  patience=10, num_workers=0,  monitor=monitor, devices="auto", accelerator="auto", strategy="ddp")  #val_loss, 'val_all_loss/dataloader_idx_0'

        # Save model
        checkpoint_path = os.path.join(m_path, prefix + "_last.pt")
        print(f"\t- Saving current model at: {checkpoint_path}")
        torch.save(t_model.state_dict(), checkpoint_path)
        checkpoint_path = None
        tr_langs_acc = []

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
    main()
    print("Done!")