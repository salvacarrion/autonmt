# On-disk layout

AutoNMT is **path-driven**: every stage of the pipeline writes its output to a numbered
folder before the next stage reads it. Understanding the layout is the key to debugging,
because when a result looks wrong you can open the actual intermediate files.

Everything for one cell lives under `base_path/<dataset>/<lang-pair>/<size>/`:

```text
multi30k/de-en/original/
├── data/
│   ├── 0_raw/                       data.de, data.en
│   ├── 1_splits/                    train|val|test.{de,en}
│   ├── 2_preprocessed/              normalized splits
│   ├── 3_pretokenized/              moses-tokenized (subword = word only)
│   └── 4_encoded/<subword>/<vocab>/ SentencePiece-encoded splits
├── vocabs/<subword>/<vocab>/        SentencePiece .model + .vocab
├── stats/<subword>/<vocab>/         per-split token statistics
├── plots/<subword>/<vocab>/         length / frequency plots
└── models/<toolkit>/runs/<run>/
    ├── checkpoints/                 *.pt
    ├── eval/<eval_ds>/              decoded src/ref/hyp + per-metric scores
    └── logs/                        config_train.json, config_predict.json, TB / wandb
```

## The data stages

The `data/` subfolders are numbered in pipeline order, splitting cleanly into
**subword-agnostic** and **subword-dependent** stages:

| Stage                          | Produced by                                                              | Subword-dependent? |
| ------------------------------ | ------------------------------------------------------------------------ | ------------------ |
| `0_raw/`                       | `download_hf_dataset` or your own files                                  | no                 |
| `1_splits/`                    | the builder's split logic                                                | no                 |
| `2_preprocessed/`              | `preprocess_raw_fn` / `preprocess_splits_fn` (filter, normalize, dedupe) | no                 |
| `3_pretokenized/`              | Moses pretokenizer - **only when `subword=word`**                        | no                 |
| `4_encoded/<subword>/<vocab>/` | SentencePiece / bytes encoding                                           | **yes**            |

Because the first stages are subword-agnostic, sweeping vocab sizes or subword models
reuses the same `0_raw`–`2_preprocessed` files and only re-runs `4_encoded`. That's what
makes a grid cheap.

!!! note "Language code is the file extension"
Split files use the language code as the **extension** (`train.es`, `train.en`), never
as a directory.

## Byte fallback never collides

When an entry sets `byte_fallback=True` (or uses the `+bytes` shorthand), the `<subword>`
segment becomes `<model>+bytes`, e.g. `bpe+bytes/8000`. Runs with and without fallback
therefore live at different paths and never overwrite each other.

## Runs live next to the data

Trained models land under `models/<toolkit>/runs/<run_name>/`. The `<toolkit>` segment
(`autonmt`, `huggingface`, `fairseq`) keeps backends separate, and each run keeps its
checkpoints, its decoded evaluation artifacts (`eval/<eval_ds>/`), and its logs together.
`AutonmtTranslator.from_dataset(...)` resolves this path automatically from the dataset
cell.

## Resumability and the golden debugging rule

Each stage checks `force_overwrite` before writing. Re-running an experiment **skips
stages that already exist on disk**, so a crashed run continues where it stopped, and
adding a new grid cell doesn't recompute the old ones.

!!! tip "Delete the stage, not the tree"
When you need to redo a stage - say you changed your normalization and want to
re-encode - **delete only that stage's directory** (e.g. `data/4_encoded/...`), then
re-run. Deleting the whole cell forces every stage to recompute from scratch.

You can pass `force_overwrite=True` to `build()` to ignore existing artifacts and rewrite
everything.
