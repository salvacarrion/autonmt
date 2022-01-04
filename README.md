# AutoNMT

![AutoNMT Logo](https://github.com/salvacarrion/autonmt/raw/main/docs/images/logos/logo.png)

 *Scikit-learn but for Seq2Seq tasks*

--------------------------------------------------------------------------------

[![Build](https://github.com/salvacarrion/autonmt/actions/workflows/python-package.yml/badge.svg)](https://github.com/salvacarrion/autonmt/actions/workflows/python-package.yml)
![GitHub](https://img.shields.io/github/license/salvacarrion/autonmt)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/salvacarrion/autonmt)


**AutoNMT** is a Python library that allows you to research seq2seq models by providing two high-level features:

- Automates the grid experimentation: Tokenization, training, scoring, logging, plotting, stats, file management, etc.
- Toolkit abstraction: Use your models, our models, or different libraries such as Fairseq or OpenNMT, changing a single of code.


**Why us?**

We follow an **almost-no code** approach so that you can remain focused on your research without wasting time learning 
the inner workings of another toolkit.


**Reproducibility**

Reproducibility is a must in research. Because of that, we only use reference libraries that produce shareable, 
comparable, and reproducible results such as Sacrebleu, Moses or SentencePiece.

Furthermore, we provide two features for reproducibility:

- All the intermediate steps (raw files, tokenized, binaries, scores,...) are saved in separated folders so that a user can inspect any part of the process (and reuse it in other projects) and, we also output all the commands use to compute
- We also output the exact commands used for the reference libraries, so you can replicate any part of the process for yourself


## Installation

Requires Python +3.6

```
git clone git@github.com:salvacarrion/autonmt.git
pip install -e autonmt/
```


## Usage

You check full examples [here](examples).

### Dataset generation

The `DatasetBuilder` is the object in charge of generating variants of your datasets ("original"). In other words, it 
creates versions of your original dataset, with different training sizes to test ideas faster, multiple vocabularies
lengths and variations (bytes, char+bytes, unigram, bpe, word), etc.

If you don't know how to use it, don't worry. Run the following code with the `interactive=True` argument enabled, 
and it will guide you step-by-step so that you can create a new dataset.


```python
from autonmt.preprocessing import DatasetBuilder

# Create preprocessing for training
builder = DatasetBuilder(
    base_path="/home/preprocessing/",
    datasets=[
        {"name": "scielo/biological", "languages": ["es-en"], "sizes": [("original", None), ("100k", 100000)]},
        {"name": "scielo/health", "languages": ["es-en"], "sizes": [("original", None), ("100k", 100000)]},
    ],
    subword_models=["bytes", "char+bytes", "char", "unigram", "word"],
    vocab_sizes=[8000],
    merge_vocabs=True,
    eval_mode="compatible", # {same, compatible}
).build(make_plots=True)

# Create preprocessing for testing
tr_datasets = builder.get_ds()
ts_datasets = builder.get_ds(ignore_variants=True)
```

> **Note:**
> 
> The `eval_model` indicates the datasets for which each model can be evaluated:
> -  `same`: evaluates a model only with its test set.
> -  `compatible`: evaluates a model with all compatible test sets}

#### Format

Once you've run the above code, the program will tell you where to put your files. Nevertheless, it expects that all 
files contain one sample per line and their language code as their extension. 

For instance, there are two ways to create a dataset: 
- From raw files: `data.es` and `data.en` => `train.es`, `train.en`, `val.es`, `val.en`, `test.es` and `test.en`.
- From split files: `train.es`, `train.en`, `val.es`, `val.en`, `test.es` and `test.en`.

### Train & Score

The `Translator` object abstracts the seq2seq pipeline so that you can train and score your custom models effortless. Similarly, 
you can use other engines such as `fairseq` or `opennmt`.

```python
from autonmt.toolkits import AutonmtTranslator
from autonmt.modules.models import Transformer
from autonmt.vocabularies import Vocabulary

# Train & Score a model for each dataset
scores = []
for ds in tr_datasets:
    
    # Instantiate vocabs and model
    src_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=ds, lang=ds.src_lang)
    trg_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=ds, lang=ds.trg_lang)
    model = Transformer(src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab), padding_idx=src_vocab.pad_id)

    # Train model
    model = AutonmtTranslator(model=model, src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.fit(train_ds=ds, max_epochs=10, learning_rate=0.001, criterion="cross_entropy", optimizer="adam", 
              max_tokens=None, batch_size=64, patience=10, seed=1234, devices=1)
    m_scores = model.predict(ts_datasets, metrics={"bleu", "chrf", "bertscore"}, beams=[1])
    scores.append(m_scores)
```


The evaluation will be performed only the compatible datasets (same source and target language).

It is worth to point out that the DatasetBuilder will create n variations for each unique dataset, therefore, n models
per unique dataset will be trained. Nevertheless, AutoNMT is smart enough to evaluate each model once per unique dataset
as the raw test files of each variation are the same (each model will encode its data).

#### Fit parameters

Our models are wrapped using Pytorch Lightning. Therefore, you can pass to the fit function all the parameters that 
training of pytorch lightning accepts (and more!).

Check the available parameters [here](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html)

### Generate a report

This code will save the score of the multiple training (as json and csv), will create a summary of the results and will 
make a few plots to visualize the performance of the models.

```python
from autonmt.bundle.report import generate_report

# Make report
df_report, df_summary = generate_report(scores=scores, output_path=".outputs", plot_metric="beam1__sacrebleu_bleu_score")

print("Summary:")
print(df_summary.to_string(index=False))
```

**Single toolkit output:**

```text
train_dataset eval_dataset subword_model vocab_size  fairseq_bleu
     europarl     europarl          word       1000     17.465943
     europarl     europarl          word       8000     11.431010
```

**Multi-toolkit output:**

```text
train_dataset eval_dataset subword_model vocab_size  custom_bleu  fairseq_bleu
     europarl     europarl          word       1000     17.218436     17.465943
     europarl     europarl          word       8000     12.080312     11.431010
```


### Toolkit abstraction

#### Custom models

To create your custom pytorch model, you only need inherit from `Seq2Seq` and then pass it as parameter to the `Translator` class. 
The only requirement is that the forward must return a tensor with shape `(batch, length, probabilities)`.

```python
from autonmt.toolkits import AutonmtTranslator
from autonmt.modules.seq2seq import LitSeq2Seq


class CustomModel(LitSeq2Seq):
    def __init__(self, src_vocab_size, trg_vocab_size, padding_idx, **kwargs):
        super().__init__(src_vocab_size, trg_vocab_size, padding_idx, **kwargs)

    def forward_encoder(self, x):
        pass
    
    def forward_decoder(self, y, memory):
        pass  # output = (Batch, Length, probabilities)

# Train & Score a model for each dataset
for train_ds in tr_datasets:
    
        # Instantiate vocabs and model
        src_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=ds, lang=ds.src_lang)
        trg_vocab = Vocabulary(max_tokens=150).build_from_ds(ds=ds, lang=ds.trg_lang)
        model = CustomModel(src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab), padding_idx=src_vocab.pad_id)

        # Train model
        model = AutonmtTranslator(model=model, src_vocab=src_vocab, trg_vocab=trg_vocab)
        model.fit(train_ds=ds)
```

**Custom trainer/evaluator**

Our Seq2Seq base model is simply a wrapper that uses PyTorchLightning. If you need to write your custom Seq2Seq trainer, you can do it like this:

```python
import pytorch_lightning as pl


class LitCustomSeq2Seq(pl.LightningModule):
    # Stuff for Pytorch Lightning modules


class CustomModel(LitCustomSeq2Seq):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# Custom model
model = AutonmtTranslator(model= CustomModel(...), ...)
```

#### Fairseq models

When using a Fairseq model, you can use it through the fairseq command-line tools:

```python
# These args are pass to fairseq using our pipeline
# Fairseq Command-line tools: https://fairseq.readthedocs.io/en/latest/command_line_tools.html
fairseq_args = [
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
]

# Train & Score a fairseq model for each dataset
scores = []
for ds in tr_datasets:
    model = FairseqTranslator(conda_fairseq_env_name="fairseq")  # conda envs will be soon deprecated
    model.fit(train_ds=ds, max_epochs=1, learning_rate=0.001, criterion="cross_entropy", optimizer="adam",
              max_tokens=None, batch_size=64, patience=10, seed=1234, devices=1, fairseq_args=fairseq_args)
    m_scores = model.predict(ts_datasets, metrics={"bleu", "chrf", "bertscore", "comet"}, beams=[1, 5])
    scores.append(m_scores)
```

> **Note:** 'fairseq_args' always has preference over the 'autonmt' parameters in case of a collision. This is because
> in order to provide framework compatibility using the same set parameters, we had to define a translation table of
> parameters between tookits (i.e. "max_epochs" (autonmt) => "--max-epoch" (fairseq). So if a user sets "max_epochs=10" 
> (autonmt) in the fit, and "--max-epoch 15" (fairseq) in the 'fairseq_args', we will consider the later.


### Reproducibility

By default, AutoNMT tries to operate directly through the python apis. However, for reproducibility purposes you can
force AutoNMT to use the command line version of thoses libraries with the flag `use_cmd=True` (available in 
the DatasetBuilder and Trainers).

```python
builder = DatasetBuilder(..., use_cmd=True)
model = AutonmtTranslator(..., use_cmd=True)
```

By enabling this flag, AutoNMT will use the command line tools as a typical user. 

```bash
...
- Command used: sed -i 's/<<unk>>/<unk>/' /home/salva/preprocessing/multi30k/de-en/original/models/fairseq/runs/model_word_8000/eval/multi30k_de-en_original/beams/beam1/ref.tok
- Command used: spm_decode --model=/home/salva/preprocessing/multi30k/de-en/original/vocabs/spm/word/8000/spm_de-en.model --input_format=piece < /home/salva/preprocessing/multi30k/de-en/original/models/fairseq/runs/model_word_8000/eval/multi30k_de-en_original/beams/beam1/src.tok > /home/salva/preprocessing/multi30k/de-en/original/models/fairseq/runs/model_word_8000/eval/multi30k_de-en_original/beams/beam1/src.txt
- Command used: spm_decode --model=/home/salva/preprocessing/multi30k/de-en/original/vocabs/spm/word/8000/spm_de-en.model --input_format=piece < /home/salva/preprocessing/multi30k/de-en/original/models/fairseq/runs/model_word_8000/eval/multi30k_de-en_original/beams/beam1/ref.tok > /home/salva/preprocessing/multi30k/de-en/original/models/fairseq/runs/model_word_8000/eval/multi30k_de-en_original/beams/beam1/ref.txt
- Command used: spm_decode --model=/home/salva/preprocessing/multi30k/de-en/original/vocabs/spm/word/8000/spm_de-en.model --input_format=piece < /home/salva/preprocessing/multi30k/de-en/original/models/fairseq/runs/model_word_8000/eval/multi30k_de-en_original/beams/beam1/hyp.tok > /home/salva/preprocessing/multi30k/de-en/original/models/fairseq/runs/model_word_8000/eval/multi30k_de-en_original/beams/beam1/hyp.txt
- Command used: sacrebleu /home/salva/preprocessing/multi30k/de-en/original/models/fairseq/runs/model_word_8000/eval/multi30k_de-en_original/beams/beam1/ref.txt -i /home/salva/preprocessing/multi30k/de-en/original/models/fairseq/runs/model_word_8000/eval/multi30k_de-en_original/beams/beam1/hyp.txt -m bleu chrf ter  -w 5 > /home/salva/preprocessing/multi30k/de-en/original/models/fairseq/runs/model_word_8000/eval/multi30k_de-en_original/beams/beam1/scores/sacrebleu_scores.json
...
```
By default, AutoNMT will try to use the programs available from the `/bin/bash` (.bashrc), but you can also specify a conda environment if you want, with the flag `conda_env_name="myenv"`

```python
model = AutonmtTranslator(model=Transformer(...), ..., conda_env_name="myenv")
```

### Plots & Stats

AutoNMT will automatically generate plots for the split sizes, the sentence length distributions, 
token frequencies, the evaluated models, etc. All these plots can be found along with either a .json or a .csv 
containing its data, summary and statistics

![](docs/images/multi30k/vocab_distr_top100__multi30k_original_de-en__word_16000.png)
![](docs/images/multi30k/sent_distr_test_de__multi30k_original_de-en__word_16000.png)
![](docs/images/multi30k/split_size_tok__multi30k_original_de-en__word_16000.png)


### Layout example

This is an example of the typical layout that the DatasetBuilder generates: (complete tree [here](docs/data/tree.txt)

```text
multi30k/
.
├── original
│   └── de-en
│       ├── data
│       │   ├── encoded
│       │   │   ├── char
│       │   │   │   └── 16000
│       │   │   │       ├── test.de
│       │   │   │       ├── test.en
│       │   │   │       ├── train.de
│       │   │   │       ├── train.en
│       │   │   │       ├── val.de
│       │   │   │       └── val.en
│       │   │   ├── unigram
│       │   │   │   └── 16000
│       │   │   │       ├── test.de
│       │   │   │       ├── test.en
│       │   │   │       ├── train.de
│       │   │   │       ├── train.en
│       │   │   │       ├── val.de
│       │   │   │       └── val.en
│       │   │   └── word
│       │   │       └── 16000
│       │   │           ├── test.de
│       │   │           ├── test.en
│       │   │           ├── train.de
│       │   │           ├── train.en
│       │   │           ├── val.de
│       │   │           └── val.en
│       │   ├── pretokenized
│       │   │   ├── test.de
│       │   │   ├── test.en
│       │   │   ├── train.de
│       │   │   ├── train.en
│       │   │   ├── val.de
│       │   │   └── val.en
│       │   ├── raw
│       │   │   ├── data.de
│       │   │   └── data.en
│       │   └── splits
│       │       ├── test.de
│       │       ├── test.en
│       │       ├── train.de
│       │       ├── train.en
│       │       ├── val.de
│       │       └── val.en
│       ├── models
...
```
