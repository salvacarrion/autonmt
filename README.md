# AutoNLP

AutoNLP is a library to build seq2seq models with almost no-code, so that you can remain focused on your research.

**What do we offer?**
- Toolkit abstraction
- Automatic and replicable experimentation in grid
  - Automatic dataset preprocessing
  - Automatic training, fine-tuning, postprocessing and scoring
  - Automatic logging, reporting, plotting and statistics of the datasets and models

Despite our "almost no-code" approach, all the control remains in the user since all the intermediate steps (pretokenization, 
subword encodings, split sizes...) are:
- Saved in separated folders so that a user can inspect any part of the process (and reuse it)
- Produced using standard libraries (moses, sentencepiece, sacrebleu,...) so that you can replicate them via the command line.


## Installation

**Requirements:**

- Python +3.6

**Installation:**

```
git clone git@github.com:salvacarrion/autonlp.git
cd autonlp
pip install -e .
```


## Usage

This is a summary but if you want to see complete examples go [here](examples).

### Dataset generation

The `DatasetBuilder` is the object in charge of generating datasets from a reference one ("original"). If you don't know how to use it, 
set `interactive=True` (default) in the `DatasetBuilder` constructor, and it will guide you step-by-step so that you can create reference datasets.

```python
from autonmt import DatasetBuilder

# Create datasets for training (2*1*2*3*2 = 24 datasets)
tr_datasets = DatasetBuilder(
    base_path="/home/salva/datasets",
    datasets=[
        {"name": "scielo/biological", "languages": ["es-en"], "sizes": [("original", None), ("100k", 100000)]},
        {"name": "scielo/health", "languages": ["es-en"], "sizes": [("original", None), ("100k", 100000)]},
    ],
    subword_models=["word", "unigram", "char"],
    vocab_sizes=[8000, 16000],
).build(make_plots=True)

# Create datasets for testing
ts_datasets = tr_datasets
```


### Train & Score

The `Translator` object abstracts the seq2seq pipeline so that you can train and score your custom models effortless. Similarly, 
you can use other engines such as `fairseq` or `opennmt`.

```python
import autonmt as al

# Train & Score a model for each dataset
for train_ds in tr_datasets:
    model = al.Translator()
    model.fit(train_ds)
    model.predict(ts_datasets, metrics={"bleu", "chrf", "bertscore", "comet"}, beams=[1, 5])
```

### Generate a report

```python
from autonmt.tasks.translation.bundle.metrics import create_report

# Train & Score a model for each dataset
scores = []
for train_ds in tr_datasets:
  model = al.Translator()
  model.fit(train_ds)
  eval_scores = model.predict(ts_datasets, metrics={"bleu"}, beams=[5])
  scores.append(eval_scores)

# Make report
create_report(scores=scores, metric_id="beam_5__sacrebleu_bleu", output_path=".outputs")
```

### Toolkit abstraction

#### Custom models

To create your custom pytorch model, you only need inherit from `Seq2Seq` and then pass it as parameter to the `Translator` class. 
The only requirement is that the forward must return a tensor with shape `(batch, length, probabilities)`.

```python
from autonmt.tasks.translation.models import Seq2Seq


class Transformer(Seq2Seq):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # do stuff        

    def forward(self, X, Y):
        # do stuff
        return  # Tensor with shape: (Batch, Length, probabilities)

# Custom model
mymodel = Transformer()

# Train & Score a model for each dataset
for train_ds in tr_datasets:
    model = al.Translator(model=mymodel)
    model.fit(train_ds)
    model.predict(ts_datasets, metrics={"bleu"}, beams=[5])
```

**Custom trainer/evaluator**

If you need to write a custom fit or evaluate function, you can either overwrite the methods you want from the `Seq2Seq` class, or simply write
your own class, like this:

```python
import torch.nn as nn
from autonmt.tasks.translation.bundle.dataset import TranslationDataset


class CustomSeq2Seq(nn.Module):
  def __init__(self, src_vocab_size, trg_vocab_size, *args, **kwargs):
    super().__init__()
    self.src_vocab_size = src_vocab_size
    self.trg_vocab_size = trg_vocab_size

  def fit(self, ds_train: TranslationDataset, ds_val: TranslationDataset, *args, **kwargs):
    pass

  def evaluate(self, *args, **kwargs):
    pass


class CustomModel(CustomSeq2Seq):
  def __init__(self, *args, **kwargs):
    super().__init__()

  def forward(self, X, Y, *args, **kwargs):
    pass


# Custom model
model = al.Translator(model=CustomModel())
```

#### Fairseq models

When using a Fairseq model, you can use it through the fairseq command-line tools:

```text
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

# Train fairseq models
for train_ds in tr_datasets:
    model = al.FairseqTranslator(conda_env_name="fairseq")
    model.fit(train_ds, fairseq_args=fairseq_args)
    model.predict(ts_datasets, metrics={"bleu"}, beams=[1, 5])
```


### Replicability

If you use AutoNMT as command-line interface, it will gives you all the commands it is using under the hood. For example, this is a typical output when working in the command-line mode:

```bash
...
- Command used: sed -i 's/<<unk>>/<unk>/' /home/salva/datasets/multi30k/de-en/original/models/fairseq/runs/model_word_8000/eval/multi30k_de-en_original/beams/beam1/ref.tok
- Command used: spm_decode --model=/home/salva/datasets/multi30k/de-en/original/vocabs/spm/word/8000/spm_de-en.model --input_format=piece < /home/salva/datasets/multi30k/de-en/original/models/fairseq/runs/model_word_8000/eval/multi30k_de-en_original/beams/beam1/src.tok > /home/salva/datasets/multi30k/de-en/original/models/fairseq/runs/model_word_8000/eval/multi30k_de-en_original/beams/beam1/src.txt
- Command used: spm_decode --model=/home/salva/datasets/multi30k/de-en/original/vocabs/spm/word/8000/spm_de-en.model --input_format=piece < /home/salva/datasets/multi30k/de-en/original/models/fairseq/runs/model_word_8000/eval/multi30k_de-en_original/beams/beam1/ref.tok > /home/salva/datasets/multi30k/de-en/original/models/fairseq/runs/model_word_8000/eval/multi30k_de-en_original/beams/beam1/ref.txt
- Command used: spm_decode --model=/home/salva/datasets/multi30k/de-en/original/vocabs/spm/word/8000/spm_de-en.model --input_format=piece < /home/salva/datasets/multi30k/de-en/original/models/fairseq/runs/model_word_8000/eval/multi30k_de-en_original/beams/beam1/hyp.tok > /home/salva/datasets/multi30k/de-en/original/models/fairseq/runs/model_word_8000/eval/multi30k_de-en_original/beams/beam1/hyp.txt
- Command used: sacrebleu /home/salva/datasets/multi30k/de-en/original/models/fairseq/runs/model_word_8000/eval/multi30k_de-en_original/beams/beam1/ref.txt -i /home/salva/datasets/multi30k/de-en/original/models/fairseq/runs/model_word_8000/eval/multi30k_de-en_original/beams/beam1/hyp.txt -m bleu chrf ter  -w 5 > /home/salva/datasets/multi30k/de-en/original/models/fairseq/runs/model_word_8000/eval/multi30k_de-en_original/beams/beam1/scores/sacrebleu_scores.json
...
```


### Plots & Stats

AutoNLP will automatically generate plots for the split sizes, the sentence length distributions, 
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