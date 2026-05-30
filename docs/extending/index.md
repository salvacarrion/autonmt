# Extending AutoNMT

AutoNMT's [core is deliberately small](../introduction/philosophy.md#extensible): instead of
growing a built-in for every use case, each layer exposes a **subclass point** or a
**callable hook**. This page is the map of those extension points — the *mechanism* for each,
with a minimal example. New research components plug in; you never fork the core.

| You want to add… | Extend | Section |
| --- | --- | --- |
| a new architecture | `LitSeq2Seq` | [Custom model](#a-custom-model) |
| a new decoding rule | `BaseSearch` / `BaseStepSearch` | [Custom decoder](#a-custom-decoder) |
| a new metric | `MetricBackend` | [Custom metric](#a-custom-metric) |
| a new tokenization scheme | `BaseVocabulary` | [Custom vocabulary](#a-custom-vocabulary) |
| custom data cleaning | callable hooks | [Hooks](#hooks) |
| a whole new toolkit | `BaseTranslator` | [Custom backend](#a-custom-backend) |

## A custom model { #a-custom-model }

Subclass [`LitSeq2Seq`](../toolkit/models.md) and implement the three forward methods,
returning logits shaped `(batch, length, vocab)`. The `states` you return from
`forward_encoder` is threaded through `forward_decoder` by every [decoder](../toolkit/decoding.md),
so any architecture works with greedy, beam, and sampling search unchanged.

```python
import torch.nn as nn
from autonmt.core.nn.seq2seq import LitSeq2Seq

class TiedTransformer(LitSeq2Seq):
    def __init__(self, src_vocab_size, tgt_vocab_size, padding_idx, d_model=256, **kw):
        super().__init__(src_vocab_size, tgt_vocab_size, padding_idx,
                         architecture="tied-transformer", **kw)
        # ... build encoder/decoder, embeddings, output projection ...

    def forward_encoder(self, x, x_len, **kw):
        ...
        return None, states                       # states: whatever the decoder needs

    def forward_decoder(self, y, y_len, states, **kw):
        ...
        return logits, states                     # logits: (B, L, V)

    def forward_enc_dec(self, x, x_len, y, y_len, **kw):
        _, states = self.forward_encoder(x, x_len)
        logits, _ = self.forward_decoder(y, y_len, states)
        return logits
```

```python
trainer = AutonmtTranslator.from_dataset(
    train_ds, model=TiedTransformer.from_vocabs(src_vocab, tgt_vocab),
    src_vocab=src_vocab, tgt_vocab=tgt_vocab, run_prefix="tied")
```

!!! tip "Reuse the layer library"
    Before writing layers from scratch, check `autonmt.core.nn.layers` — `RMSNorm`, `SwiGLU`,
    `RotaryPositionalEmbedding`, and the KV-cache-aware `IncrementalTransformerDecoder` are
    ready to drop in. Set `supports_incremental_decoding = True` on your model if your decoder
    supports KV-cached single-token steps, to get fast beam search.

## A custom decoder { #a-custom-decoder }

For a step-wise rule (anything that's "pick the next token differently"), subclass
`BaseStepSearch` and implement one method — the loop, batching, EOS handling, and length cap
come from the base:

```python
import torch
from autonmt.core.decoding.base_step_search import BaseStepSearch

class TypicalSampling(BaseStepSearch):
    def __init__(self, mass=0.9):
        self.mass = mass
    def pick_next_token(self, logits):    # logits: (B, V)
        # ... your selection rule ...
        return chosen_ids                 # (B,)
```

For a strategy that tracks **multiple hypotheses** (like beam search), subclass `BaseSearch`
and implement `decode(...)` directly. Either way, pass an instance to `predict`:

```python
trainer.predict(test, config=PredictConfig(beams=[1], decoder=TypicalSampling(mass=0.9)))
```

See [Decoding strategies](../toolkit/decoding.md) for the built-ins your subclass sits
alongside.

## A custom metric { #a-custom-metric }

Metrics are entries in the [`MetricBackend`](../evaluation/metrics.md) registry. A backend
bundles a `compute_fn` (write a score artifact) and a `parse_fn` (read it back into
`{metric: {field: value}}`). Register one and it's available to every translator:

```python
from autonmt.evaluation.metrics import MetricBackend, METRIC_BACKENDS

def compute_mymetric(*, ref_file, hyp_file, output_file, metrics, **_):
    score = my_scorer(ref_file, hyp_file)
    save_json([{"name": "mymetric", "score": score}], output_file)

def parse_mymetric(text_lines):
    import json
    d = json.loads("".join(text_lines))[0]
    return {"mymetric": {"score": float(d["score"])}}

METRIC_BACKENDS["mymetric"] = MetricBackend(
    name="mymetric", metrics=frozenset({"mymetric"}),
    compute_fn=compute_mymetric, parse_fn=parse_mymetric,
    needs_src=False,    # set True if compute_fn consumes src_file (like COMET)
)
```

```python
trainer.predict(test, config=PredictConfig(metrics={"bleu", "mymetric"}))
```

The flattened report key follows the standard schema, `mymetric_mymetric_score`.

## A custom vocabulary { #a-custom-vocabulary }

Subclass [`BaseVocabulary`](../data/vocabularies.md) (or `Vocabulary`) to implement a
different tokenization or lookup scheme. As long as it exposes the four special-token ids and
`encode` / `decode`, it drops straight into a model and a translator — `build_vocabs` is a
convenience, not a requirement:

```python
from autonmt.vocabularies.base_vocab import BaseVocabulary

class MyVocab(BaseVocabulary):
    def encode(self, text, add_special_tokens=True): ...
    def decode(self, idxs, remove_special_tokens=True): ...

src_vocab = tgt_vocab = MyVocab()
model = Transformer.from_vocabs(src_vocab, tgt_vocab)
```

## Hooks: custom data cleaning { #hooks }

Preprocessing doesn't need subclassing at all — you pass **callable hooks**. There's no new
type to learn; you write a function and AutoNMT calls it at the right stage:

```python
def clean_pairs(data, ds):                    # preprocess_raw_fn / preprocess_splits_fn
    return preprocess_pairs(data["src"]["lines"], data["tgt"]["lines"], normalize_fn=normalize)

builder = DatasetBuilder(..., preprocess_raw_fn=clean_pairs, preprocess_splits_fn=clean_pairs)
```

The predict-time `preprocess_fn` (passed to `predict`) is the matching hook for the test
source. Full details and signatures are in [Preprocessing &
encoding](../data/preprocessing-and-encoding.md#hooks).

## A custom backend { #a-custom-backend }

To run a *new* toolkit through AutoNMT, subclass
[`BaseTranslator`](../architecture/toolkit-abstraction.md) and implement its hooks. You inherit
config persistence, eval filtering, scoring, and the report schema — you only write the
toolkit-specific parts:

```python
from autonmt.backends._base.translation_engine import BaseTranslator

class MyToolkitTranslator(BaseTranslator):
    ENGINE = "mytoolkit"                 # names the models/<toolkit>/ run folder

    def _get_lang_pair(self):
        return self.src_lang, self.tgt_lang

    def _train(self, train_ds, checkpoints_dir, logs_path, force_overwrite, **kw):
        ...                              # run your toolkit's training

    def _translate(self, *, eval_ds, output_path, beam_width, ...):
        ...                              # write src.txt / ref.txt / hyp.txt (direct mode)
        # — or set self._spm = SPMTranslatePipeline(...) in __init__ and only write hyp.tok
```

Decide your [translate mode](../architecture/toolkit-abstraction.md#two-translate-modes-spm-pipeline-vs-direct):
set `self._spm` for the SentencePiece round-trip, or leave it `None` and write the artifacts
yourself. Optionally override `_get_run_metadata` to enrich the report with your model/vocab
info.

---

That's the full surface area. For the exact signatures of every type referenced here, see the
**[API reference](../reference/index.md)**.
