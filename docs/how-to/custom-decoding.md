# Change the decoding strategy

Decoding is split in two: a **strategy** (`BaseStrategy` — *how* you pick the next token) is
composed into a **driver** (`BaseSearch` — *the loop*). A new sampling rule is therefore a
tiny `BaseStrategy` you drop into either driver, and it works the same for encoder-decoder
translation **and** decoder-only LMs. (The built-ins are in
[Decoding strategies](../guide/translation/decoding.md).)

## A token-selection rule

For anything that's "pick the next token differently," subclass `BaseStrategy` and implement
**one** method — the loop, batching, EOS handling, and length cap belong to the driver, not
to you:

```python
import torch
from autonmt.core.decoding import BaseStrategy

class TypicalSampling(BaseStrategy):
    def __init__(self, mass=0.9):
        self.mass = mass

    def pick_next_token(self, logits):    # logits: (B, V) at the current step
        # ... your selection rule ...
        return chosen_ids                 # (B,) chosen token id per sequence
```

## A multi-hypothesis strategy

For something that tracks **multiple hypotheses** (like beam search), the per-step picture
doesn't apply — subclass the driver `BaseSearch` and implement `decode(...)` directly. It
returns `(token_id_lists, optional_scores)`.

## Use it

Your strategy plugs into the seq2seq driver via `decoder=StepSearch(...)`:

```python
trainer.predict(test, config=PredictConfig(
    beams=[1], decoder=StepSearch(TypicalSampling(mass=0.9))))
```

…and into a decoder-only LM via the same instance — that's the whole point of the split:

```python
trainer.generate("the quick", strategy=TypicalSampling(mass=0.9))
```

`beams` still controls the output folder name; your strategy overrides *how* tokens are
chosen. If your model supports
[incremental decoding](../guide/models/building-blocks.md#the-incremental-autoregressive-decoder),
`StepSearch` gets the KV-cache speedup for free — your strategy doesn't have to know.
