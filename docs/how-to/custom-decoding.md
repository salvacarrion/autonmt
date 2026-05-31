# Change the decoding strategy

Decoders are `BaseSearch` / `BaseStepSearch` subclasses, so a new search or sampling rule is
a small class you pass to `predict`. (The built-ins it sits alongside are in
[Decoding strategies](../guide/translation/decoding.md).)

## A step-wise rule

For anything that's "pick the next token differently," subclass `BaseStepSearch` and
implement **one** method — the loop, batching, EOS handling, and length cap come from the
base:

```python
import torch
from autonmt.core.decoding.base_step_search import BaseStepSearch

class TypicalSampling(BaseStepSearch):
    def __init__(self, mass=0.9):
        self.mass = mass

    def pick_next_token(self, logits):    # logits: (B, V) at the current step
        # ... your selection rule ...
        return chosen_ids                 # (B,) chosen token id per sequence
```

## A multi-hypothesis strategy

For a strategy that tracks **multiple hypotheses** (like beam search), subclass `BaseSearch`
and implement `decode(...)` directly — it returns `(token_id_lists, optional_scores)`.

## Use it

Either way, pass an instance via `decoder=`:

```python
trainer.predict(test, config=PredictConfig(beams=[1], decoder=TypicalSampling(mass=0.9)))
```

`beams` still controls the output folder name; your `decoder` overrides *how* tokens are
chosen. If your model supports
[incremental decoding](../guide/models/building-blocks.md#the-incremental-autoregressive-decoder),
a `BaseStepSearch` subclass gets the KV-cache speedup for free.
