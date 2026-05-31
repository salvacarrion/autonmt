# Training a model

`fit` trains a model on one dataset variant. You configure it with a `FitConfig` (or loose
kwargs) and AutoNMT handles the rest: building train/validation DataLoaders, configuring the
optimizer/scheduler/loss, attaching callbacks and loggers, running the PyTorch Lightning
trainer, and saving checkpoints.

```python
from autonmt.backends._base.config import FitConfig

trainer.fit(train_ds, config=FitConfig(
    max_epochs=20,
    batch_size=128,
    learning_rate=5e-4,
    optimizer="adam",
    scheduler="noam", warmup_steps=4000,
    patience=5,            # early stopping
    seed=42,
))
```

!!! info "What happens in a training step? (teacher forcing)"
    A seq2seq model learns to predict the next target token given the source and the target
    tokens *so far*. During training we don't feed the model its own (initially wrong)
    predictions — we feed the **ground-truth** previous tokens and ask it to predict the
    next one at every position in parallel. This is **teacher forcing**: it makes training
    stable and fully parallel. The loss is cross-entropy between the predicted distribution
    and the true next token, averaged over all positions (ignoring `<pad>`). At *inference*
    time there's no ground truth, so the model consumes its own outputs — which is what
    [decoding](../translation/decoding.md) handles.

## `FitConfig` is one object, documented across this section

Everything tunable about training lives on `FitConfig`. Its fields group naturally, and so
do these pages:

| Group | Fields | Page |
| --- | --- | --- |
| Optimization & LR schedule | `optimizer`, `learning_rate`, `scheduler`, … | this page |
| Validation & checkpoints | `monitor`, `patience`, `save_best`, … | [Validation & checkpoints](validation-checkpoints.md) |
| Batching & bucketing | `batch_size`, `max_tokens`, `use_bucketing` | [Bucketing & batching](bucketing.md) |
| Hardware, loggers, extras | `accelerator`, `devices`, `wandb_params`, … | [Advanced training control](advanced.md) |

The complete field list with defaults is in the [API reference](../../reference/backends.md).

## Optimization

| Field | Default | Meaning |
| --- | --- | --- |
| `max_epochs` | `1` | Number of passes over the training set |
| `batch_size` | `128` | Sentences per batch (see [batching](bucketing.md)) |
| `optimizer` | `"adam"` | One of `adam`, `adamw`, `sgd`, `adadelta`, `adagrad`, `nadam`, `radam`, `rmsprop`, … |
| `learning_rate` | `0.001` | Initial learning rate |
| `weight_decay` | `0` | L2 regularization |
| `criterion` | `"cross_entropy"` | Training loss |
| `gradient_clip_val` | `0.0` | Clip gradients to this norm (`0` disables) |
| `accumulate_grad_batches` | `1` | Simulate a larger batch by accumulating gradients |

`optimizer` accepts a **string** (resolved against torch's optimizer table) — handy for
sweeps — and the criterion defaults to cross-entropy with the target `<pad>` id ignored, so
padding never contributes to the loss.

## Learning-rate schedule

| Field | Default | Meaning |
| --- | --- | --- |
| `scheduler` | `None` | `"noam"`, `"inverse_sqrt"`, a callable `(optimizer) -> scheduler`, or a torch scheduler instance |
| `warmup_steps` | `None` | Warmup steps for step-based schedules |

!!! info "Why warmup + inverse-sqrt decay? (the Noam schedule)"
    Transformers train badly with a constant learning rate: too high early on and the freshly
    initialized attention blows up; too low and training crawls. The **Noam** schedule (from
    *Attention Is All You Need*) ramps the LR **up** linearly for `warmup_steps`, then decays
    it proportionally to the inverse square root of the step:

    $$
    \text{lr}(t) = d_{\text{model}}^{-0.5} \cdot \min\!\left(t^{-0.5},\; t \cdot \text{warmup}^{-1.5}\right)
    $$

    `"inverse_sqrt"` is the same idea without the $d_{\text{model}}$ scaling. Both are
    **step-based**, so AutoNMT updates them every optimizer step, not every epoch. A warmup
    of a few thousand steps is standard.

## Defaults and precedence

All `FitConfig` fields are optional; the defaults train a small model for one epoch with
Adam — enough to prove the loop, not to produce results. Remember the precedence rule from
[Configuration](../experiments/configuration.md): **defaults < `config=` < kwargs**, so you
can keep a base config for a sweep and override one knob per run.

---

A model trains, but you also need to know *which* checkpoint to keep and when to stop:
**[Validation & checkpoints](validation-checkpoints.md)**.
