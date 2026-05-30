# Training (`fit`)

`fit` trains a [`LitSeq2Seq`](models.md) model on one dataset variant. You configure it with
a [`FitConfig`](#fitconfig) (or loose kwargs), and AutoNMT handles the rest: building
train/validation DataLoaders, configuring the optimizer/scheduler/loss, attaching callbacks
and loggers, running the PyTorch Lightning trainer, and saving checkpoints.

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
    predictions — we feed the **ground-truth** previous tokens and ask it to predict the next
    one at every position in parallel. This is **teacher forcing**: it makes training stable
    and fully parallel. The loss is cross-entropy between the predicted distribution and the
    true next token, averaged over all positions (ignoring `<pad>`). At *inference* time
    there's no ground truth, so the model consumes its own outputs — which is what
    [decoding](decoding.md) handles.

## `FitConfig` { #fitconfig }

Every field, grouped by what it controls. All are optional; the defaults train a small model
for one epoch with Adam.

### Optimization

| Field | Default | Meaning |
| --- | --- | --- |
| `max_epochs` | `1` | Number of passes over the training set |
| `batch_size` | `128` | Sentences per batch (ignored if `max_tokens` is set with bucketing) |
| `max_tokens` | `None` | Token budget per batch — an alternative to `batch_size` (see [bucketing](#bucketing)) |
| `optimizer` | `"adam"` | One of `adam`, `adamw`, `sgd`, `adadelta`, `adagrad`, `nadam`, `radam`, `rmsprop`, … |
| `learning_rate` | `0.001` | Initial learning rate |
| `weight_decay` | `0` | L2 regularization |
| `criterion` | `"cross_entropy"` | Training loss |
| `gradient_clip_val` | `0.0` | Clip gradients to this norm (`0` disables) |
| `accumulate_grad_batches` | `1` | Simulate a larger batch by accumulating gradients |

### Learning-rate schedule

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
    **step-based**, so AutoNMT updates them every optimizer step, not every epoch. A warmup of
    a few thousand steps is standard.

### Checkpointing & monitoring

| Field | Default | Meaning |
| --- | --- | --- |
| `monitor` | `"val_loss"` | Metric the checkpoint/early-stopping callbacks watch |
| `patience` | `None` | Early-stopping patience (epochs without improvement); `None` disables |
| `save_best` | `True` | Save the best checkpoint by `monitor` (→ `..__best.pt`) |
| `save_last` | `False` | Also save the final checkpoint (→ `..__last.pt`) |
| `print_samples` | `0` | Log this many decoded validation samples per epoch |

The mode (minimize/maximize) is inferred from the monitor name — anything containing `loss`
is minimized, otherwise maximized. Checkpoints land in the run's `checkpoints/` folder and
are referred to later by the aliases `"best"` / `"last"` in [`predict`](predict.md).

### Hardware & reproducibility

| Field | Default | Meaning |
| --- | --- | --- |
| `accelerator` | `"auto"` | `"auto"` (CUDA → MPS → CPU), `"gpu"`, `"cpu"`, `"mps"` |
| `devices` | `"auto"` | Device count / ids |
| `strategy` | `"auto"` | Lightning strategy (e.g. `"ddp"` for multi-GPU) |
| `num_workers` | `0` | DataLoader workers |
| `seed` | `None` | Seeds Python/NumPy/Torch/Lightning via `manual_seed` |
| `use_bucketing` | `False` | Length-bucket batches (see below) |
| `force_overwrite` | `False` | Back up and retrain over an existing checkpoint |

## Optimizers and the loss

`optimizer` accepts a string (resolved against torch's optimizer table) — handy for sweeps —
or you can configure anything torch supports. The criterion defaults to cross-entropy with
the target `<pad>` id ignored, so padding never contributes to the loss.

## Callbacks

AutoNMT attaches Lightning callbacks for you based on the config:

- **`ModelCheckpoint`** — saves `__best` (when `save_best`) and/or `__last` (when
  `save_last`). Files use a `.pt` extension and embed the epoch + monitored value in the
  name. Existing checkpoints are renamed to `.pt.bak` rather than overwritten.
- **`EarlyStopping`** — added when `patience` is set; stops when `monitor` hasn't improved for
  `patience` epochs.
- **`TQDMProgressBar`** — a `\r`-based bar that works in notebooks and IDE consoles where the
  default Rich bar buffers.

!!! info "What is early stopping?"
    More epochs eventually *overfit*: training loss keeps dropping while validation quality
    plateaus or worsens. **Early stopping** watches the validation metric and halts once it
    stops improving for `patience` epochs — saving compute and keeping the best generalizing
    model. Combined with `save_best`, you always evaluate the checkpoint that was best on
    validation, not the last (possibly overfit) one.

## Loggers

- **TensorBoard** is enabled by default; logs go under the run's `logs/` folder
  (`tensorboard --logdir <run>/logs`).
- **Weights & Biases** is opt-in. Install `[wandb]` and pass `wandb_params`:

```python
trainer.fit(train_ds, config=FitConfig(max_epochs=20),
            wandb_params={"project": "my-mt-sweep"})
```

`wandb_params` rides along as a toolkit-specific extra (it isn't a `FitConfig` field) and is
forwarded to the W&B logger.

## Length bucketing { #bucketing }

By default, batches are random and padded to the longest sentence in the batch. With
`use_bucketing=True`, AutoNMT groups sentences of **similar length** into each batch, so far
less padding is wasted — and you can switch from a fixed `batch_size` to a `max_tokens`
budget (variable-size batches that pack more short sentences and fewer long ones).

```python
trainer.fit(train_ds, config=FitConfig(use_bucketing=True, max_tokens=8000))
```

The mechanics (the `BucketSampler`, why it helps, and the trade-offs) are covered in
[Samplers & the TranslationDataset](data-pipeline.md#bucketing). Note: packed-sequence
models (some RNNs) *require* bucketing.

## Toolkit-specific extras

Any kwarg that isn't a `FitConfig` field is forwarded to the underlying Lightning trainer or
to AutoNMT-specific machinery — e.g. `strategy="ddp"`, `wandb_params=...`. This is the same
"extras win, and pass through" mechanism the [contract
uses](../architecture/toolkit-abstraction.md) across all backends.

---

Trained a model? Decode and score it: **[Translating & decoding (`predict`)](predict.md)**.
