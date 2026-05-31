# Validation & checkpoints

Training without validation is guesswork: you need to know which epoch generalized best and
when to stop. AutoNMT attaches the right PyTorch Lightning callbacks based on your
[`FitConfig`](training.md), so you only set a few fields.

## The fields

| Field | Default | Meaning |
| --- | --- | --- |
| `monitor` | `"val_loss"` | Metric the checkpoint / early-stopping callbacks watch |
| `patience` | `None` | Early-stopping patience (epochs without improvement); `None` disables |
| `save_best` | `True` | Save the best checkpoint by `monitor` (→ `…__best.pt`) |
| `save_last` | `False` | Also save the final checkpoint (→ `…__last.pt`) |
| `print_samples` | `0` | Log this many decoded validation samples per epoch |

The mode (minimize vs maximize) is inferred from the monitor name — anything containing
`loss` is minimized, otherwise maximized.

## The callbacks you get

- **`ModelCheckpoint`** — saves `__best` (when `save_best`) and/or `__last` (when
  `save_last`). Files use a `.pt` extension and embed the epoch and the monitored value in
  the name. An existing checkpoint is renamed to `.pt.bak` rather than overwritten, so a
  forced retrain never silently destroys the previous best.
- **`EarlyStopping`** — added when `patience` is set; stops training once `monitor` hasn't
  improved for `patience` epochs.
- **`TQDMProgressBar`** — a `\r`-based progress bar that behaves in notebooks and IDE
  consoles where the default Rich bar buffers oddly.

!!! info "What is early stopping?"
    More epochs eventually *overfit*: training loss keeps dropping while validation quality
    plateaus or worsens. **Early stopping** watches the validation metric and halts once it
    stops improving for `patience` epochs — saving compute and keeping the best generalizing
    model. Combined with `save_best`, you always evaluate the checkpoint that was best on
    validation, not the last (possibly overfit) one.

## Checkpoints land in the run

Checkpoints are written to the run's `checkpoints/` folder and referred to later by the
aliases `"best"` / `"last"` when you decode:

```python
trainer.predict(test_ds, config=PredictConfig(load_checkpoint="best"))   # the default
```

That alias indirection is why you rarely type checkpoint paths — `predict` resolves `best` /
`last` against the run that `fit` populated. The full path scheme is in
[Understanding the output](../../get-started/understanding-the-output.md).

!!! tip "Watch a translation metric instead of loss"
    `monitor="val_loss"` is the cheap default, but low loss isn't always high BLEU. If you
    log a decoded validation metric (via `print_samples` / a custom validation hook), you can
    point `monitor` at it and checkpoint on translation quality directly — at the cost of
    decoding during validation.

---

Next, the knob with the biggest effect on training *throughput*:
**[Bucketing & batching](bucketing.md)**.
