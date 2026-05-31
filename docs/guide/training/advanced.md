# Advanced training control

`fit` covers the common case with a few config fields, but it doesn't box you in. This page
collects the escape hatches: hardware placement, loggers, and the passthrough that forwards
anything AutoNMT doesn't recognize straight to the underlying PyTorch Lightning trainer.

## Hardware & scale

| Field | Default | Meaning |
| --- | --- | --- |
| `accelerator` | `"auto"` | `"auto"` (CUDA → MPS → CPU), `"gpu"`, `"cpu"`, `"mps"` |
| `devices` | `"auto"` | Device count / ids |
| `strategy` | `"auto"` | Lightning strategy (e.g. `"ddp"` for multi-GPU) |
| `num_workers` | `0` | DataLoader workers |

`"auto"` everywhere means the same script runs on a GPU box, an Apple-silicon laptop, or
CPU-only CI unchanged. Pin them only when you need to (a specific GPU, multi-GPU DDP).

## Gradient control

Two `FitConfig` fields handle the usual stability/scale knobs:

- **`gradient_clip_val`** — clip the gradient norm (helps unstable training; `0` disables).
- **`accumulate_grad_batches`** — accumulate gradients over N batches to simulate a larger
  effective batch without the memory cost.

## Loggers

- **TensorBoard** is on by default; logs go under the run's `logs/` folder
  (`tensorboard --logdir <run>/logs`).
- **Weights & Biases** is opt-in. Install the `[wandb]` extra and pass `wandb_params`:

```python
trainer.fit(train_ds, config=FitConfig(max_epochs=20),
            wandb_params={"project": "my-mt-sweep"})
```

`wandb_params` isn't a `FitConfig` field — it rides along as a toolkit-specific extra and is
forwarded to the W&B logger (see passthrough below).

## Toolkit-specific extras (passthrough)

Any keyword that isn't a `FitConfig` field is forwarded to the underlying Lightning trainer
or to AutoNMT-specific machinery — `strategy="ddp"`, `wandb_params=...`, and other Lightning
`Trainer` arguments. This is the same "extras win, and pass through" mechanism the
[translator contract](../backends/choosing.md) uses across every backend, so a backend can
expose its native knobs without AutoNMT having to mirror each one as a config field.

```python
# FitConfig fields + a raw Lightning Trainer kwarg in the same call:
trainer.fit(train_ds, config=FitConfig(max_epochs=20), gradient_clip_algorithm="value")
```

## When you need more than `fit`

If you want to own the loop entirely — custom Lightning callbacks beyond the built-ins, a
hand-built DataLoader, inspecting tensors mid-training, or splitting the pipeline into
separate stages — every component is public and documented in
[How-to → Drive the pipeline manually](../../how-to/manual-pipeline.md).

---

The model is trained. Now turn it into translations:
**[Translation → Generating translations](../translation/generating.md)**.
