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
| `precision` | `"fp32"` | Mixed-precision regime: `"fp32"`, `"fp16"`, `"bf16"` (see below) |

`"auto"` everywhere means the same script runs on a GPU box, an Apple-silicon laptop, or
CPU-only CI unchanged. Pin them only when you need to (a specific GPU, multi-GPU DDP).

## Mixed precision

`precision` is a single knob that works the same way on **every** backend — set it once and
AutoNMT renders it in the right dialect:

| `precision` | Native (Lightning) | HuggingFace | Fairseq |
| --- | --- | --- | --- |
| `"fp32"` *(default)* | `"32-true"` | — | — |
| `"fp16"` | `"16-mixed"` | `fp16=True` | `--fp16` |
| `"bf16"` | `"bf16-mixed"` | `bf16=True` | `--bf16` |

```python
trainer.fit(train_ds, config=FitConfig(max_epochs=20, precision="bf16"))
```

Half precision usually gives a **1.5–3× speedup on tensor-core GPUs** and roughly halves
activation memory — the bigger win for grid sweeps is the throughput, not just fitting a model
in VRAM. The labels select a *mixed*-precision regime (fp32 master weights, half-precision
compute), which is what you almost always want.

!!! warning "Two things to keep in mind"
    - **`"bf16"` needs Ampere-class hardware (or newer) / TPUs.** On older GPUs (Volta, Turing)
      use `"fp16"`. bf16 is otherwise preferred — it has fp32's dynamic range, so no loss-scaling
      and far fewer NaN surprises in attention/softmax.
    - **The same label is _not_ bit-identical across toolkits.** Loss-scaling and kernel choices
      differ, so `precision` controls the *dtype regime*, not exact numerics — don't assume a
      cross-backend comparison is numerically identical just because the label matches.

Exotic modes stay in each backend's escape hatch: the native backend passes **Lightning-native
strings** straight through (`precision="16-true"`, `"64"`, …), and fairseq variants like
`--memory-efficient-fp16` go via `fairseq_args` (leave `precision` at its default then).

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
