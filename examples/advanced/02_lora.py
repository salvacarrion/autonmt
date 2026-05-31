"""
============================================================================
 Advanced 02 — LoRA fine-tuning
============================================================================

What you'll learn
-----------------
How to plug LoRA (Hu et al., 2021, arXiv:2106.09685) into any AutoNMT model
without modifying the framework. The recipe:

    1. Train (or load) a baseline Transformer.
    2. Freeze every original parameter.
    3. Wrap each ``nn.Linear`` with a low-rank update  W' = W + α/r · B·A,
       where A, B are tiny trainable matrices (rank r << d).
    4. Fine-tune only A/B. The base weights stay frozen; the LoRA adapter
       costs <1% of the original parameter count for r=8.
    5. Compare baseline BLEU vs. LoRA-tuned BLEU and report the trainable
       parameter count.

The LoRA wrapper here is a self-contained ~30-line `nn.Module` (no extra
dependency on `peft`/`minlora`). It's intentionally minimal so you can read
it, modify it, and confirm what's going on. For production you'd usually
reach for HuggingFace's `peft` library — drop-in equivalent ideas.

Why this matters
----------------
- Adapter-style tuning lets you ship one base model + many tiny LoRA deltas
  (per language, domain, customer) instead of N full-size fine-tunes.
- The frozen base means catastrophic forgetting on the pretraining
  distribution is structurally impossible — the original W is untouched.
  (Compare with tutorial 03, which addresses CF for full fine-tuning.)

Run
---
    pip install -e '.[hf]'
    python examples/advanced/02_lora.py
"""
import datetime
import math
import os

import torch
from torch import nn
from tokenizers.normalizers import NFKC, Strip

from autonmt.backends import AutonmtTranslator
from autonmt.backends._base.config import FitConfig, PredictConfig
from autonmt.core.nn.models import Transformer
from autonmt.datasets import DatasetBuilder
from autonmt.datasets.hf_loader import download_hf_dataset
from autonmt.datasets.preprocessing import normalize_lines, preprocess_lines, preprocess_pairs
from autonmt.reporting.report import Report

BASE_PATH = "datasets/adv_02_lora"
DATASET = "multi30k"
LANG_PAIR = "de-en"

# LoRA hyperparameters. `r` controls capacity (and trainable params); `alpha`
# scales the update at inference (the canonical LoRA paper uses alpha/r as the
# multiplier). `r=8, alpha=16` is a common starting point.
LORA_RANK = 8
LORA_ALPHA = 16


# ---------------------------------------------------------------------------
# (1) The LoRA layer — a frozen Linear wrapped by a trainable low-rank delta
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Wraps an existing ``nn.Linear`` so that

        y = (W + (alpha/r) · B·A) · x + b

    where W and b stay frozen and only A ∈ R^{r×in}, B ∈ R^{out×r} are
    trained. A is init from Kaiming-uniform (matches the paper) and B is
    zero-init so the very first forward pass equals the base model.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        in_features, out_features = base.in_features, base.out_features
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        self.scaling = alpha / rank

    def forward(self, x):
        # x: (..., in_features). F.linear is identical to base(x) when the
        # delta is zero, so the initial loss matches the baseline exactly.
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling


def inject_lora(module: nn.Module, rank: int, alpha: float, skip: tuple = ()) -> int:
    """Recursively replace ``nn.Linear`` children with ``LoRALinear``.

    ``skip`` is a tuple of attribute names to leave alone. We skip
    ``output_layer`` by default: it's tied to the target embedding in some
    Transformer configs, and the LM head is rarely what you want to LoRA-ize
    for translation.
    """
    n_replaced = 0
    for name, child in module.named_children():
        if name in skip:
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            n_replaced += 1
        else:
            n_replaced += inject_lora(child, rank=rank, alpha=alpha, skip=skip)
    return n_replaced


def count_trainable(model: nn.Module) -> tuple:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ---------------------------------------------------------------------------
# (2) Standard AutoNMT preprocessing
# ---------------------------------------------------------------------------
def normalize(lines):
    return normalize_lines(lines, seq=[NFKC(), Strip()])


def preprocess_train(data, ds):
    return preprocess_pairs(data["src"]["lines"], data["tgt"]["lines"], normalize_fn=normalize)


def preprocess_predict(data, ds):
    return preprocess_lines(data["lines"], normalize_fn=normalize)


def main():
    download_hf_dataset(
        hf_id="bentrevett/multi30k", base_path=BASE_PATH,
        dataset_name=DATASET, lang_pair=LANG_PAIR,
        src_field="de", tgt_field="en",
    )

    builder = DatasetBuilder(
        base_path=BASE_PATH,
        datasets=[{
            "name": DATASET,
            "languages": [LANG_PAIR],
            "sizes": [("original", None)],
        }],
        encoding=[{"subword_models": ["bpe"], "vocab_sizes": [4000]}],
        preprocess_raw_fn=preprocess_train,
        preprocess_splits_fn=preprocess_train,
        merge_vocabs=False,
    ).build(force_overwrite=False)

    train_ds = builder.get_train_ds()[0]
    test_datasets = builder.get_test_ds()
    src_vocab, tgt_vocab = train_ds.build_vocabs(max_tokens=150)

    pred_cfg = PredictConfig(
        metrics={"bleu", "chrf"}, beams=[5],
        load_checkpoint="best",
        preprocess_fn=preprocess_predict,
        eval_mode="compatible",
    )

    # -----------------------------------------------------------------------
    # (3) Baseline — full fine-tuning, all params trainable
    # -----------------------------------------------------------------------
    baseline_model = Transformer.from_vocabs(src_vocab, tgt_vocab)
    n_train, n_total = count_trainable(baseline_model)
    print(f"\n[baseline] trainable params: {n_train:,} / {n_total:,} ({100*n_train/n_total:.1f}%)")

    baseline = AutonmtTranslator.from_dataset(
        train_ds, model=baseline_model,
        src_vocab=src_vocab, tgt_vocab=tgt_vocab,
        run_prefix="baseline",
    )
    baseline.fit(
        train_ds,
        config=FitConfig(max_epochs=2, batch_size=128, learning_rate=1e-3, seed=42),
    )
    baseline_scores = baseline.predict(test_datasets, config=pred_cfg)

    # -----------------------------------------------------------------------
    # (4) LoRA fine-tuning — base frozen, only A/B trainable
    # -----------------------------------------------------------------------
    # We load the baseline checkpoint into a *fresh* model so the LoRA run
    # starts from the same place the baseline finished. (For real adapter
    # workflows you'd start from a pretrained checkpoint instead — same code.)
    lora_model = Transformer.from_vocabs(src_vocab, tgt_vocab)
    lora_model.load_state_dict(baseline_model.state_dict())

    # Freeze everything first, then inject_lora unfreezes only A/B.
    for p in lora_model.parameters():
        p.requires_grad = False
    n_wrapped = inject_lora(lora_model, rank=LORA_RANK, alpha=LORA_ALPHA,
                            skip=("output_layer",))
    n_train, n_total = count_trainable(lora_model)
    print(f"\n[lora] r={LORA_RANK}, wrapped {n_wrapped} Linear layers")
    print(f"[lora] trainable params: {n_train:,} / {n_total:,} ({100*n_train/n_total:.2f}%)")

    # Bigger LR is the LoRA convention — A/B start from a small subspace, so
    # they need to move faster than a full fine-tune to catch up.
    lora_trainer = AutonmtTranslator.from_dataset(
        train_ds, model=lora_model,
        src_vocab=src_vocab, tgt_vocab=tgt_vocab,
        run_prefix=f"lora-r{LORA_RANK}",
    )
    lora_trainer.fit(
        train_ds,
        config=FitConfig(max_epochs=2, batch_size=128, learning_rate=5e-3, seed=42),
    )
    lora_scores = lora_trainer.predict(test_datasets, config=pred_cfg)

    # -----------------------------------------------------------------------
    # (5) Report — baseline vs LoRA side by side
    # -----------------------------------------------------------------------
    out = f".outputs/adv_02_lora/{datetime.datetime.now():%Y%m%d_%H%M%S}"
    report = Report.from_runs([baseline_scores, lora_scores], output_path=out).save()
    print(f"\nReport saved to: {os.path.abspath(out)}\n")
    print(report)

    # -----------------------------------------------------------------------
    # (6) Optional: save just the LoRA delta
    # -----------------------------------------------------------------------
    # Adapter shipping pattern: persist only the A/B matrices (a few hundred
    # KB) rather than the full state_dict. To reload, build a fresh
    # `Transformer`, run `inject_lora(...)` with the same rank, then
    # `load_state_dict(..., strict=False)`.
    lora_state = {k: v for k, v in lora_model.state_dict().items()
                  if k.endswith(".A") or k.endswith(".B")}
    lora_ckpt = os.path.join(out, f"lora_delta_r{LORA_RANK}.pt")
    torch.save(lora_state, lora_ckpt)
    print(f"\n[lora] adapter saved ({len(lora_state)} tensors): {lora_ckpt}")


if __name__ == "__main__":
    main()
