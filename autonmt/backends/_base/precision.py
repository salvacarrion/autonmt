"""Unified mixed-precision knob shared by every training backend.

``FitConfig.precision`` accepts a small canonical vocabulary — ``"fp32"``
(default), ``"fp16"``, ``"bf16"`` — that each backend renders in its own dialect:

* ``AutonmtTranslator`` / ``AutonmtCausalLM`` (PyTorch Lightning) → ``Trainer(precision=...)``
* ``HuggingFaceTranslator`` (``Seq2SeqTrainingArguments``) → ``fp16=`` / ``bf16=``
* ``FairseqTranslator`` (CLI) → ``--fp16`` / ``--bf16``

The canonical labels select a *mixed*-precision regime (fp32 master weights,
half-precision compute), which is what you almost always want. Pure-half and
exotic variants (Lightning ``"16-true"``, fairseq ``--memory-efficient-fp16``,
…) stay in each backend's escape hatch.

Two caveats worth stating plainly:

* The same label is **not** bit-identical across toolkits — loss-scaling and
  kernel choices differ. ``precision`` selects the *dtype regime*, not exact
  numerics, so it does not by itself guarantee numerically identical runs when
  comparing backends.
* ``"bf16"`` needs Ampere-class hardware (or newer) / TPUs; on older GPUs
  (e.g. Volta, Turing) use ``"fp16"`` instead.

References
----------
Micikevicius et al. (2018). *Mixed Precision Training.*
[arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
"""

#: Canonical precision tokens accepted by ``FitConfig.precision``.
PRECISIONS = ("fp32", "fp16", "bf16")

# Canonical token -> PyTorch Lightning ``Trainer(precision=...)`` value.
_LIGHTNING = {"fp32": "32-true", "fp16": "16-mixed", "bf16": "bf16-mixed"}


def to_lightning(precision):
    """Map a canonical precision to a Lightning ``precision=`` value.

    Unknown values pass through unchanged, so power users can hand Lightning a
    native string (``"16-true"``, ``"64"``, …) directly. ``None`` disables the
    mapping (Lightning keeps its own default).
    """
    if precision is None:
        return None
    return _LIGHTNING.get(precision, precision)


def to_hf_kwargs(precision):
    """Map a canonical precision to ``Seq2SeqTrainingArguments`` kwargs.

    Returns ``{"fp16": True}`` / ``{"bf16": True}`` for the half formats, or an
    empty dict for ``"fp32"`` (or anything unrecognised) so HuggingFace keeps
    its fp32 default. Backend-native precision strings are *not* understood here
    — only the canonical tokens map onto HF.
    """
    if precision == "fp16":
        return {"fp16": True}
    if precision == "bf16":
        return {"bf16": True}
    return {}


def to_fairseq_args(precision):
    """Map a canonical precision to fairseq CLI flags (``--fp16`` / ``--bf16``).

    Returns an empty list for ``"fp32"`` (or anything unrecognised). Exotic
    fairseq variants (``--memory-efficient-fp16`` …) should be passed via
    ``fairseq_args`` with ``precision`` left at its default.
    """
    if precision == "fp16":
        return ["--fp16"]
    if precision == "bf16":
        return ["--bf16"]
    return []
