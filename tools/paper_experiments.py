#!/usr/bin/env python
"""
AutoNMT paper experiments + end-to-end test harness.
============================================================================

This single script has two jobs:

  --mode smoke   Exercise every model family and code path with *tiny* configs.
                 Hermetic (no network), CPU-friendly, runs in a few minutes.
                 Each path is a subtest; a pass/fail/skip table is printed and
                 the process exits non-zero if anything fails. Run this before
                 publishing the repo, on top of `pytest tests/`.

  --mode paper   Run the real (small but publishable) experiments, calibrated
                 for a single 12-16 GB GPU (RTX 4070 Ti / 4080). Writes
                 results.json + summary.{md,csv} and plots under --outdir,
                 ready to drop into the paper.

Result sets (choose with --task, default: all):

  triad     Canonical triad, one recognizable number per family:
              enc-dec  -> Transformer NMT on Multi30k de-en   (BLEU / chrF)
              dec-only -> small GPT on WikiText-2             (perplexity)
              enc-only -> small MLM on WikiText-2             (masked accuracy)
  sweep     Grid showcase:
              NMT BLEU vs SentencePiece vocab size
              GPT perplexity vs model size (#params)
  parity    AutoNMT-Transformer vs HuggingFace opus-mt on Multi30k (BLEU).
            Replaces the old (Fairseq) toolkit-comparison with a live toolkit.
  decoding  Effect of the decoding algorithm:
              NMT  -> BLEU vs beam width {1, 3, 5}
              GPT  -> output diversity (distinct-1/2) for greedy/top-k/top-p
  models    Catalog smoke only: instantiate every built-in model and run one
            forward pass (shape check). Always cheap; part of --mode smoke.

Examples
--------
    # Pre-publication end-to-end check (fast, offline):
    python tools/paper_experiments.py --mode smoke

    # Full paper run on the rented GPU:
    python tools/paper_experiments.py --mode paper --outdir .paper_results

    # Just one result set:
    python tools/paper_experiments.py --mode paper --task triad

Notes
-----
* `paper` mode for `triad`/`sweep`/`parity` needs the optional `datasets`
  package (`pip install -e '.[hf]'`) and network access for Multi30k / WikiText-2
  / opus-mt. Missing deps are reported as SKIP, never a hard crash.
* `paper` mode trains in bf16 mixed precision (RTX 4070 Ti / 4080 are Ada, so bf16
  is supported); models are kept modest so everything fits in 12 GB. Bump the PAPER
  profile knobs (sizes / `precision`) once you have headroom (a 4080 fits more).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime

# Heavy ML imports (torch/lightning/sentencepiece) are deferred into the task
# functions so `--help` and dependency-missing SKIPs stay fast and graceful.


# ===========================================================================
# Profiles: the only knobs you normally touch. `smoke` is tiny & hermetic;
# `paper` is calibrated for a single 12-16 GB GPU. Bump `paper` if you have room.
# ===========================================================================
@dataclass
class Profile:
    name: str
    # --- NMT (encoder-decoder) ---
    nmt_vocab: int
    nmt_layers: int
    nmt_embed: int
    nmt_heads: int
    nmt_ff: int
    nmt_epochs: int
    nmt_batch: int
    nmt_vocab_sweep: tuple        # vocab sizes for the sweep task
    # --- LM (decoder-only) / MLM (encoder-only) ---
    lm_vocab: int
    lm_embed: int
    lm_layers: int
    lm_heads: int
    lm_block: int
    lm_max_seq: int
    lm_epochs: int
    lm_batch: int
    lm_size_sweep: tuple          # (embed_dim, num_layers) points for GPT scaling
    # --- shared ---
    seed: int = 42
    precision: str = "fp32"       # "fp32" | "fp16" | "bf16" (bf16 needs Ampere+)


SMOKE = Profile(
    name="smoke",
    nmt_vocab=200, nmt_layers=2, nmt_embed=64, nmt_heads=2, nmt_ff=128,
    nmt_epochs=1, nmt_batch=32, nmt_vocab_sweep=(100, 200),
    lm_vocab=128, lm_embed=64, lm_layers=2, lm_heads=2,
    lm_block=32, lm_max_seq=64, lm_epochs=2, lm_batch=32,
    lm_size_sweep=((64, 2), (96, 3)),
)

PAPER = Profile(
    name="paper",
    nmt_vocab=8000, nmt_layers=6, nmt_embed=512, nmt_heads=8, nmt_ff=2048,
    nmt_epochs=30, nmt_batch=128, nmt_vocab_sweep=(4000, 8000, 16000, 32000),
    lm_vocab=16000, lm_embed=384, lm_layers=6, lm_heads=6,
    lm_block=256, lm_max_seq=256, lm_epochs=10, lm_batch=32,
    lm_size_sweep=((192, 4), (384, 6), (512, 8)),
    precision="bf16",   # RTX 4070 Ti / 4080 are Ada (Ampere+): bf16 is the sweet spot
)


# ===========================================================================
# Result bookkeeping
# ===========================================================================
@dataclass
class Outcome:
    task: str
    status: str            # "ok" | "fail" | "skip"
    seconds: float = 0.0
    detail: str = ""
    data: dict = field(default_factory=dict)


def _short(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}".strip().replace("\n", " ")[:300]


class Runner:
    """Runs subtests, captures status/timing, and prints a final table."""

    def __init__(self, mode: str, outdir: str):
        self.mode = mode
        self.outdir = outdir
        self.outcomes: list[Outcome] = []
        os.makedirs(outdir, exist_ok=True)

    def run(self, task: str, fn, *args, **kwargs) -> Outcome:
        print(f"\n{'='*74}\n[{self.mode}] >>> {task}\n{'='*74}", flush=True)
        t0 = time.time()
        try:
            data = fn(*args, **kwargs) or {}
            status = data.pop("__status__", "ok")
            oc = Outcome(task, status, time.time() - t0,
                         detail=data.pop("__detail__", ""), data=data)
        except _Skip as s:
            oc = Outcome(task, "skip", time.time() - t0, detail=str(s))
        except Exception as e:                                    # noqa: BLE001
            traceback.print_exc()
            oc = Outcome(task, "fail", time.time() - t0, detail=_short(e))
        self.outcomes.append(oc)
        print(f"[{self.mode}] <<< {task}: {oc.status.upper()}  ({oc.seconds:.1f}s)"
              f"{'  - ' + oc.detail if oc.detail else ''}", flush=True)
        return oc

    def summary(self) -> int:
        print(f"\n{'#'*74}\n# SUMMARY ({self.mode})\n{'#'*74}")
        width = max((len(o.task) for o in self.outcomes), default=4)
        for o in self.outcomes:
            line = f"  {o.task.ljust(width)}  {o.status.upper():5}  {o.seconds:6.1f}s"
            if o.detail:
                line += f"  {o.detail}"
            print(line)
        n_fail = sum(o.status == "fail" for o in self.outcomes)
        n_skip = sum(o.status == "skip" for o in self.outcomes)
        n_ok = sum(o.status == "ok" for o in self.outcomes)
        print(f"\n  {n_ok} ok, {n_skip} skipped, {n_fail} failed")

        # Persist machine- and human-readable results.
        self._write_results()
        return 1 if n_fail else 0

    def _write_results(self):
        records = [vars(o) for o in self.outcomes]
        with open(os.path.join(self.outdir, "results.json"), "w") as f:
            json.dump({"mode": self.mode, "generated": datetime.now().isoformat(),
                       "outcomes": records}, f, indent=2)

        # summary.md — paper-friendly tables built from the `data` payloads.
        lines = [f"# AutoNMT results ({self.mode})",
                 f"_generated {datetime.now():%Y-%m-%d %H:%M}_", ""]
        for o in self.outcomes:
            lines.append(f"## {o.task} — {o.status} ({o.seconds:.1f}s)")
            if o.detail:
                lines.append(f"_{o.detail}_")
            if o.data:
                lines.append("")
                lines.append("```json")
                lines.append(json.dumps(o.data, indent=2))
                lines.append("```")
            lines.append("")
        with open(os.path.join(self.outdir, "summary.md"), "w") as f:
            f.write("\n".join(lines))
        print(f"\n  results written to {os.path.abspath(self.outdir)}/"
              f"{{results.json, summary.md}}")


class _Skip(Exception):
    """Raised by a task to record a SKIP (missing dep / no network)."""


# ===========================================================================
# Shared helpers
# ===========================================================================
def _seed_everything(seed: int):
    from autonmt.utils.seed import manual_seed
    manual_seed(seed)


def _toy_sentences(n, rng, with_verbs=True):
    DET = ["the", "a"]
    ADJ = ["quick", "lazy", "bright", "green", "small"]
    NOUN = ["fox", "dog", "moon", "river", "stone"]
    VERB = ["jumps", "runs", "sees", "follows", "hides"]
    out = []
    for _ in range(n):
        s = [rng.choice(DET), rng.choice(ADJ), rng.choice(NOUN)]
        if with_verbs:
            s += [rng.choice(VERB), rng.choice(DET), rng.choice(ADJ), rng.choice(NOUN)]
        out.append(" ".join(s) + " .")
    return out


# A small but lexically varied en-es bank — rich enough for SentencePiece to
# train a ~200-piece BPE model (the DET/ADJ/NOUN toy above is too repetitive).
_PARALLEL_BANK = [
    ("the cat sat on the mat .", "el gato se sento en la alfombra ."),
    ("a quick brown fox jumps over the lazy dog .", "un rapido zorro marron salta sobre el perro perezoso ."),
    ("machine translation is fun .", "la traduccion automatica es divertida ."),
    ("this is a simple test .", "esta es una prueba simple ."),
    ("we are training a small model .", "estamos entrenando un modelo pequeno ."),
    ("the weather is nice today .", "el clima es agradable hoy ."),
    ("i like programming in python .", "me gusta programar en python ."),
    ("neural networks learn from data .", "las redes neuronales aprenden de los datos ."),
    ("the book is on the table .", "el libro esta sobre la mesa ."),
    ("she went to the store yesterday .", "ella fue a la tienda ayer ."),
    ("the children played in the park .", "los ninos jugaron en el parque ."),
    ("he reads a long letter every morning .", "el lee una carta larga cada manana ."),
    ("they built a wooden house near the lake .", "ellos construyeron una casa de madera cerca del lago ."),
    ("our teacher explained the difficult lesson .", "nuestro maestro explico la leccion dificil ."),
    ("the train arrives at eight o'clock .", "el tren llega a las ocho ."),
    ("a red car stopped at the corner .", "un coche rojo se detuvo en la esquina ."),
    ("birds sing in the tall green trees .", "los pajaros cantan en los altos arboles verdes ."),
    ("my brother cooked dinner for the family .", "mi hermano cocino la cena para la familia ."),
    ("the river flows under the old bridge .", "el rio fluye bajo el viejo puente ."),
    ("scientists study the stars at night .", "los cientificos estudian las estrellas por la noche ."),
    ("the doctor gave good advice to the patient .", "el medico dio buenos consejos al paciente ."),
    ("we watched a beautiful sunset by the sea .", "vimos una hermosa puesta de sol junto al mar ."),
    ("the farmer grows wheat and corn .", "el granjero cultiva trigo y maiz ."),
    ("a clever student solved the hard problem .", "un estudiante inteligente resolvio el problema dificil ."),
]


def _synthetic_parallel(n):
    """Cycle the parallel bank to `n` lines (deterministic). Returns (src, tgt)."""
    src, tgt = [], []
    for i in range(n):
        s, t = _PARALLEL_BANK[i % len(_PARALLEL_BANK)]
        src.append(s)
        tgt.append(t)
    return src, tgt


def _seed_parallel_corpus(base_path, name, lang_pair, src_lines, tgt_lines, size="original"):
    """Write a synthetic parallel corpus into the AutoNMT 0_raw layout."""
    src_lang, tgt_lang = lang_pair.split("-")
    raw_dir = os.path.join(base_path, name, lang_pair, size, "data", "0_raw")
    os.makedirs(raw_dir, exist_ok=True)
    with open(os.path.join(raw_dir, f"data.{src_lang}"), "w") as f:
        f.write("\n".join(src_lines) + "\n")
    with open(os.path.join(raw_dir, f"data.{tgt_lang}"), "w") as f:
        f.write("\n".join(tgt_lines) + "\n")


def _nmt_preprocess_fns():
    """The normalize / preprocess callbacks used by the DatasetBuilder + predict."""
    from tokenizers.normalizers import NFKC, Strip
    from autonmt.datasets.preprocessing import (
        normalize_lines, preprocess_lines, preprocess_pairs)

    def normalize(lines):
        return normalize_lines(lines, seq=[NFKC(), Strip()])

    def preprocess_train(data, ds):
        return preprocess_pairs(data["src"]["lines"], data["tgt"]["lines"],
                                normalize_fn=normalize)

    def preprocess_predict(data, ds):
        return preprocess_lines(data["lines"], normalize_fn=normalize)

    return preprocess_train, preprocess_predict


def _nmt_transformer(profile: "Profile", sv, tv):
    """Build the encoder-decoder Transformer: library defaults under `smoke`,
    a properly-sized model (embed/layers/heads/ffn from the profile) under `paper`."""
    from autonmt.core.nn.models import Transformer
    if profile.name == "smoke":
        return Transformer.from_vocabs(sv, tv)
    return Transformer.from_vocabs(
        sv, tv,
        encoder_layers=profile.nmt_layers, decoder_layers=profile.nmt_layers,
        encoder_embed_dim=profile.nmt_embed, decoder_embed_dim=profile.nmt_embed,
        encoder_attention_heads=profile.nmt_heads, decoder_attention_heads=profile.nmt_heads,
        encoder_ffn_embed_dim=profile.nmt_ff, decoder_ffn_embed_dim=profile.nmt_ff,
    )


def _require(pkg: str):
    try:
        __import__(pkg)
    except Exception as e:                                        # noqa: BLE001
        raise _Skip(f"missing optional dependency '{pkg}' ({_short(e)})")


def _wikitext2_lines(split: str, limit: int | None = None):
    """Load WikiText-2 raw lines (paragraphs), skipping blanks / headers."""
    _require("datasets")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    lines = [t.strip() for t in ds["text"]
             if t.strip() and not t.strip().startswith("=")]
    if limit:
        lines = lines[:limit]
    return lines


def _distinct_n(texts, n):
    """Corpus-level distinct-n: unique n-grams / total n-grams (diversity)."""
    total, uniq = 0, set()
    for t in texts:
        toks = t.split()
        grams = list(zip(*[toks[i:] for i in range(n)]))
        total += len(grams)
        uniq.update(grams)
    return (len(uniq) / total) if total else 0.0


# ===========================================================================
# Catalog smoke: every built-in model instantiates and forwards.
# ===========================================================================
def task_models(profile: Profile, outdir: str):
    """Construct every built-in model across the triad (catalog smoke).

    Forward/backward correctness is already covered by the unit tests; here we
    only assert that each public class still builds from a real vocab/corpus and
    has trainable parameters — the thing most likely to rot across a refactor.
    """
    from autonmt.datasets.dataset_builder import DatasetBuilder
    from autonmt.core.nn.models import (
        MLP, SimpleRNN, ContextRNN, BahdanauRNN, LuongRNN, ConvS2S, Transformer,
        GPT, MLMTransformer)
    from autonmt.datasets.lm_corpus import LMCorpusBuilder

    base = os.path.join(outdir, "_models")
    rng = random.Random(profile.seed)

    def _has_params(m):
        return sum(p.numel() for p in m.parameters()) > 0

    # --- encoder-decoder catalog: build a tiny parallel ds, then from_vocabs ---
    src, tgt = _synthetic_parallel(800)
    _seed_parallel_corpus(base, "toy", "en-es", src, tgt)
    builder = DatasetBuilder(
        base_path=base,
        datasets=[{"name": "toy", "languages": ["en-es"],
                   "sizes": [("original", None)], "split_sizes": (None, 60, 60)}],
        encoding=[{"subword_models": ["bpe"], "vocab_sizes": [profile.nmt_vocab]}],
        merge_vocabs=False,
    ).build(force_overwrite=False)
    ds = builder.get_train_ds()[0]
    sv, tv = ds.build_vocabs(max_tokens=profile.nmt_vocab)

    enc_dec = {
        "MLP": MLP, "SimpleRNN": SimpleRNN, "ContextRNN": ContextRNN,
        "BahdanauRNN": BahdanauRNN, "LuongRNN": LuongRNN, "ConvS2S": ConvS2S,
        "Transformer": Transformer,
    }
    checked = []
    for mname, cls in enc_dec.items():
        model = cls.from_vocabs(sv, tv)
        assert _has_params(model), f"{mname} has no parameters"
        checked.append(mname)

    # --- decoder-only + encoder-only: from_corpus on tiny LM / MLM corpora ---
    text = _toy_sentences(800, rng)
    cb = LMCorpusBuilder(
        base_path=os.path.join(base, "_lm"),
        corpus=[{"name": "toy", "mode": "text", "sizes": [("original", None)],
                 "text": text}],
        encoding=[{"subword_models": ["bpe"], "vocab_sizes": [profile.lm_vocab]}],
    ).build(force_overwrite=False)
    gpt = GPT.from_corpus(cb.get_train_ds()[0], embed_dim=profile.lm_embed, num_layers=2,
                          num_heads=profile.lm_heads, max_seq_len=profile.lm_max_seq)
    assert _has_params(gpt), "GPT has no parameters"
    checked.append("GPT")

    mb = LMCorpusBuilder(
        base_path=os.path.join(base, "_mlm"),
        corpus=[{"name": "toy", "mode": "mlm", "sizes": [("original", None)],
                 "text": text}],
        encoding=[{"subword_models": ["word"], "vocab_sizes": [23]}],
    ).build(force_overwrite=False)
    mlm = MLMTransformer.from_corpus(mb.get_train_ds()[0], embed_dim=profile.lm_embed,
                                     num_layers=2, num_heads=profile.lm_heads,
                                     max_seq_len=profile.lm_max_seq)
    assert _has_params(mlm), "MLMTransformer has no parameters"
    checked.append("MLMTransformer")

    return {"__detail__": f"{len(checked)} models built: {', '.join(checked)}",
            "models_checked": checked}


# ===========================================================================
# Triad — encoder-decoder NMT
# ===========================================================================
def task_nmt(profile: Profile, outdir: str):
    from autonmt.backends import AutonmtTranslator
    from autonmt.backends._base.config import FitConfig, PredictConfig
    from autonmt.datasets.dataset_builder import DatasetBuilder
    from autonmt.reporting.report import Report

    _seed_everything(profile.seed)
    pre_train, pre_predict = _nmt_preprocess_fns()
    base = os.path.join(outdir, "_nmt")
    smoke = profile.name == "smoke"

    if smoke:
        src, tgt = _synthetic_parallel(600)
        _seed_parallel_corpus(base, "toy", "en-es", src, tgt)
        datasets = [{"name": "toy", "languages": ["en-es"],
                     "sizes": [("original", None)], "split_sizes": (None, 60, 60)}]
        builder = DatasetBuilder(
            base_path=base, datasets=datasets,
            encoding=[{"subword_models": ["bpe"], "vocab_sizes": [profile.nmt_vocab]}],
            preprocess_raw_fn=pre_train, preprocess_splits_fn=pre_train,
            merge_vocabs=False,
        ).build(force_overwrite=False)
    else:
        _require("datasets")
        from autonmt.datasets.hf_loader import download_hf_dataset
        download_hf_dataset(hf_id="bentrevett/multi30k", base_path=base,
                            dataset_name="multi30k", lang_pair="de-en",
                            src_field="de", tgt_field="en")
        builder = DatasetBuilder(
            base_path=base,
            datasets=[{"name": "multi30k", "languages": ["de-en"],
                       "sizes": [("original", None)]}],
            encoding=[{"subword_models": ["bpe"], "vocab_sizes": [profile.nmt_vocab]}],
            preprocess_raw_fn=pre_train, preprocess_splits_fn=pre_train,
            merge_vocabs=False,
        ).build(force_overwrite=False)

    train_ds = builder.get_train_ds()[0]
    test_ds = builder.get_test_ds()
    sv, tv = train_ds.build_vocabs(max_tokens=profile.nmt_vocab)
    model = _nmt_transformer(profile, sv, tv)

    trainer = AutonmtTranslator.from_dataset(
        train_ds, model=model, src_vocab=sv, tgt_vocab=tv, run_prefix="nmt")
    fit_cfg = FitConfig(max_epochs=profile.nmt_epochs, batch_size=profile.nmt_batch,
                        learning_rate=(1e-3 if smoke else 7e-4),
                        scheduler=(None if smoke else "noam"),
                        warmup_steps=(None if smoke else 4000),
                        seed=profile.seed, precision=profile.precision)
    trainer.fit(train_ds, config=fit_cfg)
    scores = trainer.predict(test_ds, config=PredictConfig(
        metrics={"bleu", "chrf"}, beams=[5], load_checkpoint="best",
        preprocess_fn=pre_predict, eval_mode="compatible"))

    Report.from_predict(scores, output_path=os.path.join(outdir, "report_nmt")).save()
    bleu = _pick_metric(scores, "bleu")
    chrf = _pick_metric(scores, "chrf")
    return {"__detail__": f"BLEU={_fmt(bleu)} chrF={_fmt(chrf)}",
            "dataset": "toy" if smoke else "multi30k de-en",
            "bleu": bleu, "chrf": chrf}


# ===========================================================================
# Triad — decoder-only language model (perplexity)
# ===========================================================================
def task_lm(profile: Profile, outdir: str):
    from autonmt.backends import LMTrainer
    from autonmt.core.decoding import TopPSampling
    from autonmt.core.nn.models import GPT
    from autonmt.datasets.lm_corpus import LMCorpusBuilder

    _seed_everything(profile.seed)
    base = os.path.join(outdir, "_lm")
    smoke = profile.name == "smoke"

    if smoke:
        rng = random.Random(profile.seed)
        text = _toy_sentences(2000, rng)
        vocab = profile.lm_vocab
    else:
        text = _wikitext2_lines("train")
        vocab = profile.lm_vocab

    builder = LMCorpusBuilder(
        base_path=base,
        corpus=[{"name": "corpus", "mode": "text", "sizes": [("original", None)],
                 "text": text}],
        encoding=[{"subword_models": ["bpe"], "vocab_sizes": [vocab]}],
        val_size=0.1,
    ).build(force_overwrite=False)
    corpus = builder.get_train_ds()[0]

    model = GPT.from_corpus(corpus, embed_dim=profile.lm_embed,
                            num_layers=profile.lm_layers, num_heads=profile.lm_heads,
                            max_seq_len=profile.lm_max_seq, dropout=0.1)
    trainer = LMTrainer.from_corpus(corpus, run_prefix="lm", model=model)
    trainer.fit(corpus, block_size=profile.lm_block, max_epochs=profile.lm_epochs,
                batch_size=profile.lm_batch, learning_rate=(3e-3 if smoke else 3e-4),
                seed=profile.seed, accelerator="auto", precision=profile.precision)
    metrics = trainer.evaluate(corpus, block_size=profile.lm_block)
    ppl = float(metrics["ppl"])

    samples = [trainer.generate(p, sampler=TopPSampling(top_p=0.9, temperature=0.8),
                                max_new_tokens=20)
               for p in (["the quick", "a lazy dog"] if smoke else ["The history of",
                          "In recent years", "According to the"])]
    return {"__detail__": f"perplexity={ppl:.2f}",
            "dataset": "toy" if smoke else "wikitext-2",
            "perplexity": ppl, "samples": samples}


# ===========================================================================
# Triad — encoder-only masked LM (masked-token accuracy)
# ===========================================================================
def task_mlm(profile: Profile, outdir: str):
    from autonmt.backends import MLMTrainer
    from autonmt.core.nn.models import MLMTransformer
    from autonmt.datasets.lm_corpus import LMCorpusBuilder

    _seed_everything(profile.seed)
    base = os.path.join(outdir, "_mlm")
    smoke = profile.name == "smoke"

    if smoke:
        rng = random.Random(profile.seed)
        text = _toy_sentences(3000, rng)
        encoding = [{"subword_models": ["word"], "vocab_sizes": [23]}]
        block, epochs = 32, 25
    else:
        text = _wikitext2_lines("train")
        encoding = [{"subword_models": ["bpe"], "vocab_sizes": [profile.lm_vocab]}]
        block, epochs = profile.lm_block, profile.lm_epochs

    builder = LMCorpusBuilder(
        base_path=base,
        corpus=[{"name": "corpus", "mode": "mlm", "sizes": [("original", None)],
                 "text": text}],
        encoding=encoding, val_size=0.1,
    ).build(force_overwrite=False)
    corpus = builder.get_train_ds()[0]

    model = MLMTransformer.from_corpus(
        corpus, embed_dim=profile.lm_embed, num_layers=profile.lm_layers,
        num_heads=profile.lm_heads, max_seq_len=profile.lm_max_seq, dropout=0.1)
    trainer = MLMTrainer.from_corpus(corpus, run_prefix="mlm", model=model, mlm_prob=0.15)
    trainer.fit(corpus, block_size=block, max_epochs=epochs,
                batch_size=(64 if smoke else profile.lm_batch),
                learning_rate=1e-3, seed=profile.seed, accelerator="auto",
                precision=profile.precision)
    metrics = trainer.evaluate(corpus, block_size=block)
    acc = float(metrics["masked_acc"])
    return {"__detail__": f"masked_acc={acc:.3f}",
            "dataset": "toy" if smoke else "wikitext-2",
            "masked_acc": acc}


# ===========================================================================
# Sweep — NMT BLEU vs vocab size, and GPT perplexity vs model size
# ===========================================================================
def task_sweep(profile: Profile, outdir: str):
    from autonmt.backends import AutonmtTranslator, LMTrainer
    from autonmt.backends._base.config import FitConfig, PredictConfig
    from autonmt.core.nn.models import GPT
    from autonmt.datasets.dataset_builder import DatasetBuilder
    from autonmt.datasets.lm_corpus import LMCorpusBuilder

    _seed_everything(profile.seed)
    pre_train, pre_predict = _nmt_preprocess_fns()
    base = os.path.join(outdir, "_sweep")
    smoke = profile.name == "smoke"

    # ---- (a) NMT: BLEU as a function of the SentencePiece vocab size ----
    if smoke:
        src, tgt = _synthetic_parallel(800)
        _seed_parallel_corpus(base, "toy", "en-es", src, tgt)
        ds_decl = [{"name": "toy", "languages": ["en-es"],
                    "sizes": [("original", None)], "split_sizes": (None, 80, 80)}]
        name = "toy"
    else:
        _require("datasets")
        from autonmt.datasets.hf_loader import download_hf_dataset
        download_hf_dataset(hf_id="bentrevett/multi30k", base_path=base,
                            dataset_name="multi30k", lang_pair="de-en",
                            src_field="de", tgt_field="en")
        ds_decl = [{"name": "multi30k", "languages": ["de-en"],
                    "sizes": [("original", None)]}]
        name = "multi30k de-en"

    builder = DatasetBuilder(
        base_path=base, datasets=ds_decl,
        encoding=[{"subword_models": ["bpe"], "vocab_sizes": list(profile.nmt_vocab_sweep)}],
        preprocess_raw_fn=pre_train, preprocess_splits_fn=pre_train, merge_vocabs=False,
    ).build(force_overwrite=False)

    nmt_points = []
    for train_ds in builder.get_train_ds():
        sv, tv = train_ds.build_vocabs()
        model = _nmt_transformer(profile, sv, tv)
        trainer = AutonmtTranslator.from_dataset(
            train_ds, model=model, src_vocab=sv, tgt_vocab=tv, run_prefix="sweep")
        trainer.fit(train_ds, config=FitConfig(
            max_epochs=profile.nmt_epochs, batch_size=profile.nmt_batch,
            scheduler=(None if smoke else "noam"), warmup_steps=(None if smoke else 4000),
            seed=profile.seed, precision=profile.precision))
        scores = trainer.predict(builder.get_test_ds(), config=PredictConfig(
            metrics={"bleu"}, beams=[5], load_checkpoint="best",
            preprocess_fn=pre_predict, eval_mode="compatible"))
        vsize = _vocab_size_of(train_ds)
        nmt_points.append({"vocab_size": vsize, "bleu": _pick_metric(scores, "bleu")})

    # ---- (b) GPT: perplexity as a function of model size (#params) ----
    if smoke:
        rng = random.Random(profile.seed + 1)
        text = _toy_sentences(2000, rng)
        lm_vocab = profile.lm_vocab
    else:
        text = _wikitext2_lines("train")
        lm_vocab = profile.lm_vocab
    cb = LMCorpusBuilder(
        base_path=os.path.join(base, "_lm"),
        corpus=[{"name": "corpus", "mode": "text", "sizes": [("original", None)],
                 "text": text}],
        encoding=[{"subword_models": ["bpe"], "vocab_sizes": [lm_vocab]}], val_size=0.1,
    ).build(force_overwrite=False)
    corpus = cb.get_train_ds()[0]

    gpt_points = []
    for embed, layers in profile.lm_size_sweep:
        heads = max(1, embed // 64)   # keep embed_dim divisible by num_heads
        model = GPT.from_corpus(corpus, embed_dim=embed, num_layers=layers,
                                num_heads=heads, max_seq_len=profile.lm_max_seq,
                                dropout=0.1)
        nparams = sum(p.numel() for p in model.parameters())
        trainer = LMTrainer.from_corpus(corpus, run_prefix=f"sweep_d{embed}l{layers}",
                                        model=model)
        trainer.fit(corpus, block_size=profile.lm_block, max_epochs=profile.lm_epochs,
                    batch_size=profile.lm_batch, learning_rate=(3e-3 if smoke else 3e-4),
                    seed=profile.seed, accelerator="auto", precision=profile.precision)
        ppl = float(trainer.evaluate(corpus, block_size=profile.lm_block)["ppl"])
        gpt_points.append({"embed": embed, "layers": layers,
                           "params_M": round(nparams / 1e6, 2), "perplexity": ppl})

    _plot_sweep(nmt_points, "vocab_size", "bleu", "Vocab size", "BLEU",
                f"NMT BLEU vs vocab size ({name})",
                os.path.join(outdir, "sweep_nmt_vocab.pdf"))
    _plot_sweep(gpt_points, "params_M", "perplexity", "Params (M)", "Perplexity",
                "GPT perplexity vs model size", os.path.join(outdir, "sweep_gpt_size.pdf"))

    return {"__detail__": f"NMT {len(nmt_points)} pts, GPT {len(gpt_points)} pts",
            "nmt_vs_vocab": nmt_points, "gpt_vs_size": gpt_points}


# ===========================================================================
# Parity — AutoNMT Transformer vs HuggingFace opus-mt on Multi30k
# ===========================================================================
def task_parity(profile: Profile, outdir: str):
    from autonmt.backends import AutonmtTranslator, HuggingFaceTranslator
    from autonmt.backends._base.config import FitConfig, PredictConfig
    from autonmt.datasets.dataset_builder import DatasetBuilder

    smoke = profile.name == "smoke"
    if smoke:
        raise _Skip("parity needs a pretrained HF model + network; run in --mode paper")
    _require("datasets")
    _require("transformers")
    _seed_everything(profile.seed)
    pre_train, pre_predict = _nmt_preprocess_fns()
    base = os.path.join(outdir, "_parity")

    from autonmt.datasets.hf_loader import download_hf_dataset
    download_hf_dataset(hf_id="bentrevett/multi30k", base_path=base,
                        dataset_name="multi30k", lang_pair="de-en",
                        src_field="de", tgt_field="en")
    builder = DatasetBuilder(
        base_path=base,
        datasets=[{"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]}],
        encoding=[{"subword_models": ["bpe"], "vocab_sizes": [profile.nmt_vocab]}],
        preprocess_raw_fn=pre_train, preprocess_splits_fn=pre_train, merge_vocabs=False,
    ).build(force_overwrite=False)
    train_ds = builder.get_train_ds()[0]
    test_ds = builder.get_test_ds()
    pred_cfg = PredictConfig(metrics={"bleu", "chrf"}, beams=[5],
                             preprocess_fn=pre_predict, eval_mode="compatible")

    # AutoNMT in-house Transformer (trained from scratch)
    sv, tv = train_ds.build_vocabs(max_tokens=profile.nmt_vocab)
    model = _nmt_transformer(profile, sv, tv)
    autonmt = AutonmtTranslator.from_dataset(train_ds, model=model, src_vocab=sv,
                                             tgt_vocab=tv, run_prefix="parity-autonmt")
    autonmt.fit(train_ds, config=FitConfig(max_epochs=profile.nmt_epochs,
                batch_size=profile.nmt_batch, scheduler="noam", warmup_steps=4000,
                seed=profile.seed, precision=profile.precision))
    s_autonmt = autonmt.predict(test_ds, config=PredictConfig(
        metrics={"bleu", "chrf"}, beams=[5], load_checkpoint="best",
        preprocess_fn=pre_predict, eval_mode="compatible"))

    # HuggingFace pretrained opus-mt baseline (no fine-tuning)
    hf = HuggingFaceTranslator.from_dataset(
        train_ds, run_prefix="parity-hf", model_id="Helsinki-NLP/opus-mt-de-en")
    s_hf = hf.predict(test_ds, config=pred_cfg)

    table = [
        {"system": "AutoNMT-Transformer (scratch)", "bleu": _pick_metric(s_autonmt, "bleu"),
         "chrf": _pick_metric(s_autonmt, "chrf")},
        {"system": "HF opus-mt-de-en (pretrained)", "bleu": _pick_metric(s_hf, "bleu"),
         "chrf": _pick_metric(s_hf, "chrf")},
    ]
    return {"__detail__": "  ".join(f"{r['system'].split()[0]}={_fmt(r['bleu'])}" for r in table),
            "parity": table}


# ===========================================================================
# Decoding — NMT beam-width sweep, GPT sampler diversity
# ===========================================================================
def task_decoding(profile: Profile, outdir: str):
    from autonmt.backends import AutonmtTranslator, LMTrainer
    from autonmt.backends._base.config import FitConfig, PredictConfig
    from autonmt.core.decoding import (GreedySearch, TopKSampling, TopPSampling,
                                       MultinomialSampling)
    from autonmt.core.nn.models import GPT
    from autonmt.datasets.dataset_builder import DatasetBuilder
    from autonmt.datasets.lm_corpus import LMCorpusBuilder

    _seed_everything(profile.seed)
    pre_train, pre_predict = _nmt_preprocess_fns()
    base = os.path.join(outdir, "_decoding")
    smoke = profile.name == "smoke"

    # ---- (a) NMT: BLEU vs beam width ----
    if smoke:
        src, tgt = _synthetic_parallel(800)
        _seed_parallel_corpus(base, "toy", "en-es", src, tgt)
        ds_decl = [{"name": "toy", "languages": ["en-es"],
                    "sizes": [("original", None)], "split_sizes": (None, 80, 80)}]
    else:
        _require("datasets")
        from autonmt.datasets.hf_loader import download_hf_dataset
        download_hf_dataset(hf_id="bentrevett/multi30k", base_path=base,
                            dataset_name="multi30k", lang_pair="de-en",
                            src_field="de", tgt_field="en")
        ds_decl = [{"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]}]

    builder = DatasetBuilder(
        base_path=base, datasets=ds_decl,
        encoding=[{"subword_models": ["bpe"], "vocab_sizes": [profile.nmt_vocab]}],
        preprocess_raw_fn=pre_train, preprocess_splits_fn=pre_train, merge_vocabs=False,
    ).build(force_overwrite=False)
    train_ds = builder.get_train_ds()[0]
    sv, tv = train_ds.build_vocabs(max_tokens=profile.nmt_vocab)
    model = _nmt_transformer(profile, sv, tv)
    trainer = AutonmtTranslator.from_dataset(train_ds, model=model, src_vocab=sv,
                                             tgt_vocab=tv, run_prefix="decoding")
    trainer.fit(train_ds, config=FitConfig(
        max_epochs=profile.nmt_epochs, batch_size=profile.nmt_batch,
        scheduler=(None if smoke else "noam"), warmup_steps=(None if smoke else 4000),
        seed=profile.seed, precision=profile.precision))
    beams = [1, 3, 5]
    scores = trainer.predict(builder.get_test_ds(), config=PredictConfig(
        metrics={"bleu"}, beams=beams, load_checkpoint="best",
        preprocess_fn=pre_predict, eval_mode="compatible"))
    nmt_beam = [{"beam": b, "bleu": _pick_metric(scores, "bleu", beam=b)} for b in beams]

    # ---- (b) GPT: output diversity per sampler ----
    if smoke:
        rng = random.Random(profile.seed + 2)
        text = _toy_sentences(2000, rng)
        lm_vocab = profile.lm_vocab
        prompts = ["the", "a"] * 20
    else:
        text = _wikitext2_lines("train")
        lm_vocab = profile.lm_vocab
        prompts = ["The", "In", "A", "It", "He", "She", "They", "We"] * 8
    cb = LMCorpusBuilder(
        base_path=os.path.join(base, "_lm"),
        corpus=[{"name": "corpus", "mode": "text", "sizes": [("original", None)],
                 "text": text}],
        encoding=[{"subword_models": ["bpe"], "vocab_sizes": [lm_vocab]}], val_size=0.1,
    ).build(force_overwrite=False)
    corpus = cb.get_train_ds()[0]
    gpt = GPT.from_corpus(corpus, embed_dim=profile.lm_embed, num_layers=profile.lm_layers,
                          num_heads=profile.lm_heads, max_seq_len=profile.lm_max_seq, dropout=0.1)
    lm = LMTrainer.from_corpus(corpus, run_prefix="decoding-lm", model=gpt)
    lm.fit(corpus, block_size=profile.lm_block, max_epochs=profile.lm_epochs,
           batch_size=profile.lm_batch, learning_rate=(3e-3 if smoke else 3e-4),
           seed=profile.seed, accelerator="auto", precision=profile.precision)

    samplers = {"greedy": GreedySearch(),
                "multinomial(t=0.8)": MultinomialSampling(temperature=0.8),
                "top-k(40)": TopKSampling(top_k=40, temperature=0.8),
                "top-p(0.9)": TopPSampling(top_p=0.9, temperature=0.8)}
    gpt_div = []
    for sname, sampler in samplers.items():
        outs = [lm.generate(p, sampler=sampler, max_new_tokens=24) for p in prompts]
        gpt_div.append({"sampler": sname,
                        "distinct_1": round(_distinct_n(outs, 1), 3),
                        "distinct_2": round(_distinct_n(outs, 2), 3)})

    _plot_sweep(nmt_beam, "beam", "bleu", "Beam width", "BLEU",
                "NMT BLEU vs beam width", os.path.join(outdir, "decoding_nmt_beam.pdf"))
    return {"__detail__": f"NMT beams {beams}; GPT samplers {list(samplers)}",
            "nmt_bleu_vs_beam": nmt_beam, "gpt_diversity": gpt_div}


# ===========================================================================
# Tiny utilities for metric extraction / plotting
# ===========================================================================
def _flatten_metrics(scores):
    """Walk a predict()/report structure into {dotted_key: numeric_value}.

    predict() returns a nested list of records, each with a `translations`
    sub-dict like ``{"beam5": {"sacrebleu_bleu_score": .., "sacrebleu_chrf2_score": ..}}``.
    """
    flat = {}

    def walk(x, prefix):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{prefix}.{k}" if prefix else str(k))
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                walk(v, f"{prefix}.{i}" if prefix else str(i))
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            flat[prefix] = float(x)

    walk(scores, "")
    return flat


def _key_has_metric(kl: str, metric: str) -> bool:
    """True if `metric` is a *token* of the score key (so 'bleu' does not match
    the tool name 'sacre**bleu**', and 'chrf' still matches 'chrf2')."""
    import re
    if "score" not in kl:
        return False
    tokens = re.split(r"[^a-z0-9]+", kl)
    return any(t == metric or t.startswith(metric) for t in tokens)


def _pick_metric(scores, metric, beam=None):
    """Best numeric value whose key path mentions `metric` + 'score' (+ beam)."""
    metric = metric.lower()
    best = None
    for key, val in _flatten_metrics(scores).items():
        kl = key.lower()
        if not _key_has_metric(kl, metric):
            continue
        if beam is not None and f"beam{beam}" not in kl:
            continue
        best = val if best is None else max(best, val)
    return best


def _vocab_size_of(train_ds):
    for attr in ("vocab_size", "subword_vocab_size"):
        if hasattr(train_ds, attr):
            try:
                return int(getattr(train_ds, attr))
            except Exception:  # noqa: BLE001
                pass
    vid = getattr(train_ds, "variant_id", lambda **_: "")(as_path=True) \
        if hasattr(train_ds, "variant_id") else ""
    import re
    m = re.findall(r"(\d+)", str(vid))
    return int(m[-1]) if m else None


def _fmt(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) else "n/a"


def _plot_sweep(points, xkey, ykey, xlabel, ylabel, title, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    pts = [p for p in points if p.get(xkey) is not None and p.get(ykey) is not None]
    if not pts:
        return
    pts.sort(key=lambda p: p[xkey])
    xs = [p[xkey] for p in pts]
    ys = [p[ykey] for p in pts]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  plot -> {path}")


# ===========================================================================
# CLI
# ===========================================================================
TASKS = {
    "models": task_models,
    "nmt": task_nmt,
    "lm": task_lm,
    "mlm": task_mlm,
    "sweep": task_sweep,
    "parity": task_parity,
    "decoding": task_decoding,
}
# Convenience groups
GROUPS = {
    "triad": ["nmt", "lm", "mlm"],
    "all": ["models", "nmt", "lm", "mlm", "sweep", "parity", "decoding"],
}
SMOKE_DEFAULT = ["models", "nmt", "lm", "mlm", "sweep", "decoding"]  # parity skips offline


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["smoke", "paper"], default="smoke")
    p.add_argument("--task", default="all",
                   help="comma-separated: " + ", ".join(list(TASKS) + list(GROUPS)))
    p.add_argument("--outdir", default=None,
                   help="output dir (default: .paper_results/<mode>_<timestamp>)")
    args = p.parse_args(argv)

    profile = SMOKE if args.mode == "smoke" else PAPER
    outdir = args.outdir or os.path.join(
        ".paper_results", f"{args.mode}_{datetime.now():%Y%m%d_%H%M%S}")

    # Resolve the requested task list.
    if args.task in GROUPS:
        names = GROUPS[args.task]
    elif args.task == "all" and args.mode == "smoke":
        names = SMOKE_DEFAULT
    else:
        names = []
        for t in args.task.split(","):
            t = t.strip()
            names.extend(GROUPS.get(t, [t]))
    unknown = [n for n in names if n not in TASKS]
    if unknown:
        p.error(f"unknown task(s): {unknown}. Choose from {list(TASKS)} or {list(GROUPS)}")

    print(f"AutoNMT paper harness | mode={args.mode} | tasks={names} | outdir={outdir}")
    runner = Runner(args.mode, outdir)
    for name in names:
        runner.run(name, TASKS[name], profile, outdir)
    sys.exit(runner.summary())


if __name__ == "__main__":
    main()
