"""Translation-quality metrics.

Each scoring backend (sacrebleu, bertscore, comet, huggingface) is described by
a :class:`MetricBackend` that knows how to score files into a JSON/txt artifact
*and* how to parse that artifact back into a flat ``{metric_name: {field: value}}``
dict. ``BaseTranslator`` consumes the :data:`METRIC_BACKENDS` registry directly
— there is no parallel parser table to keep in sync.

Backends that need optional dependencies (``comet``, ``evaluate``) raise at
*call time*, not at import time, so the package keeps importing on machines
without them.

References
----------
Papineni et al. (2002). *BLEU: a Method for Automatic Evaluation of Machine
Translation.* https://aclanthology.org/P02-1040/

Popović (2015). *chrF: character n-gram F-score for automatic MT evaluation.*
https://aclanthology.org/W15-3049/

Snover et al. (2006). *A Study of Translation Edit Rate with Targeted Human
Annotation* (TER). https://aclanthology.org/2006.amta-papers.25/

Post (2018). *A Call for Clarity in Reporting BLEU Scores* (sacrebleu).
[arXiv:1804.08771](https://arxiv.org/abs/1804.08771)

Zhang et al. (2020). *BERTScore: Evaluating Text Generation with BERT.*
[arXiv:1904.09675](https://arxiv.org/abs/1904.09675)

Rei et al. (2020). *COMET: A Neural Framework for MT Evaluation.*
[arXiv:2009.09025](https://arxiv.org/abs/2009.09025)
"""
from __future__ import annotations

import functools
import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Iterable, List, Set

import bert_score
import sacrebleu

from autonmt.utils import fileio
from autonmt.utils.logger import get_logger

log = get_logger(__name__)


# HuggingFace migrated metrics out of ``datasets`` into the standalone ``evaluate``
# package (``load_metric`` was removed). We support both for backwards compatibility.
try:
    from evaluate import load as _hf_load_metric
except ImportError:
    try:
        from datasets import load_metric as _hf_load_metric  # legacy datasets<2.20
    except ImportError:
        _hf_load_metric = None


# ---------------------------------------------------------------------------
# Score parsers (read a backend's output artifact -> {metric: {field: value}})
# ---------------------------------------------------------------------------

def _parse_json_metrics(text, fields):
    result = {}
    metrics = json.loads("".join(text))
    metrics = [metrics] if isinstance(metrics, dict) else metrics
    for m_dict in metrics:
        m_name = m_dict['name'].lower().strip()
        result[m_name] = {f: float(m_dict[f]) for f in fields}
    return result


def parse_score_json(text):
    """For sacrebleu / comet / huggingface: a single ``score`` field per metric."""
    return _parse_json_metrics(text, fields={"score"})


def parse_bertscore_json(text):
    return _parse_json_metrics(text, fields={"precision", "recall", "f1"})


def parse_fairseq_txt(text):
    pattern = r"beam=(\d+): BLEU = (\d+\.\d*)"
    groups = re.search(pattern, text[-1].strip()).groups()
    return {"bleu": {"score": float(groups[1])}}


# ---------------------------------------------------------------------------
# Score computers
# ---------------------------------------------------------------------------

_SACREBLEU_METRICS = {
    "bleu": sacrebleu.metrics.BLEU,
    "chrf": sacrebleu.metrics.CHRF,
    "ter":  sacrebleu.metrics.TER,
}


def score_sacrebleu(hyp_lines, ref_lines, metrics: Iterable[str],
                    tgt_lang: str = "", tokenize=None) -> List[dict]:
    """Public scorer used both by the eval pipeline and by validation-time BLEU
    inside ``LitSeq2Seq``. Returns the list-of-dicts shape sacrebleu produces."""
    scores = []
    for name in metrics:
        cls = _SACREBLEU_METRICS.get(name)
        if cls is None:
            continue
        # NOTE: sacrebleu's BLEU kwarg is spelled ``trg_lang`` (external API);
        # don't rename it to match our internal ``tgt_lang`` variable.
        scorer = cls(trg_lang=tgt_lang, tokenize=tokenize) if name == "bleu" else cls()
        d = scorer.corpus_score(hyp_lines, [ref_lines]).__dict__
        d["signature"] = str(scorer.get_signature())
        scores.append(d)
    return scores


def _score_bertscore(hyp_lines, ref_lines, lang):
    precision, recall, f1 = bert_score.score(hyp_lines, ref_lines, lang=lang)
    return [{"name": "bertscore",
             "precision": float(precision.mean()),
             "recall": float(recall.mean()),
             "f1": float(f1.mean())}]


# Override with AUTONMT_COMET_MODEL=<checkpoint> to use a different one
# (e.g. "Unbabel/wmt22-cometkiwi-da" for reference-free scoring — note that
# the current pipeline still passes ref, so swap only with a ref-based model).
COMET_CHECKPOINT = os.environ.get("AUTONMT_COMET_MODEL", "Unbabel/wmt22-comet-da")


@functools.lru_cache(maxsize=2)
def _load_comet_model(checkpoint_name: str):
    # Cached so consecutive (subset, beam) eval passes reuse the same
    # ~2 GB model instead of re-downloading + reloading it from disk.
    try:
        import comet
    except ImportError as e:
        raise ImportError(
            "'unbabel-comet' is required to compute COMET scores. "
            "Install with: pip install unbabel-comet"
        ) from e
    return comet.load_from_checkpoint(comet.download_model(checkpoint_name))


def _score_comet(src_lines, hyp_lines, ref_lines):
    model = _load_comet_model(COMET_CHECKPOINT)
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(src_lines, hyp_lines, ref_lines)]
    _, sys_score = model.predict(data)
    return [{"name": "comet", "score": sys_score}]


def _score_huggingface(hyp_lines, ref_lines, metrics):
    if _hf_load_metric is None:
        raise ImportError(
            "HuggingFace metrics require the 'evaluate' package (or legacy 'datasets'<2.20). "
            "Install with: pip install evaluate"
        )
    scores = []
    for metric in metrics:
        try:
            hg_metric = _hf_load_metric(metric)
            hg_metric.add_batch(predictions=list(hyp_lines),
                                references=[[r] for r in ref_lines])
            d = {"name": metric}
            d.update(hg_metric.compute())
            scores.append(d)
        except Exception as e:
            log.info(f"\t- [HUGGINGFACE ERROR]: Ignoring metric: {metric}.\n"
                     f"\t                       Message: {e}")
    return scores


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

def _read_aligned(*paths):
    lines = [fileio.read_file_lines(p, autoclean=True) for p in paths]
    assert len({len(L) for L in lines}) == 1, "ref/hyp/src must be line-aligned"
    for p, L in zip(paths, lines):
        if not L:
            raise ValueError(f"Empty file: {p}")
        # Blank lines slip past sacrebleu but break bertscore / COMET (NaN or
        # tokenizer errors). Warn rather than raise — the user may have legit
        # empty refs (e.g. unanswerable items) and want to score the rest.
        n_blank = sum(1 for line in L if not line.strip())
        if n_blank:
            log.warning(f"\t- {n_blank}/{len(L)} blank line(s) in {p} — "
                        f"bertscore/COMET may produce NaN")
    return lines


def _compute_sacrebleu(*, ref_file, hyp_file, output_file, metrics, **_):
    if not metrics:
        return
    ref, hyp = _read_aligned(ref_file, hyp_file)
    fileio.save_json(score_sacrebleu(hyp, ref, metrics=metrics), output_file)


def _compute_bertscore(*, ref_file, hyp_file, output_file, tgt_lang, **_):
    ref, hyp = _read_aligned(ref_file, hyp_file)
    fileio.save_json(_score_bertscore(hyp, ref, lang=tgt_lang), output_file)


def _compute_comet(*, src_file, ref_file, hyp_file, output_file, **_):
    src, ref, hyp = _read_aligned(src_file, ref_file, hyp_file)
    fileio.save_json(_score_comet(src, hyp, ref), output_file)


def _compute_huggingface(*, ref_file, hyp_file, output_file, metrics, **_):
    if not metrics:
        return
    ref, hyp = _read_aligned(ref_file, hyp_file)
    fileio.save_json(_score_huggingface(hyp, ref, metrics), output_file)


def _extract_fairseq_score(*, hyp_file, output_file, **_):
    """Copy the BLEU summary line out of fairseq's ``generate-test.txt`` artifact.

    Fairseq computes BLEU internally during translation; we do not re-score.
    Skips silently if the artifact is missing (older runs / non-fairseq engines).
    """
    generate_test_path = os.path.join(os.path.dirname(hyp_file), "generate-test.txt")
    if not os.path.exists(generate_test_path):
        log.info("\t- No 'generate-test.txt' was found.")
        return
    last = fileio.read_file_lines(generate_test_path, autoclean=True)[-1]
    fileio.write_file_lines(lines=[last], filename=output_file, insert_break_line=True)


@dataclass(frozen=True)
class MetricBackend:
    """One scoring backend.

    Attributes:
        name:        identifier used in paths and registry lookup.
        metrics:     metric names this backend can compute (e.g. {"bleu","chrf","ter"}).
        compute_fn:  ``(*, ref_file, hyp_file, src_file?, output_file, metrics?, tgt_lang?) -> None``.
                     Writes the score artifact to ``output_file``.
        parse_fn:    ``(text_lines) -> {metric_name: {field_name: value}}``.
        output_ext:  artifact extension (``json``/``txt``).
        needs_src:   whether ``compute_fn`` consumes the source file (comet does).
        explicit_only: if True, only triggered when one of its metrics is *explicitly*
                       in the user-requested set (used by huggingface ``hg_*`` prefix).
    """
    name: str
    metrics: FrozenSet[str]
    compute_fn: Callable[..., None]
    parse_fn: Callable[[List[str]], Dict[str, Dict[str, float]]]
    output_ext: str = "json"
    needs_src: bool = False
    explicit_only: bool = False

    @property
    def output_filename(self) -> str:
        return f"{self.name}_scores.{self.output_ext}"


def _build_registry():
    backends = [
        MetricBackend(
            name="sacrebleu",
            metrics=frozenset({"bleu", "chrf", "ter"}),
            compute_fn=_compute_sacrebleu,
            parse_fn=parse_score_json,
        ),
        MetricBackend(
            name="bertscore",
            metrics=frozenset({"bertscore"}),
            compute_fn=_compute_bertscore,
            parse_fn=parse_bertscore_json,
        ),
        MetricBackend(
            name="comet",
            metrics=frozenset({"comet"}),
            compute_fn=_compute_comet,
            parse_fn=parse_score_json,
            needs_src=True,
        ),
        MetricBackend(
            name="huggingface",
            metrics=frozenset(),  # opt-in via "hg_*" prefix; no canonical metric list
            compute_fn=_compute_huggingface,
            parse_fn=parse_score_json,
            explicit_only=True,
        ),
        MetricBackend(
            name="fairseq",
            metrics=frozenset({"fairseq"}),
            compute_fn=_extract_fairseq_score,
            parse_fn=parse_fairseq_txt,
            output_ext="txt",
        ),
    ]
    return {b.name: b for b in backends}


METRIC_BACKENDS: Dict[str, MetricBackend] = _build_registry()

# Reverse index (metric name -> owning backend), built once from the registry.
_METRIC_TO_BACKEND: Dict[str, MetricBackend] = {
    m: b for b in METRIC_BACKENDS.values() for m in b.metrics
}


def metric_to_backend() -> Dict[str, MetricBackend]:
    """Reverse index: each metric name -> the backend that owns it."""
    return _METRIC_TO_BACKEND


def resolve_backends(metrics: Iterable[str]) -> Dict[MetricBackend, Set[str]]:
    """Group a flat set of requested metrics by the backend that handles them.

    ``hg_<metric>`` requests are routed to the huggingface backend (stripped prefix).
    Returns ``{backend: {metric_names_for_that_backend}}``.
    """
    grouped: Dict[MetricBackend, Set[str]] = {}
    reverse = _METRIC_TO_BACKEND
    hg = METRIC_BACKENDS["huggingface"]
    for m in metrics:
        if m.startswith("hg_"):
            grouped.setdefault(hg, set()).add(m[3:])
        elif m in reverse:
            grouped.setdefault(reverse[m], set()).add(m)
    return grouped
