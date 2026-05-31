"""Unit tests for the reporting layer (``autonmt.reporting.report``).

These are pure-data tests — no matplotlib, no disk. They cover the bits with
real logic: the constructors / accumulation, the score-key flattening, and the
generic metric resolver (beam inference, beam / tool / split disambiguation,
and the token-aware matching that stops ``"bleu"`` matching ``sacrebleu_chrf``).
"""
import pytest

from autonmt.reporting.report import (
    Report,
    _metric_matches,
    _split_of,
)


def _score(train, test, translations):
    """One (run, eval_ds) score dict, shaped like ``build_run_report`` output."""
    return {
        "engine": "autonmt",
        "run_name": f"{train}-{test}",
        "train_dataset": train,
        "test_dataset": test,
        "train__lang_pair": "es-en",
        "test__lang_pair": "es-en",
        "vocab__subword_model": "bpe",
        "vocab__size": 8000,
        "model__architecture": "transformer",
        "model__total_params": 1000,
        "translations": translations,
    }


# Single beam, single tool, single metric — the common case.
_SIMPLE = [
    _score("ds1", "ds1", {"beam5": {"sacrebleu_bleu_score": 30.0}}),
    _score("ds1", "ds2", {"beam5": {"sacrebleu_bleu_score": 25.0}}),
]


# ---------------------------------------------------------------------------
# Token-aware matching
# ---------------------------------------------------------------------------

class TestMetricMatches:
    def test_bleu_does_not_match_chrf_via_tool_name(self):
        # 'sacrebleu' contains the substring 'bleu' — must NOT match chrf.
        assert _metric_matches("bleu", "translations.beam5.sacrebleu_bleu_score")
        assert not _metric_matches("bleu", "translations.beam5.sacrebleu_chrf_score")

    def test_compound_and_other_tools(self):
        assert _metric_matches("chrf", "translations.beam5.sacrebleu_chrf_score")
        assert _metric_matches("hg_bleu", "translations.beam5.hg_bleu_score")
        assert _metric_matches("bertscore_f1", "translations.beam1.bertscore_bertscore_f1")

    def test_full_column_name_matches_itself(self):
        col = "translations.beam5.sacrebleu_bleu_score"
        assert _metric_matches(col, col)


class TestSplitOf:
    def test_no_split_shape(self):
        assert _split_of("translations.beam5.sacrebleu_bleu_score") is None

    def test_subset_shape(self):
        assert _split_of("translations.taskA.beam5.sacrebleu_bleu_score") == "taskA"


# ---------------------------------------------------------------------------
# Constructors / accumulation
# ---------------------------------------------------------------------------

class TestConstructors:
    def test_from_predict_one_run(self):
        r = Report.from_predict(_SIMPLE)
        assert len(r.df) == 2

    def test_from_runs_grid(self):
        r = Report.from_runs([_SIMPLE, _SIMPLE])
        assert len(r.df) == 4

    def test_add_appends_and_invalidates_cache(self):
        r = Report.from_predict(_SIMPLE)
        assert len(r.df) == 2
        r.add([_score("ds3", "ds3", {"beam5": {"sacrebleu_bleu_score": 10.0}})])
        assert len(r.df) == 3

    def test_summary_keeps_id_and_metric_columns(self):
        r = Report.from_predict(_SIMPLE)
        cols = set(r.summary.columns)
        assert "train_dataset" in cols
        assert any("bleu" in c for c in cols)
        assert set(r.summary.columns).issubset(set(r.df.columns))


# ---------------------------------------------------------------------------
# Metric resolution
# ---------------------------------------------------------------------------

class TestResolveMetric:
    def test_available_metrics(self):
        r = Report.from_predict(_SIMPLE)
        assert r.available_metrics() == ["sacrebleu_bleu_score"]

    def test_single_beam_inferred(self):
        r = Report.from_predict(_SIMPLE)
        assert r.resolve_metric("bleu") == "translations.beam5.sacrebleu_bleu_score"

    def test_multi_beam_requires_beam(self):
        runs = [_score("ds1", "ds1", {
            "beam1": {"sacrebleu_bleu_score": 28.0},
            "beam5": {"sacrebleu_bleu_score": 30.0},
        })]
        r = Report.from_predict(runs)
        with pytest.raises(ValueError, match="ambiguous across beams"):
            r.resolve_metric("bleu")
        assert r.resolve_metric("bleu", beam=5) == "translations.beam5.sacrebleu_bleu_score"
        assert r.resolve_metric("bleu", beam=1) == "translations.beam1.sacrebleu_bleu_score"

    def test_tool_disambiguation(self):
        runs = [_score("ds1", "ds1", {"beam5": {
            "sacrebleu_bleu_score": 30.0,
            "hg_bleu_score": 29.0,
        }})]
        r = Report.from_predict(runs)
        with pytest.raises(ValueError, match="ambiguous"):
            r.resolve_metric("bleu")
        assert r.resolve_metric("bleu", tool="hg") == "translations.beam5.hg_bleu_score"
        assert r.resolve_metric("bleu", tool="sacrebleu") == "translations.beam5.sacrebleu_bleu_score"

    def test_split_disambiguation(self):
        runs = [_score("ds1", "ds1", {
            "taskA": {"beam5": {"sacrebleu_bleu_score": 30.0}},
            "taskB": {"beam5": {"sacrebleu_bleu_score": 20.0}},
        })]
        r = Report.from_predict(runs)
        with pytest.raises(ValueError, match="ambiguous across splits"):
            r.resolve_metric("bleu")
        assert r.resolve_metric("bleu", split="taskA") == \
            "translations.taskA.beam5.sacrebleu_bleu_score"

    def test_unknown_metric_raises(self):
        r = Report.from_predict(_SIMPLE)
        with pytest.raises(ValueError, match="No metric column matches"):
            r.resolve_metric("rouge")

    def test_full_column_name_pins_beam(self):
        runs = [_score("ds1", "ds1", {
            "beam1": {"sacrebleu_bleu_score": 28.0},
            "beam5": {"sacrebleu_bleu_score": 30.0},
        })]
        r = Report.from_predict(runs)
        # The full name already pins beam5, so no beam= needed.
        col = "translations.beam5.sacrebleu_bleu_score"
        assert r.resolve_metric(col) == col


# ---------------------------------------------------------------------------
# Output-path guard
# ---------------------------------------------------------------------------

def test_save_without_output_path_raises():
    r = Report.from_predict(_SIMPLE)
    with pytest.raises(ValueError, match="No output_path"):
        r.save()
