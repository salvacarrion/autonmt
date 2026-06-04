"""Unified `precision` knob → per-backend dialect mapping.

The canonical vocabulary is {"fp32", "fp16", "bf16"}; each backend renders it
differently (Lightning string / HF bool kwargs / fairseq CLI flags). The
autonmt (Lightning) backend additionally passes unknown values through so power
users can hand Lightning a native string.
"""
import pytest

from autonmt.backends._base.precision import (
    PRECISIONS, to_lightning, to_hf_kwargs, to_fairseq_args,
)


class TestLightning:
    @pytest.mark.parametrize("value,expected", [
        ("fp32", "32-true"),
        ("fp16", "16-mixed"),
        ("bf16", "bf16-mixed"),
    ])
    def test_canonical_tokens_map(self, value, expected):
        assert to_lightning(value) == expected

    def test_unknown_passes_through(self):
        # Power users can hand Lightning a native string directly.
        assert to_lightning("16-true") == "16-true"
        assert to_lightning("64") == "64"

    def test_none_disables(self):
        assert to_lightning(None) is None


class TestHuggingFace:
    def test_fp16(self):
        assert to_hf_kwargs("fp16") == {"fp16": True}

    def test_bf16(self):
        assert to_hf_kwargs("bf16") == {"bf16": True}

    def test_fp32_is_noop(self):
        assert to_hf_kwargs("fp32") == {}

    def test_never_sets_both(self):
        # HF errors if fp16 and bf16 are both True; the mapping sets at most one.
        for v in PRECISIONS:
            assert len(to_hf_kwargs(v)) <= 1

    def test_unknown_is_noop(self):
        # Lightning-native strings aren't HF concepts → leave HF at fp32.
        assert to_hf_kwargs("16-true") == {}


class TestFairseq:
    def test_fp16(self):
        assert to_fairseq_args("fp16") == ["--fp16"]

    def test_bf16(self):
        assert to_fairseq_args("bf16") == ["--bf16"]

    def test_fp32_is_empty(self):
        assert to_fairseq_args("fp32") == []

    def test_unknown_is_empty(self):
        assert to_fairseq_args("16-true") == []
