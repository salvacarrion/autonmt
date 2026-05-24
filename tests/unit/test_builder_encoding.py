"""Encoding-list unrolling logic for DatasetBuilder."""
from autonmt.preprocessing.builder import DatasetBuilder


def _builder():
    # Empty datasets list -> no disk access; we only exercise _unroll_encoding.
    return DatasetBuilder(base_path="/tmp", datasets=[])


class TestEncodingSugar:
    def test_plus_bytes_resolves_to_flag(self):
        out = _builder()._unroll_encoding([
            {"subword_models": ["bpe+bytes"], "vocab_sizes": [8000]},
        ])
        assert out == [{"subword_model": "bpe", "vocab_size": 8000, "byte_fallback": True}]

    def test_entry_level_flag_applies_to_plain_models(self):
        out = _builder()._unroll_encoding([
            {"subword_models": ["bpe", "unigram"], "vocab_sizes": [8000], "byte_fallback": True},
        ])
        assert {(e["subword_model"], e["byte_fallback"]) for e in out} == {
            ("bpe", True), ("unigram", True),
        }

    def test_plus_bytes_overrides_false_entry_flag(self):
        # "bpe+bytes" wins over the entry-level byte_fallback=False
        out = _builder()._unroll_encoding([
            {"subword_models": ["bpe+bytes"], "vocab_sizes": [8000], "byte_fallback": False},
        ])
        assert out[0]["byte_fallback"] is True

    def test_canonical_dedup_across_forms(self):
        # "bpe+bytes" and ("bpe", byte_fallback=True) are the same entry; the
        # builder must deduplicate them.
        out = _builder()._unroll_encoding([
            {"subword_models": ["bpe+bytes"], "vocab_sizes": [8000]},
            {"subword_models": ["bpe"], "vocab_sizes": [8000], "byte_fallback": True},
        ])
        assert len(out) == 1

    def test_plain_and_fallback_are_distinct(self):
        out = _builder()._unroll_encoding([
            {"subword_models": ["bpe"], "vocab_sizes": [8000]},
            {"subword_models": ["bpe+bytes"], "vocab_sizes": [8000]},
        ])
        assert len(out) == 2
