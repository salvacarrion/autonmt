"""Enums must keep string-equality compat with the legacy literal API."""
import os

import pytest

from autonmt.utils.enums import (
    EvalMode, SourceData, SubwordModel, has_vocab, is_bytes_only, is_no_model,
)


class TestStrCompat:
    def test_subword_model_equals_string(self):
        assert SubwordModel.BPE == "bpe"
        assert "bpe" == SubwordModel.BPE
        assert SubwordModel.BYTES in {None, "none", "bytes"}

    def test_str_returns_value_not_qualname(self):
        # Critical: members get stringified inside os.path.join calls.
        assert str(SubwordModel.UNIGRAM) == "unigram"
        assert str(EvalMode.SAME) == "same"
        assert str(SourceData.RAW_PREPROCESSED) == "raw_preprocessed"

    def test_os_path_join_uses_value(self):
        p = os.path.join("a", SubwordModel.UNIGRAM, "8000")
        assert p == "a/unigram/8000"


class TestCoerce:
    def test_subword_model_case_insensitive(self):
        assert SubwordModel.coerce("Word") is SubwordModel.WORD
        assert SubwordModel.coerce("BPE") is SubwordModel.BPE

    def test_subword_model_none_passthrough(self):
        assert SubwordModel.coerce(None) is None

    def test_subword_model_member_passthrough(self):
        assert SubwordModel.coerce(SubwordModel.BYTES) is SubwordModel.BYTES

    def test_subword_model_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown subword_model"):
            SubwordModel.coerce("nope")

    def test_eval_mode_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown eval_mode"):
            EvalMode.coerce("nonsense")

    def test_eval_mode_string_coerces(self):
        assert EvalMode.coerce("all") is EvalMode.ALL


class TestProperties:
    def test_uses_sentencepiece(self):
        assert SubwordModel.WORD.uses_sentencepiece
        assert SubwordModel.BPE.uses_sentencepiece
        assert not SubwordModel.BYTES.uses_sentencepiece
        assert not SubwordModel.NONE.uses_sentencepiece

    def test_has_vocab_excludes_none_and_bytes(self):
        assert has_vocab(SubwordModel.BPE)
        assert has_vocab(SubwordModel.UNIGRAM)
        assert not has_vocab(SubwordModel.NONE)
        assert not has_vocab(SubwordModel.BYTES)
        assert not has_vocab(None)

    def test_is_no_model_and_bytes_only(self):
        assert is_no_model(None)
        assert is_no_model(SubwordModel.NONE)
        assert not is_no_model(SubwordModel.BPE)
        assert is_bytes_only(SubwordModel.BYTES)
        assert not is_bytes_only(SubwordModel.WORD)


class TestNoCompoundMembers:
    """+bytes is a separate flag (byte_fallback), not an enum member."""

    def test_compound_strings_no_longer_resolve(self):
        for name in ("char+bytes", "unigram+bytes", "bpe+bytes"):
            with pytest.raises(ValueError, match="Unknown subword_model"):
                SubwordModel.coerce(name)
