"""Regression tests for autonmt.vocabularies.whitespace_vocab.Vocabulary."""
import os

from autonmt.vocabularies.whitespace_vocab import Vocabulary


def test_save_with_special_tokens_does_not_mix_tuples_and_strings(tmp_path):
    """Previously, save(include_special_tokens=True) appended (str, int) tuples for
    specials and strings for regular tokens, then handed the mixed list to
    write_file_lines, which crashed with `TypeError: can only concatenate tuple ...
    to tuple` on `line + tail`. Guard against the regression.
    """
    vocab = Vocabulary()
    vocab.voc2idx = {"hello": 4, "world": 5}
    vocab.idx2voc = {4: "hello", 5: "world"}
    vocab.voc2freq = {"hello": 12, "world": 7}

    out = tmp_path / "vocab.txt"
    vocab.save(filename=str(out), include_special_tokens=True)

    assert out.exists()
    lines = out.read_text().splitlines()
    # 4 specials + 2 regular tokens
    assert len(lines) == 6
    assert lines[0].startswith("<unk>\t")
    assert lines[1].startswith("<s>\t")
    assert lines[2].startswith("</s>\t")
    assert lines[3].startswith("<pad>\t")
    assert lines[4] == "hello\t12"
    assert lines[5] == "world\t7"
