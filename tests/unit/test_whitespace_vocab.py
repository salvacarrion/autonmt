"""Regression tests for autonmt.vocabularies.whitespace_vocab.Vocabulary."""
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


def test_save_then_load_roundtrip(tmp_path):
    """Save a vocab with specials, reload it via ``_build_from_vocab`` with
    ``includes_special_tokens=True`` (the renamed parameter — the previous typo
    ``includes_special_tokes`` would silently swap the branch and double the
    special tokens). The reloaded vocab must match the saved one and pass the
    invariant check on special-token positions."""
    original = Vocabulary()
    original.voc2idx = {"hello": 4, "world": 5}
    original.idx2voc = {4: "hello", 5: "world"}
    original.voc2freq = {"hello": 12, "world": 7}

    out = tmp_path / "vocab.txt"
    original.save(filename=str(out), include_special_tokens=True)

    reloaded = Vocabulary()
    reloaded._build_from_vocab(str(out), includes_special_tokens=True)

    assert reloaded.voc2idx[reloaded.unk_piece] == reloaded.unk_id
    assert reloaded.voc2idx[reloaded.sos_piece] == reloaded.sos_id
    assert reloaded.voc2idx[reloaded.eos_piece] == reloaded.eos_id
    assert reloaded.voc2idx[reloaded.pad_piece] == reloaded.pad_id
    assert reloaded.voc2idx["hello"] == 4
    assert reloaded.voc2idx["world"] == 5
    assert len(reloaded.voc2idx) == 6  # 4 specials + hello + world
    # Frequencies must round-trip as ints, not raw '12\n' strings — otherwise
    # repeated save→load cycles accumulate stray newlines in the file.
    assert reloaded.voc2freq["hello"] == 12
    assert reloaded.voc2freq["world"] == 7
