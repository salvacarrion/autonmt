"""Snapshot tests for the on-disk layout. These paths are persisted across runs
and are part of AutoNMT's public contract — changing them would invalidate every
existing experiment folder, so the layout is pinned here.
"""
import pytest

from autonmt.bundle.enums import SubwordModel
from autonmt.preprocessing.dataset import Dataset, DatasetLayout


@pytest.fixture
def ds_bpe():
    return Dataset(
        base_path="/tmp/x", parent_ds=False,
        dataset_name="multi30k", dataset_lang_pair="de-en",
        dataset_size_name="small", dataset_lines=None,
        splits_sizes=(None, 100, 100),
        subword_model="bpe", vocab_size=8000, merge_vocabs=False,
    )


class TestDatasetPaths:
    def test_raw_layout(self, ds_bpe):
        assert ds_bpe.get_raw_path("data.de") == "/tmp/x/multi30k/de-en/small/data/0_raw/data.de"

    def test_split_layout(self, ds_bpe):
        assert ds_bpe.get_split_path("train.de") == "/tmp/x/multi30k/de-en/small/data/1_splits/train.de"

    def test_encoded_includes_subword_and_vocab_size(self, ds_bpe):
        assert ds_bpe.get_encoded_path("train.de") == \
            "/tmp/x/multi30k/de-en/small/data/4_encoded/bpe/8000/train.de"

    def test_vocab_file_per_language(self, ds_bpe):
        assert ds_bpe.get_vocab_file(lang="de") == "/tmp/x/multi30k/de-en/small/vocabs/bpe/8000/de"

    def test_run_path_layout(self, ds_bpe):
        assert ds_bpe.get_runs_path("autonmt") == "/tmp/x/multi30k/de-en/small/models/autonmt/runs/"

    def test_str_repr(self, ds_bpe):
        assert str(ds_bpe) == "multi30k_de-en_small_bpe_8000"

    def test_split_filenames_pair_each_lang(self, ds_bpe):
        assert ds_bpe.get_split_fnames() == [
            "train.de", "train.en", "val.de", "val.en", "test.de", "test.en"
        ]


class TestSubwordModelEdgeCases:
    def test_bytes_uses_singleton_vocab_segment(self):
        ds = Dataset(
            base_path="/x", parent_ds=False, dataset_name="d", dataset_lang_pair="es-en",
            dataset_size_name="s", dataset_lines=None, splits_sizes=(None, 10, 10),
            subword_model="bytes", vocab_size=None, merge_vocabs=False,
        )
        assert ds.vocab_size_id() == ["bytes"]
        assert ds.get_encoded_path("train.es") == "/x/d/es-en/s/data/4_encoded/bytes/train.es"

    def test_none_falls_back_to_preprocessed(self):
        ds = Dataset(
            base_path="/x", parent_ds=False, dataset_name="d", dataset_lang_pair="es-en",
            dataset_size_name="s", dataset_lines=None, splits_sizes=(None, 10, 10),
            subword_model=None, vocab_size=None, merge_vocabs=False,
        )
        assert ds.subword_model is None
        assert ds.get_encoded_path("train.es") == "/x/d/es-en/s/data/2_preprocessed/train.es"

    def test_word_triggers_pretokenization(self):
        ds = Dataset(
            base_path="/x", parent_ds=False, dataset_name="d", dataset_lang_pair="es-en",
            dataset_size_name="s", dataset_lines=None, splits_sizes=(None, 10, 10),
            subword_model="word", vocab_size=4000, merge_vocabs=False,
        )
        assert ds.pretok_flag is True


class TestDatasetLayoutStandalone:
    """DatasetLayout can be used without a full Dataset for path inspection."""

    def test_basic_construction(self):
        lay = DatasetLayout(
            base_path="/tmp", dataset_name="x", dataset_lang_pair="de-en",
            dataset_size_name="100k",
            subword_model=SubwordModel.UNIGRAM, vocab_size="4000",
        )
        assert lay.get_encoded_path("train.de") == "/tmp/x/de-en/100k/data/4_encoded/unigram/4000/train.de"
        assert lay.pretok_flag is False


class TestByteFallback:
    """byte_fallback is an orthogonal flag (not a subword model variant). It
    must show up in on-disk paths so vocabs with/without fallback don't collide."""

    def test_path_segment_gets_bytes_suffix(self):
        ds = Dataset(
            base_path="/x", parent_ds=False, dataset_name="d", dataset_lang_pair="es-en",
            dataset_size_name="s", dataset_lines=None, splits_sizes=(None, 10, 10),
            subword_model="unigram", vocab_size=8000, byte_fallback=True, merge_vocabs=False,
        )
        assert ds.byte_fallback is True
        assert ds.vocab_size_id() == ("unigram+bytes", "8000")
        assert ds.get_encoded_path("train.es") == "/x/d/es-en/s/data/4_encoded/unigram+bytes/8000/train.es"
        assert ds.get_vocab_file(lang="es") == "/x/d/es-en/s/vocabs/unigram+bytes/8000/es"

    def test_default_off(self):
        ds = Dataset(
            base_path="/x", parent_ds=False, dataset_name="d", dataset_lang_pair="es-en",
            dataset_size_name="s", dataset_lines=None, splits_sizes=(None, 10, 10),
            subword_model="bpe", vocab_size=8000, merge_vocabs=False,
        )
        assert ds.byte_fallback is False
        assert ds.vocab_size_id() == ("bpe", "8000")

    def test_run_name_distinguishes_fallback(self):
        common = dict(base_path="/x", parent_ds=False, dataset_name="d",
                      dataset_lang_pair="es-en", dataset_size_name="s",
                      dataset_lines=None, splits_sizes=(None, 10, 10),
                      subword_model="bpe", vocab_size=8000, merge_vocabs=False)
        plain = Dataset(**common)
        fb = Dataset(**common, byte_fallback=True)
        assert plain.get_run_name("m") != fb.get_run_name("m")

    def test_plus_bytes_string_sugar(self):
        # "bpe+bytes" as subword_model is sugar for (subword_model="bpe", byte_fallback=True).
        ds = Dataset(
            base_path="/x", parent_ds=False, dataset_name="d", dataset_lang_pair="es-en",
            dataset_size_name="s", dataset_lines=None, splits_sizes=(None, 10, 10),
            subword_model="bpe+bytes", vocab_size=8000, merge_vocabs=False,
        )
        assert ds.subword_model == "bpe"
        assert ds.byte_fallback is True
        assert ds.vocab_size_id() == ("bpe+bytes", "8000")

    def test_byte_fallback_rejected_for_bytes_only(self):
        # bytes-only is its own scheme; byte_fallback would be meaningless.
        with pytest.raises(ValueError, match="byte_fallback=True is incompatible"):
            Dataset(
                base_path="/x", parent_ds=False, dataset_name="d", dataset_lang_pair="es-en",
                dataset_size_name="s", dataset_lines=None, splits_sizes=(None, 10, 10),
                subword_model="bytes", vocab_size=None, byte_fallback=True, merge_vocabs=False,
            )


class TestBackwardsCompatAttributes:
    """Older user code reads these directly."""

    def test_base_path_attr(self, ds_bpe):
        assert ds_bpe.base_path == "/tmp/x"

    def test_data_raw_path_attr(self, ds_bpe):
        assert ds_bpe.data_raw_path == "data/0_raw"

    def test_pretok_flag_attr(self, ds_bpe):
        assert ds_bpe.pretok_flag is False
