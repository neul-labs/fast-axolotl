"""Tests for fast-axolotl."""

import pytest


def test_import():
    """Test that fast_axolotl can be imported."""
    import fast_axolotl

    assert hasattr(fast_axolotl, "__version__")
    assert hasattr(fast_axolotl, "is_available")
    assert hasattr(fast_axolotl, "install")
    assert hasattr(fast_axolotl, "uninstall")


def test_version():
    """Test version string."""
    import fast_axolotl

    version = fast_axolotl.get_version()
    # Version should contain the package version
    import fast_axolotl as fa
    assert fa.__version__ in version


def test_is_available():
    """Test is_available function."""
    import fast_axolotl

    result = fast_axolotl.is_available()
    assert isinstance(result, bool)


@pytest.mark.skipif(
    not pytest.importorskip("fast_axolotl").is_available(),
    reason="Rust extension not available",
)
class TestRustExtension:
    """Tests that require the Rust extension."""

    def test_streaming_reader_validation(self):
        """Test parameter validation in streaming reader."""
        from fast_axolotl import streaming_dataset_reader

        # Empty file_path should raise ValueError
        with pytest.raises(ValueError):
            list(streaming_dataset_reader("", "parquet"))

        # Empty dataset_type is valid (auto-detect), but non-existent file raises RuntimeError
        with pytest.raises(RuntimeError):
            list(streaming_dataset_reader("/tmp/nonexistent.parquet", ""))

    def test_rust_streaming_dataset_init(self):
        """Test RustStreamingDataset initialization."""
        from fast_axolotl import RustStreamingDataset

        dataset = RustStreamingDataset("/tmp/test.parquet", "parquet")
        assert dataset.file_path == "/tmp/test.parquet"
        assert dataset.dataset_type == "parquet"

        with pytest.raises(ValueError):
            dataset_empty = RustStreamingDataset("", "parquet")
            list(dataset_empty)

    def test_create_rust_streaming_dataset(self):
        """Test create_rust_streaming_dataset factory."""
        from fast_axolotl import create_rust_streaming_dataset

        dataset = create_rust_streaming_dataset(
            "/tmp/test.parquet", "parquet", batch_size=100
        )
        assert dataset.file_path == "/tmp/test.parquet"
        assert dataset.dataset_type == "parquet"
        assert dataset.batch_size == 100


@pytest.mark.skipif(
    not pytest.importorskip("fast_axolotl").is_available(),
    reason="Rust extension not available",
)
class TestTokenPacking:
    """Tests for token packing acceleration."""

    def test_pack_sequences_basic(self):
        """Test basic sequence packing."""
        from fast_axolotl import pack_sequences

        sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        result = pack_sequences(
            sequences, max_length=10, pad_token_id=0, eos_token_id=2, label_pad_id=-100
        )

        assert "input_ids" in result
        assert "labels" in result
        assert "attention_mask" in result
        assert all(len(seq) == 10 for seq in result["input_ids"])

    def test_concatenate_and_pack(self):
        """Test concatenate_and_pack function."""
        from fast_axolotl import concatenate_and_pack

        input_ids = [[1, 2, 3], [4, 5, 6]]
        labels = [[1, 2, 3], [4, 5, 6]]
        attention_masks = [[1, 1, 1], [1, 1, 1]]

        result = concatenate_and_pack(
            input_ids,
            labels,
            attention_masks,
            max_length=10,
            pad_token_id=0,
            label_pad_id=-100,
        )

        assert "input_ids" in result
        assert all(len(seq) == 10 for seq in result["input_ids"])


@pytest.mark.skipif(
    not pytest.importorskip("fast_axolotl").is_available(),
    reason="Rust extension not available",
)
class TestParallelHashing:
    """Tests for parallel hashing acceleration."""

    def test_parallel_hash_rows(self):
        """Test parallel hashing."""
        from fast_axolotl import parallel_hash_rows

        rows = ["row1", "row2", "row3", "row1"]  # row1 repeated
        hashes = parallel_hash_rows(rows, num_threads=2)

        assert len(hashes) == 4
        assert hashes[0] == hashes[3]  # Same input = same hash
        assert hashes[0] != hashes[1]  # Different input = different hash
        assert all(len(h) == 64 for h in hashes)  # SHA256 = 64 hex chars

    def test_deduplicate_indices(self):
        """Test deduplication with parallel hashing."""
        from fast_axolotl import deduplicate_indices

        rows = ["a", "b", "a", "c", "b"]
        unique_indices, new_hashes = deduplicate_indices(rows, num_threads=2)

        assert unique_indices == [0, 1, 3]  # a, b, c (first occurrences)
        assert len(new_hashes) == 3

    def test_deduplicate_with_existing(self):
        """Test deduplication with existing hashes."""
        from fast_axolotl import deduplicate_indices

        # First batch
        rows1 = ["a", "b"]
        _, hashes1 = deduplicate_indices(rows1)

        # Second batch with existing hashes
        rows2 = ["b", "c", "a", "d"]
        unique_indices, _ = deduplicate_indices(rows2, existing_hashes=hashes1)

        # b and a already seen, so only c and d are unique
        assert unique_indices == [1, 3]


@pytest.mark.skipif(
    not pytest.importorskip("fast_axolotl").is_available(),
    reason="Rust extension not available",
)
class TestBatchPadding:
    """Tests for batch padding acceleration."""

    def test_pad_sequences_right(self):
        """Test right padding."""
        from fast_axolotl import pad_sequences

        sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]]
        padded = pad_sequences(sequences, pad_value=0, padding_side="right")

        assert all(len(seq) == 5 for seq in padded)
        assert padded[0] == [1, 2, 3, 0, 0]
        assert padded[1] == [4, 5, 0, 0, 0]
        assert padded[2] == [6, 7, 8, 9, 10]

    def test_pad_sequences_left(self):
        """Test left padding."""
        from fast_axolotl import pad_sequences

        sequences = [[1, 2, 3], [4, 5]]
        padded = pad_sequences(sequences, pad_value=0, padding_side="left")

        assert padded[0] == [1, 2, 3]
        assert padded[1] == [0, 4, 5]

    def test_pad_to_multiple_of(self):
        """Test padding to multiple of a value."""
        from fast_axolotl import pad_sequences

        sequences = [[1, 2, 3]]  # Length 3
        padded = pad_sequences(sequences, pad_value=0, pad_to_multiple_of=8)

        assert len(padded[0]) == 8  # Rounded up to 8

    def test_create_padding_mask(self):
        """Test padding mask creation."""
        from fast_axolotl import create_padding_mask

        mask = create_padding_mask(3, 8)
        assert mask == [0, 1, 2, 3, 4]  # 5 positions to pad

        mask_empty = create_padding_mask(8, 8)
        assert mask_empty == []


class TestShim:
    """Tests for the axolotl shim functionality."""

    def test_install_uninstall(self):
        """Test shim install/uninstall."""
        import fast_axolotl

        if fast_axolotl.is_available():
            fast_axolotl.uninstall()
            result = fast_axolotl.install()
            assert result is True
            result = fast_axolotl.install()
            assert result is False

    def test_shim_creates_modules(self):
        """Test that shim creates expected modules."""
        import sys
        import fast_axolotl

        if fast_axolotl.is_available():
            fast_axolotl.install()
            assert "axolotl.rust_ext.axolotl_ext" in sys.modules
            assert "axolotl.utils.data.rust_streaming" in sys.modules
            assert "axolotl.utils.data.rust_wrapper" in sys.modules

    def test_all_exports(self):
        """Test that all expected functions are exported."""
        import fast_axolotl

        expected = [
            "is_available",
            "get_version",
            "install",
            "uninstall",
            "list_supported_formats",
            "detect_format",
            "streaming_dataset_reader",
            "RustStreamingDataset",
            "pack_sequences",
            "concatenate_and_pack",
            "parallel_hash_rows",
            "deduplicate_indices",
            "pad_sequences",
            "create_padding_mask",
        ]

        for name in expected:
            assert hasattr(fast_axolotl, name), f"Missing export: {name}"


@pytest.mark.skipif(
    not pytest.importorskip("fast_axolotl").is_available(),
    reason="Rust extension not available",
)
class TestFormatDetection:
    """Tests for format detection functionality."""

    def test_list_supported_formats(self):
        """Test listing supported formats."""
        from fast_axolotl import list_supported_formats

        formats = list_supported_formats()
        assert isinstance(formats, list)
        assert len(formats) > 0

        # Check base formats
        assert "parquet" in formats
        assert "arrow" in formats
        assert "feather" in formats
        assert "csv" in formats
        assert "json" in formats
        assert "jsonl" in formats
        assert "text" in formats

        # Check compressed formats
        assert "parquet.zst" in formats
        assert "json.gz" in formats
        assert "jsonl.zst" in formats

        # Check directory format
        assert "hf_dataset" in formats

    def test_detect_format_parquet(self):
        """Test format detection for parquet files."""
        from fast_axolotl import detect_format

        base, compression = detect_format("/path/to/data.parquet")
        assert base == "parquet"
        assert compression is None

    def test_detect_format_compressed_zstd(self):
        """Test format detection for ZSTD compressed files."""
        from fast_axolotl import detect_format

        base, compression = detect_format("/path/to/data.jsonl.zst")
        assert base == "jsonl"
        assert compression == "zstd"

        base, compression = detect_format("/path/to/data.parquet.zst")
        assert base == "parquet"
        assert compression == "zstd"

    def test_detect_format_compressed_gzip(self):
        """Test format detection for Gzip compressed files."""
        from fast_axolotl import detect_format

        base, compression = detect_format("/path/to/data.json.gz")
        assert base == "json"
        assert compression == "gzip"

        base, compression = detect_format("/path/to/data.csv.gz")
        assert base == "csv"
        assert compression == "gzip"

    def test_detect_format_arrow(self):
        """Test format detection for arrow/feather files."""
        from fast_axolotl import detect_format

        base, compression = detect_format("/path/to/data.arrow")
        assert base == "arrow"
        assert compression is None

        base, compression = detect_format("/path/to/data.feather")
        assert base == "feather"
        assert compression is None

    def test_detect_format_text(self):
        """Test format detection for text files."""
        from fast_axolotl import detect_format

        base, compression = detect_format("/path/to/data.txt")
        assert base == "text"
        assert compression is None

        base, compression = detect_format("/path/to/data.txt.gz")
        assert base == "text"
        assert compression == "gzip"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_detect_format_no_extension(self):
        """Test format detection for files without extension."""
        from fast_axolotl import detect_format

        # File without recognized extension - returns "json" as fallback
        base, compression = detect_format("/path/to/Makefile")
        assert base == "json"
        assert compression is None

    def test_pad_sequences_empty(self):
        """Test pad_sequences with empty input."""
        from fast_axolotl import pad_sequences

        result = pad_sequences([])
        assert result == []

    def test_pad_sequences_invalid_padding_side(self):
        """Test pad_sequences with invalid padding_side raises error."""
        from fast_axolotl import pad_sequences
        import pytest

        with pytest.raises(ValueError, match="padding_side must be 'right' or 'left'"):
            pad_sequences([[1, 2, 3]], padding_side="invalid")

    def test_pad_sequences_zero_target_length(self):
        """Test pad_sequences with target_length=None uses max sequence length."""
        from fast_axolotl import pad_sequences

        result = pad_sequences([[1, 2, 3], [1, 2]], target_length=None)
        assert result == [[1, 2, 3], [1, 2, 0]]

    def test_pack_sequences_empty(self):
        """Test pack_sequences with empty input."""
        from fast_axolotl import pack_sequences

        result = pack_sequences([], max_length=10, pad_token_id=0, eos_token_id=2)
        assert result == {"input_ids": [], "labels": [], "attention_mask": []}

    def test_streaming_reader_empty_path(self):
        """Test streaming_dataset_reader with empty path raises error."""
        from fast_axolotl import streaming_dataset_reader
        import pytest

        with pytest.raises(ValueError):
            list(streaming_dataset_reader("", "parquet"))

    def test_parallel_hash_rows_empty(self):
        """Test parallel_hash_rows with empty input."""
        from fast_axolotl import parallel_hash_rows

        result = parallel_hash_rows([])
        assert result == []


class TestErrorPaths:
    """Tests for error handling paths."""

    def test_detect_format_compressed_no_base_extension(self):
        """Test format detection for files without recognized extension."""
        from fast_axolotl import detect_format

        # File without recognized extension falls back to "json"
        base, compression = detect_format("/path/to/datazst")
        assert base == "json"
        assert compression is None

    def test_pad_sequences_to_multiple_of_zero(self):
        """Test pad_sequences with pad_to_multiple_of=0 is a no-op."""
        from fast_axolotl import pad_sequences

        result = pad_sequences([[1, 2, 3]], pad_to_multiple_of=0)
        assert result == [[1, 2, 3]]

    def test_pack_sequences_max_length_zero(self):
        """Test pack_sequences with larger sequences."""
        from fast_axolotl import pack_sequences

        # Using reasonable max_length that fits sequences
        result = pack_sequences([[1, 2, 3], [4, 5, 6, 7]], max_length=10, pad_token_id=0, eos_token_id=2)
        # Should pack sequences efficiently
        assert len(result["input_ids"]) > 0
        assert 2 in result["input_ids"][0]  # Contains EOS token

    def test_concatenate_and_pack_empty(self):
        """Test concatenate_and_pack with empty input."""
        from fast_axolotl import concatenate_and_pack

        result = concatenate_and_pack([], [], [], max_length=10, pad_token_id=0, label_pad_id=-100)
        assert result == {"input_ids": [], "labels": [], "attention_mask": []}

    def test_create_padding_mask_zero_length(self):
        """Test create_padding_mask when current >= target."""
        from fast_axolotl import create_padding_mask

        result = create_padding_mask(10, 5)
        assert result == []  # No padding needed

    def test_parallel_hash_rows_single_thread(self):
        """Test parallel_hash_rows with single row."""
        from fast_axolotl import parallel_hash_rows

        result = parallel_hash_rows(["single row"])
        assert len(result) == 1
        assert len(result[0]) == 64  # SHA256 hex

    def test_deduplicate_indices_empty(self):
        """Test deduplicate_indices with empty input."""
        from fast_axolotl import deduplicate_indices

        indices, hashes = deduplicate_indices([])
        assert indices == []
        assert hashes == []

    def test_deduplicate_indices_with_existing(self):
        """Test deduplicate_indices with pre-existing hashes."""
        from fast_axolotl import deduplicate_indices

        # "a" at index 0 is NOT in existing_hashes, so it's kept
        # "b" at index 1 is NOT in existing_hashes, so it's kept
        # "a" at index 2 IS a duplicate of index 0, so it's skipped
        indices, hashes = deduplicate_indices(["a", "b", "a"], existing_hashes=["a"])
        assert indices == [0, 1]  # "a" at 0 is new (not same hash object), "b" at 1 is new, "a" at 2 is duplicate
        assert len(hashes) == 2

