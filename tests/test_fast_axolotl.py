"""Tests for fast-axolotl."""

import pytest


def test_import():
    """Test that fast_axolotl can be imported."""
    import fast_axolotl
    assert hasattr(fast_axolotl, '__version__')
    assert hasattr(fast_axolotl, 'is_available')
    assert hasattr(fast_axolotl, 'install')
    assert hasattr(fast_axolotl, 'uninstall')


def test_version():
    """Test version string."""
    import fast_axolotl
    version = fast_axolotl.get_version()
    assert '0.1.0' in version


def test_is_available():
    """Test is_available function."""
    import fast_axolotl
    # Should return bool
    result = fast_axolotl.is_available()
    assert isinstance(result, bool)


@pytest.mark.skipif(
    not pytest.importorskip("fast_axolotl").is_available(),
    reason="Rust extension not available"
)
class TestRustExtension:
    """Tests that require the Rust extension."""

    def test_streaming_reader_validation(self):
        """Test parameter validation in streaming reader."""
        from fast_axolotl import streaming_dataset_reader

        with pytest.raises(Exception):
            # Empty file path should raise
            list(streaming_dataset_reader("", "parquet"))

        with pytest.raises(Exception):
            # Empty dataset type should raise
            list(streaming_dataset_reader("/tmp/test.parquet", ""))

    def test_rust_streaming_dataset_init(self):
        """Test RustStreamingDataset initialization."""
        from fast_axolotl import RustStreamingDataset

        # Should raise for empty file path
        with pytest.raises(Exception):
            RustStreamingDataset("", "parquet")

    def test_create_rust_streaming_dataset(self):
        """Test create_rust_streaming_dataset factory."""
        from fast_axolotl import create_rust_streaming_dataset

        dataset = create_rust_streaming_dataset(
            "/tmp/test.parquet",
            "parquet",
            batch_size=100
        )
        assert dataset.file_path == "/tmp/test.parquet"
        assert dataset.dataset_type == "parquet"
        assert dataset.batch_size == 100


class TestShim:
    """Tests for the axolotl shim functionality."""

    def test_install_uninstall(self):
        """Test shim install/uninstall."""
        import fast_axolotl

        # If Rust is available, shim should auto-install
        if fast_axolotl.is_available():
            # Uninstall first
            fast_axolotl.uninstall()

            # Should be able to reinstall
            result = fast_axolotl.install()
            assert result is True

            # Second install should return False (already installed)
            result = fast_axolotl.install()
            assert result is False

    def test_shim_creates_modules(self):
        """Test that shim creates expected modules."""
        import sys
        import fast_axolotl

        if fast_axolotl.is_available():
            fast_axolotl.install()

            # Check shimmed modules exist
            assert 'axolotl.rust_ext.axolotl_ext' in sys.modules
            assert 'axolotl.utils.data.rust_streaming' in sys.modules
            assert 'axolotl.utils.data.rust_wrapper' in sys.modules
