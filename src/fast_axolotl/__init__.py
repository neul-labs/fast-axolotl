"""
Fast-Axolotl: High-performance Rust extensions for Axolotl.

This package provides drop-in acceleration for existing Axolotl installations
by shimming Rust-based streaming dataset loading into the axolotl namespace.

Usage:
    # Simply import fast_axolotl before using axolotl
    import fast_axolotl

    # Or explicitly install the shim
    fast_axolotl.install()

    # Now axolotl will use the fast Rust-based dataset loading
    import axolotl
"""

__version__ = "0.1.0"

import logging
import sys
from typing import Iterator, Dict, Any, Optional

LOG = logging.getLogger(__name__)

# Track if shim is installed
_SHIM_INSTALLED = False

# Try to import the Rust extension
try:
    from fast_axolotl._rust_ext import streaming_dataset_reader as _rust_streaming_reader
    from fast_axolotl._rust_ext import get_version as _get_rust_version
    RUST_AVAILABLE = True
except ImportError as e:
    LOG.warning(f"Fast-axolotl Rust extension not available: {e}")
    RUST_AVAILABLE = False
    _rust_streaming_reader = None
    _get_rust_version = None


def is_available() -> bool:
    """Check if the Rust extension is available."""
    return RUST_AVAILABLE


def get_version() -> str:
    """Get the fast-axolotl version."""
    if RUST_AVAILABLE and _get_rust_version:
        return f"{__version__} (rust: {_get_rust_version()})"
    return f"{__version__} (rust: not available)"


def streaming_dataset_reader(
    file_path: str,
    dataset_type: str,
    batch_size: int = 1000,
    num_threads: int = 4
) -> Iterator[Dict[str, Any]]:
    """
    Stream data from a dataset file using the Rust extension.

    Args:
        file_path: Path to the dataset file
        dataset_type: Type of dataset ('parquet', 'arrow', 'csv', 'json', 'text')
        batch_size: Number of rows per batch
        num_threads: Number of threads for processing

    Yields:
        Dictionary containing column data for each batch
    """
    if not RUST_AVAILABLE:
        raise ImportError("Rust extension is not available")

    batches = _rust_streaming_reader(file_path, dataset_type, batch_size, num_threads)
    for batch in batches:
        yield batch


class RustStreamingDataset:
    """HuggingFace Dataset-compatible wrapper for Rust-based streaming."""

    def __init__(
        self,
        file_path: str,
        dataset_type: str,
        batch_size: int = 1000,
        num_threads: int = 4,
        dataset_keep_in_memory: bool = False,
    ):
        if not RUST_AVAILABLE:
            raise ImportError("Rust extension is not available")

        self.file_path = file_path
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.dataset_keep_in_memory = dataset_keep_in_memory

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        yield from streaming_dataset_reader(
            self.file_path,
            self.dataset_type,
            self.batch_size,
            self.num_threads
        )

    def with_format(self, format: str):
        """Return self for HuggingFace compatibility."""
        return self


def create_rust_streaming_dataset(
    file_path: str,
    dataset_type: str,
    batch_size: int = 1000,
    num_threads: int = 4,
    dataset_keep_in_memory: bool = False,
) -> RustStreamingDataset:
    """Create a HuggingFace-compatible streaming dataset using Rust."""
    return RustStreamingDataset(
        file_path, dataset_type, batch_size, num_threads, dataset_keep_in_memory
    )


def should_use_rust_streaming(
    file_path: str,
    dataset_config: Dict[str, Any],
    cfg: Dict[str, Any],
) -> bool:
    """Determine if Rust streaming should be used for this dataset."""
    if not RUST_AVAILABLE or not cfg.get('dataset_use_rust_streaming', False):
        return False

    # Only use when dataset_keep_in_memory is False
    if cfg.get('dataset_keep_in_memory', False):
        return False

    # Check file size (> 1GB)
    try:
        import os
        file_size = os.path.getsize(file_path)
        if file_size < 1024 * 1024 * 1024:  # 1GB
            return False
    except OSError:
        pass

    # Check sequence length (> 10K)
    if cfg.get('sequence_len', 0) < 10000:
        return False

    return True


def install() -> bool:
    """
    Install the fast-axolotl shim into the axolotl namespace.

    This patches axolotl's data loading utilities to use Rust-based
    streaming when available and beneficial.

    Returns:
        True if shim was installed, False if already installed or failed
    """
    global _SHIM_INSTALLED

    if _SHIM_INSTALLED:
        LOG.debug("Fast-axolotl shim already installed")
        return False

    if not RUST_AVAILABLE:
        LOG.warning("Cannot install fast-axolotl shim: Rust extension not available")
        return False

    try:
        # Create shim modules that will be injected into axolotl's namespace
        _install_rust_ext_shim()
        _install_rust_streaming_shim()
        _install_rust_wrapper_shim()

        _SHIM_INSTALLED = True
        LOG.info("Fast-axolotl shim installed successfully")
        return True

    except Exception as e:
        LOG.error(f"Failed to install fast-axolotl shim: {e}")
        return False


def _install_rust_ext_shim():
    """Install shim for axolotl.rust_ext module."""
    import types

    # Create axolotl.rust_ext module if it doesn't exist
    if 'axolotl' not in sys.modules:
        axolotl_mod = types.ModuleType('axolotl')
        sys.modules['axolotl'] = axolotl_mod
    else:
        axolotl_mod = sys.modules['axolotl']

    if 'axolotl.rust_ext' not in sys.modules:
        rust_ext_mod = types.ModuleType('axolotl.rust_ext')
        sys.modules['axolotl.rust_ext'] = rust_ext_mod
        if hasattr(axolotl_mod, '__path__'):
            pass  # Module has path, ok
    else:
        rust_ext_mod = sys.modules['axolotl.rust_ext']

    # Create axolotl_ext submodule with streaming_dataset_reader
    if 'axolotl.rust_ext.axolotl_ext' not in sys.modules:
        axolotl_ext_mod = types.ModuleType('axolotl.rust_ext.axolotl_ext')
        axolotl_ext_mod.streaming_dataset_reader = _rust_streaming_reader
        sys.modules['axolotl.rust_ext.axolotl_ext'] = axolotl_ext_mod
        rust_ext_mod.axolotl_ext = axolotl_ext_mod


def _install_rust_streaming_shim():
    """Install shim for axolotl.utils.data.rust_streaming module."""
    import types

    # Ensure parent modules exist
    if 'axolotl.utils' not in sys.modules:
        utils_mod = types.ModuleType('axolotl.utils')
        sys.modules['axolotl.utils'] = utils_mod

    if 'axolotl.utils.data' not in sys.modules:
        data_mod = types.ModuleType('axolotl.utils.data')
        sys.modules['axolotl.utils.data'] = data_mod

    # Create rust_streaming module
    if 'axolotl.utils.data.rust_streaming' not in sys.modules:
        rust_streaming_mod = types.ModuleType('axolotl.utils.data.rust_streaming')
        rust_streaming_mod.get_rust_extension_status = is_available
        rust_streaming_mod.streaming_dataset_reader = streaming_dataset_reader
        rust_streaming_mod.RUST_EXTENSION_AVAILABLE = RUST_AVAILABLE
        sys.modules['axolotl.utils.data.rust_streaming'] = rust_streaming_mod


def _install_rust_wrapper_shim():
    """Install shim for axolotl.utils.data.rust_wrapper module."""
    import types

    # Ensure parent modules exist
    if 'axolotl.utils.data' not in sys.modules:
        data_mod = types.ModuleType('axolotl.utils.data')
        sys.modules['axolotl.utils.data'] = data_mod

    # Create rust_wrapper module
    if 'axolotl.utils.data.rust_wrapper' not in sys.modules:
        rust_wrapper_mod = types.ModuleType('axolotl.utils.data.rust_wrapper')
        rust_wrapper_mod.is_rust_streaming_available = is_available
        rust_wrapper_mod.load_dataset_with_rust_streaming = streaming_dataset_reader
        rust_wrapper_mod.RustStreamingDataset = RustStreamingDataset
        rust_wrapper_mod.create_rust_streaming_dataset = create_rust_streaming_dataset
        rust_wrapper_mod.should_use_rust_streaming = should_use_rust_streaming
        sys.modules['axolotl.utils.data.rust_wrapper'] = rust_wrapper_mod


def uninstall() -> bool:
    """
    Remove the fast-axolotl shim from the axolotl namespace.

    Returns:
        True if shim was removed, False if not installed
    """
    global _SHIM_INSTALLED

    if not _SHIM_INSTALLED:
        return False

    # Remove shimmed modules
    modules_to_remove = [
        'axolotl.rust_ext.axolotl_ext',
        'axolotl.utils.data.rust_streaming',
        'axolotl.utils.data.rust_wrapper',
    ]

    for mod_name in modules_to_remove:
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    _SHIM_INSTALLED = False
    LOG.info("Fast-axolotl shim uninstalled")
    return True


# Auto-install on import if Rust extension is available
if RUST_AVAILABLE:
    install()
