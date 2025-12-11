"""
Fast-Axolotl: High-performance Rust extensions for Axolotl.

This package provides drop-in acceleration for existing Axolotl installations
by shimming Rust-based optimizations into the axolotl namespace.

## Accelerations Provided

1. **Streaming Dataset Loading** - Memory-efficient streaming for large datasets
2. **Token Packing** - Fast sequence concatenation without torch.cat() loops
3. **Parallel Hashing** - Multi-threaded SHA256 for deduplication
4. **Batch Padding** - Efficient sequence padding without list loops

## Usage

```python
# Simply import fast_axolotl before using axolotl
import fast_axolotl

# Or explicitly install the shim
fast_axolotl.install()

# Now axolotl will use fast Rust-based implementations
import axolotl
```
"""

__version__ = "0.2.0"

import logging
import sys
from typing import Iterator, Dict, Any, Optional, List, Tuple

LOG = logging.getLogger(__name__)

# Track if shim is installed
_SHIM_INSTALLED = False

# Try to import the Rust extension
try:
    from fast_axolotl._rust_ext import (
        # Streaming
        streaming_dataset_reader as _rust_streaming_reader,
        get_version as _get_rust_version,
        # Format detection
        detect_format as _rust_detect_format,
        list_supported_formats as _rust_list_supported_formats,
        # Token Packing (Acceleration #1)
        pack_sequences as _rust_pack_sequences,
        concatenate_and_pack as _rust_concatenate_and_pack,
        # Parallel Hashing (Acceleration #2)
        parallel_hash_rows as _rust_parallel_hash_rows,
        deduplicate_indices as _rust_deduplicate_indices,
        # Batch Padding (Acceleration #3)
        pad_sequences as _rust_pad_sequences,
        create_padding_mask as _rust_create_padding_mask,
    )

    RUST_AVAILABLE = True
except ImportError as e:
    LOG.warning(f"Fast-axolotl Rust extension not available: {e}")
    RUST_AVAILABLE = False
    _rust_streaming_reader = None
    _get_rust_version = None
    _rust_detect_format = None
    _rust_list_supported_formats = None
    _rust_pack_sequences = None
    _rust_concatenate_and_pack = None
    _rust_parallel_hash_rows = None
    _rust_deduplicate_indices = None
    _rust_pad_sequences = None
    _rust_create_padding_mask = None


def is_available() -> bool:
    """Check if the Rust extension is available."""
    return RUST_AVAILABLE


def get_version() -> str:
    """Get the fast-axolotl version."""
    if RUST_AVAILABLE and _get_rust_version:
        return f"{__version__} (rust: {_get_rust_version()})"
    return f"{__version__} (rust: not available)"


def list_supported_formats() -> List[str]:
    """
    List all supported file formats.

    Returns:
        List of format strings including:
        - Base formats: parquet, arrow, feather, csv, json, jsonl, text
        - Compressed: parquet.zst, json.gz, etc.
        - Directory: hf_dataset (HuggingFace Arrow Dataset)
    """
    if not RUST_AVAILABLE:
        return [
            "parquet",
            "arrow",
            "feather",
            "csv",
            "json",
            "jsonl",
            "text",
            "parquet.zst",
            "parquet.gz",
            "arrow.zst",
            "arrow.gz",
            "json.zst",
            "json.gz",
            "jsonl.zst",
            "jsonl.gz",
            "csv.zst",
            "csv.gz",
            "text.zst",
            "text.gz",
            "hf_dataset",
        ]
    return _rust_list_supported_formats()


def detect_format(file_path: str) -> Tuple[str, Optional[str]]:
    """
    Auto-detect file format and compression from path.

    Args:
        file_path: Path to the file or directory

    Returns:
        Tuple of (base_format, compression)
        Examples:
        - "data.parquet" -> ("parquet", None)
        - "data.jsonl.zst" -> ("jsonl", "zstd")
        - "data.csv.gz" -> ("csv", "gzip")
        - "/path/to/hf_dataset/" -> ("hf_dataset", None)
    """
    if not RUST_AVAILABLE:
        raise ImportError("Rust extension is not available")
    return _rust_detect_format(file_path)


# =============================================================================
# ACCELERATION #1: Token Packing
# =============================================================================


def pack_sequences(
    sequences: List[List[int]],
    max_length: int,
    pad_token_id: int,
    eos_token_id: int,
    label_pad_id: int = -100,
) -> Dict[str, List[List[int]]]:
    """
    Pack multiple sequences into fixed-length chunks efficiently.

    This replaces the inefficient torch.cat() loops in encode_streaming.
    Pre-allocates buffers and fills them in Rust for maximum performance.

    Args:
        sequences: List of token ID lists to pack
        max_length: Maximum length for each packed sequence
        pad_token_id: Token ID to use for padding
        eos_token_id: End-of-sequence token ID
        label_pad_id: Padding value for labels (typically -100)

    Returns:
        Dict with 'input_ids', 'labels', 'attention_mask' as lists of lists
    """
    if not RUST_AVAILABLE:
        raise ImportError("Rust extension is not available")
    return _rust_pack_sequences(
        sequences, max_length, pad_token_id, eos_token_id, label_pad_id
    )


def concatenate_and_pack(
    input_ids: List[List[int]],
    labels: List[List[int]],
    attention_masks: List[List[int]],
    max_length: int,
    pad_token_id: int,
    label_pad_id: int = -100,
) -> Dict[str, List[List[int]]]:
    """
    Concatenate sequences and pack into max_length chunks.

    Lower-level function for direct token list manipulation when you
    already have separate input_ids, labels, and attention_mask lists.

    Args:
        input_ids: List of input token ID lists
        labels: List of label token ID lists
        attention_masks: List of attention mask lists
        max_length: Maximum length for each packed sequence
        pad_token_id: Token ID for padding input_ids
        label_pad_id: Value for padding labels (typically -100)

    Returns:
        Dict with 'input_ids', 'labels', 'attention_mask' as lists of lists
    """
    if not RUST_AVAILABLE:
        raise ImportError("Rust extension is not available")
    return _rust_concatenate_and_pack(
        input_ids, labels, attention_masks, max_length, pad_token_id, label_pad_id
    )


# =============================================================================
# ACCELERATION #2: Parallel Hashing for Deduplication
# =============================================================================


def parallel_hash_rows(
    rows: List[str],
    num_threads: int = 0,
) -> List[str]:
    """
    Compute SHA256 hashes for multiple rows in parallel.

    This replaces the sequential sha256(str(row)) loop in _deduplicate_dataset.
    Uses multiple threads to hash rows concurrently.

    Args:
        rows: List of string representations of rows
        num_threads: Number of threads to use (0 = auto-detect)

    Returns:
        List of hex-encoded SHA256 hashes in same order as input
    """
    if not RUST_AVAILABLE:
        raise ImportError("Rust extension is not available")
    return _rust_parallel_hash_rows(rows, num_threads)


def deduplicate_indices(
    rows: List[str],
    existing_hashes: Optional[List[str]] = None,
    num_threads: int = 0,
) -> Tuple[List[int], List[str]]:
    """
    Find indices of unique rows using parallel hashing.

    Combines parallel hashing with deduplication logic for maximum efficiency.
    Can optionally filter against a set of existing hashes.

    Args:
        rows: List of string representations of rows
        existing_hashes: Optional list of already-seen hashes to filter against
        num_threads: Number of threads (0 = auto-detect)

    Returns:
        Tuple of (unique_indices, new_hashes)
    """
    if not RUST_AVAILABLE:
        raise ImportError("Rust extension is not available")
    return _rust_deduplicate_indices(rows, existing_hashes, num_threads)


# =============================================================================
# ACCELERATION #3: Batch Padding
# =============================================================================


def pad_sequences(
    sequences: List[List[int]],
    target_length: Optional[int] = None,
    pad_value: int = 0,
    padding_side: str = "right",
    pad_to_multiple_of: Optional[int] = None,
) -> List[List[int]]:
    """
    Pad a batch of sequences to a target length efficiently.

    This replaces the list creation loops in batching.py collators.
    Uses Rust for efficient memory allocation and padding.

    Args:
        sequences: List of token ID lists
        target_length: Length to pad to (None = max in batch)
        pad_value: Value to use for padding (default: 0)
        padding_side: "right" or "left"
        pad_to_multiple_of: Optionally pad to multiple of this value

    Returns:
        Padded sequences as list of lists
    """
    if not RUST_AVAILABLE:
        raise ImportError("Rust extension is not available")
    return _rust_pad_sequences(
        sequences, target_length, pad_value, padding_side, pad_to_multiple_of
    )


def create_padding_mask(
    current_length: int,
    target_length: int,
) -> List[int]:
    """
    Create position IDs for padding efficiently.

    Replaces list(range(remainder_len)) pattern in collators.

    Args:
        current_length: Current sequence length
        target_length: Target length after padding

    Returns:
        List of position IDs for padding (0, 1, 2, ...)
    """
    if not RUST_AVAILABLE:
        raise ImportError("Rust extension is not available")
    return _rust_create_padding_mask(current_length, target_length)


# =============================================================================
# Streaming (Original)
# =============================================================================


def streaming_dataset_reader(
    file_path: str, dataset_type: str, batch_size: int = 1000, num_threads: int = 4
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
            self.file_path, self.dataset_type, self.batch_size, self.num_threads
        )

    def with_format(self, format: str):
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
    if not RUST_AVAILABLE or not cfg.get("dataset_use_rust_streaming", False):
        return False
    if cfg.get("dataset_keep_in_memory", False):
        return False
    try:
        import os

        file_size = os.path.getsize(file_path)
        if file_size < 1024 * 1024 * 1024:
            return False
    except OSError:
        pass
    if cfg.get("sequence_len", 0) < 10000:
        return False
    return True


# =============================================================================
# Auto-Shimming
# =============================================================================


def install() -> bool:
    """
    Install the fast-axolotl shim into the axolotl namespace.

    This patches axolotl's modules to use Rust-based implementations:
    - axolotl.utils.data.streaming -> Token packing acceleration
    - axolotl.utils.data.utils -> Parallel hashing for deduplication
    - axolotl.utils.collators.batching -> Fast batch padding

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
        _install_rust_ext_shim()
        _install_rust_streaming_shim()
        _install_rust_wrapper_shim()
        _install_data_utils_shim()
        _install_collators_shim()

        _SHIM_INSTALLED = True
        LOG.info("Fast-axolotl shim installed successfully")
        return True

    except Exception as e:
        LOG.error(f"Failed to install fast-axolotl shim: {e}")
        return False


def _install_rust_ext_shim():
    """Install shim for axolotl.rust_ext module."""
    import types

    if "axolotl" not in sys.modules:
        axolotl_mod = types.ModuleType("axolotl")
        sys.modules["axolotl"] = axolotl_mod
    else:
        axolotl_mod = sys.modules["axolotl"]

    if "axolotl.rust_ext" not in sys.modules:
        rust_ext_mod = types.ModuleType("axolotl.rust_ext")
        sys.modules["axolotl.rust_ext"] = rust_ext_mod

    if "axolotl.rust_ext.axolotl_ext" not in sys.modules:
        axolotl_ext_mod = types.ModuleType("axolotl.rust_ext.axolotl_ext")
        axolotl_ext_mod.streaming_dataset_reader = _rust_streaming_reader
        sys.modules["axolotl.rust_ext.axolotl_ext"] = axolotl_ext_mod


def _install_rust_streaming_shim():
    """Install shim for axolotl.utils.data.rust_streaming module."""
    import types

    if "axolotl.utils" not in sys.modules:
        utils_mod = types.ModuleType("axolotl.utils")
        sys.modules["axolotl.utils"] = utils_mod

    if "axolotl.utils.data" not in sys.modules:
        data_mod = types.ModuleType("axolotl.utils.data")
        sys.modules["axolotl.utils.data"] = data_mod

    if "axolotl.utils.data.rust_streaming" not in sys.modules:
        rust_streaming_mod = types.ModuleType("axolotl.utils.data.rust_streaming")
        rust_streaming_mod.get_rust_extension_status = is_available
        rust_streaming_mod.streaming_dataset_reader = streaming_dataset_reader
        rust_streaming_mod.RUST_EXTENSION_AVAILABLE = RUST_AVAILABLE
        sys.modules["axolotl.utils.data.rust_streaming"] = rust_streaming_mod


def _install_rust_wrapper_shim():
    """Install shim for axolotl.utils.data.rust_wrapper module."""
    import types

    if "axolotl.utils.data" not in sys.modules:
        data_mod = types.ModuleType("axolotl.utils.data")
        sys.modules["axolotl.utils.data"] = data_mod

    if "axolotl.utils.data.rust_wrapper" not in sys.modules:
        rust_wrapper_mod = types.ModuleType("axolotl.utils.data.rust_wrapper")
        rust_wrapper_mod.is_rust_streaming_available = is_available
        rust_wrapper_mod.load_dataset_with_rust_streaming = streaming_dataset_reader
        rust_wrapper_mod.RustStreamingDataset = RustStreamingDataset
        rust_wrapper_mod.create_rust_streaming_dataset = create_rust_streaming_dataset
        rust_wrapper_mod.should_use_rust_streaming = should_use_rust_streaming
        sys.modules["axolotl.utils.data.rust_wrapper"] = rust_wrapper_mod


def _install_data_utils_shim():
    """Install acceleration for axolotl.utils.data.utils (parallel hashing)."""
    import types

    if "axolotl.utils.data" not in sys.modules:
        data_mod = types.ModuleType("axolotl.utils.data")
        sys.modules["axolotl.utils.data"] = data_mod
    else:
        data_mod = sys.modules["axolotl.utils.data"]

    # Add fast deduplication functions to data module
    data_mod.fast_parallel_hash_rows = parallel_hash_rows
    data_mod.fast_deduplicate_indices = deduplicate_indices


def _install_collators_shim():
    """Install acceleration for axolotl.utils.collators (batch padding)."""
    import types

    if "axolotl.utils" not in sys.modules:
        utils_mod = types.ModuleType("axolotl.utils")
        sys.modules["axolotl.utils"] = utils_mod
    else:
        utils_mod = sys.modules["axolotl.utils"]

    if "axolotl.utils.collators" not in sys.modules:
        collators_mod = types.ModuleType("axolotl.utils.collators")
        sys.modules["axolotl.utils.collators"] = collators_mod
    else:
        collators_mod = sys.modules["axolotl.utils.collators"]

    # Add fast padding functions
    collators_mod.fast_pad_sequences = pad_sequences
    collators_mod.fast_create_padding_mask = create_padding_mask


def uninstall() -> bool:
    """
    Remove the fast-axolotl shim from the axolotl namespace.

    Returns:
        True if shim was removed, False if not installed
    """
    global _SHIM_INSTALLED

    if not _SHIM_INSTALLED:
        return False

    modules_to_remove = [
        "axolotl.rust_ext.axolotl_ext",
        "axolotl.utils.data.rust_streaming",
        "axolotl.utils.data.rust_wrapper",
    ]

    for mod_name in modules_to_remove:
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    _SHIM_INSTALLED = False
    LOG.info("Fast-axolotl shim uninstalled")
    return True


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Core
    "is_available",
    "get_version",
    "install",
    "uninstall",
    "RUST_AVAILABLE",
    # Format Detection
    "list_supported_formats",
    "detect_format",
    # Streaming
    "streaming_dataset_reader",
    "RustStreamingDataset",
    "create_rust_streaming_dataset",
    "should_use_rust_streaming",
    # Token Packing (Acceleration #1)
    "pack_sequences",
    "concatenate_and_pack",
    # Parallel Hashing (Acceleration #2)
    "parallel_hash_rows",
    "deduplicate_indices",
    # Batch Padding (Acceleration #3)
    "pad_sequences",
    "create_padding_mask",
]


# Auto-install on import if Rust extension is available
if RUST_AVAILABLE:
    install()
