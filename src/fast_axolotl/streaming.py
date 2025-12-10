"""
Direct streaming utilities for fast-axolotl.

This module provides direct access to the Rust-based streaming functionality
without requiring axolotl to be installed.
"""

from fast_axolotl import (
    RUST_AVAILABLE,
    streaming_dataset_reader,
    RustStreamingDataset,
    create_rust_streaming_dataset,
    is_available,
)

__all__ = [
    "RUST_AVAILABLE",
    "streaming_dataset_reader",
    "RustStreamingDataset",
    "create_rust_streaming_dataset",
    "is_available",
]
