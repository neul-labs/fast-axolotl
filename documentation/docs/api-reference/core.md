# Core API

The core API covers package metadata, runtime detection, and shim control.
All names listed here are importable directly from `fast_axolotl`.

## Module metadata

### `__version__`

```python
fast_axolotl.__version__   # "0.2.0"
```

A plain Python string. For a combined Python + Rust version string, prefer
[`get_version`](#get_version).

### `RUST_AVAILABLE`

```python
fast_axolotl.RUST_AVAILABLE   # bool
```

`True` if the compiled `_rust_ext` module loaded successfully on this
platform. The package auto-runs [`install`](#install) when this is `True`.

---

## Runtime detection

### `is_available`

```python
def is_available() -> bool
```

Convenience wrapper around `RUST_AVAILABLE`. Returns `True` if the Rust
extension is loaded.

```python
if not fast_axolotl.is_available():
    raise RuntimeError("fast-axolotl Rust extension is not available")
```

### `get_version`

```python
def get_version() -> str
```

Returns a combined version string:

```python
fast_axolotl.get_version()
# "0.2.0 (rust: 0.2.0)"
# or  "0.2.0 (rust: not available)" if the Rust extension is missing
```

---

## Format catalogue

### `list_supported_formats`

```python
def list_supported_formats() -> List[str]
```

Returns every file format the streaming reader recognises, including
compressed variants and the `hf_dataset` directory marker:

```python
fast_axolotl.list_supported_formats()
# ['parquet', 'arrow', 'feather', 'csv', 'json', 'jsonl', 'text',
#  'parquet.zst', 'parquet.gz', 'arrow.zst', 'arrow.gz',
#  'json.zst', 'json.gz', 'jsonl.zst', 'jsonl.gz',
#  'csv.zst', 'csv.gz', 'text.zst', 'text.gz',
#  'hf_dataset']
```

### `detect_format`

```python
def detect_format(file_path: str) -> Tuple[str, Optional[str]]
```

Returns `(base_format, compression)` based on the path's extension(s):

```python
fast_axolotl.detect_format("data.parquet")        # ("parquet", None)
fast_axolotl.detect_format("data.jsonl.zst")      # ("jsonl", "zstd")
fast_axolotl.detect_format("data.csv.gz")         # ("csv", "gzip")
fast_axolotl.detect_format("/path/to/hf_dir/")    # ("hf_dataset", None)
```

Raises `ImportError` if the Rust extension is not loaded.

---

## Shim control

### `install`

```python
def install() -> bool
```

Install the shim entries into `sys.modules`. Idempotent: returns `True` if
new entries were installed, `False` if the shim was already in place or
the Rust extension is unavailable.

The shim is also called automatically at the bottom of
`fast_axolotl/__init__.py` whenever `RUST_AVAILABLE` is `True`.

### `uninstall`

```python
def uninstall() -> bool
```

Remove the shim. Returns `True` if entries were removed, `False` otherwise.
Useful for benchmarking against the Axolotl Python baseline.

---

## Full export list

`fast_axolotl.__all__` defines the public API:

```python
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
    # Token Packing
    "pack_sequences",
    "concatenate_and_pack",
    # Parallel Hashing
    "parallel_hash_rows",
    "deduplicate_indices",
    # Batch Padding
    "pad_sequences",
    "create_padding_mask",
]
```

The streaming entries are documented in [Streaming API](streaming.md); the
packing, hashing, and padding entries in [Data Processing API](data-processing.md).
