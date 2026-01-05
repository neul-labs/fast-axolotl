# Core Functions

This page documents the core functions available in fast-axolotl.

## Module Info

### `__version__`

The package version string.

```python
import fast_axolotl
print(fast_axolotl.__version__)  # e.g., "0.1.12"
```

### `rust_available()`

Check if the Rust extension is loaded and available.

```python
fast_axolotl.rust_available() -> bool
```

**Returns**: `True` if Rust acceleration is available, `False` otherwise.

**Example**:
```python
if fast_axolotl.rust_available():
    print("Rust acceleration enabled")
else:
    print("Falling back to Python implementation")
```

---

## Shimming Functions

### `install()`

Install acceleration shims into the Axolotl namespace.

```python
fast_axolotl.install() -> None
```

This function replaces Axolotl's Python implementations with Rust-accelerated versions. Call this before importing any Axolotl modules.

**Example**:
```python
import fast_axolotl
fast_axolotl.install()

# Now import axolotl - it will use accelerated functions
import axolotl
```

**Notes**:

- Safe to call multiple times (idempotent)
- Safe to call even if Axolotl is not installed
- Must be called before importing Axolotl modules

---

### `uninstall()`

Remove acceleration shims and restore original Axolotl implementations.

```python
fast_axolotl.uninstall() -> None
```

**Example**:
```python
# Temporarily disable acceleration
fast_axolotl.uninstall()

# Run with original implementation
result = some_axolotl_function()

# Re-enable acceleration
fast_axolotl.install()
```

---

## Format Detection

### `detect_format(path)`

Detect the file format and compression of a data file.

```python
fast_axolotl.detect_format(path: str) -> dict
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Path to the file to analyze |

**Returns**: A dictionary with keys:

- `format`: The detected format (`"parquet"`, `"arrow"`, `"json"`, `"jsonl"`, `"csv"`, `"text"`)
- `compression`: The detected compression (`"zstd"`, `"gzip"`, `None`)

**Example**:
```python
info = fast_axolotl.detect_format("data/train.parquet.zst")
print(info)
# {'format': 'parquet', 'compression': 'zstd'}

info = fast_axolotl.detect_format("data/train.jsonl")
print(info)
# {'format': 'jsonl', 'compression': None}
```

---

### `list_supported_formats()`

List all supported file formats.

```python
fast_axolotl.list_supported_formats() -> list[str]
```

**Returns**: A list of supported format names.

**Example**:
```python
formats = fast_axolotl.list_supported_formats()
print(formats)
# ['parquet', 'arrow', 'feather', 'json', 'jsonl', 'csv', 'text']
```

---

## Quick Reference

| Function | Description |
|----------|-------------|
| `rust_available()` | Check if Rust is loaded |
| `install()` | Enable shimming |
| `uninstall()` | Disable shimming |
| `detect_format(path)` | Detect file format |
| `list_supported_formats()` | List supported formats |

## See Also

- [Streaming API](streaming.md) - Data loading functions
- [Data Processing API](data-processing.md) - Packing, hashing, padding functions
