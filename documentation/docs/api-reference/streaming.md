# Streaming API

This page documents the streaming data loading functions.

## Primary Functions

### `streaming_dataset_reader()`

Create a streaming reader for data files.

```python
fast_axolotl.streaming_dataset_reader(
    path: str,
    batch_size: int = 1000,
    columns: list[str] | None = None
) -> Iterator[dict]
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | required | File path or glob pattern |
| `batch_size` | `int` | `1000` | Number of rows per batch |
| `columns` | `list[str]` | `None` | Columns to load (None = all) |

**Returns**: An iterator yielding dictionaries with column names as keys and lists of values.

**Supported Formats**:

- Parquet (`.parquet`)
- Arrow (`.arrow`)
- Feather (`.feather`)
- JSON (`.json`)
- JSONL (`.jsonl`, `.ndjson`)
- CSV (`.csv`)
- Text (`.txt`)

**Compression**: ZSTD (`.zst`, `.zstd`) and Gzip (`.gz`) are automatically detected.

**Examples**:

Basic usage:
```python
from fast_axolotl import streaming_dataset_reader

for batch in streaming_dataset_reader("data/train.parquet"):
    print(batch.keys())
    print(len(batch["input_ids"]))
```

With options:
```python
reader = streaming_dataset_reader(
    "data/**/*.parquet",
    batch_size=500,
    columns=["input_ids", "labels"]
)

for batch in reader:
    process(batch)
```

---

### `create_rust_streaming_dataset()`

Create a HuggingFace-compatible streaming dataset.

```python
fast_axolotl.create_rust_streaming_dataset(
    path: str,
    batch_size: int = 32,
    columns: list[str] | None = None
) -> RustStreamingDataset
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | required | File path or glob pattern |
| `batch_size` | `int` | `32` | Number of rows per batch |
| `columns` | `list[str]` | `None` | Columns to load (None = all) |

**Returns**: A `RustStreamingDataset` object compatible with PyTorch DataLoader.

**Example**:
```python
from fast_axolotl import create_rust_streaming_dataset
from torch.utils.data import DataLoader

dataset = create_rust_streaming_dataset(
    "data/train.parquet",
    batch_size=32
)

# Use with DataLoader (batch_size=None since dataset handles batching)
loader = DataLoader(dataset, batch_size=None, num_workers=0)

for batch in loader:
    model.train_step(batch)
```

---

## Classes

### `RustStreamingDataset`

A PyTorch-compatible dataset wrapper for streaming data.

```python
class RustStreamingDataset:
    def __init__(
        self,
        path: str,
        batch_size: int = 32,
        columns: list[str] | None = None
    )

    def __iter__(self) -> Iterator[dict]
    def __len__(self) -> int  # Approximate length
```

**Methods**:

| Method | Description |
|--------|-------------|
| `__iter__()` | Iterate over batches |
| `__len__()` | Get approximate dataset length |

**Example**:
```python
from fast_axolotl import RustStreamingDataset

dataset = RustStreamingDataset("data.parquet", batch_size=32)

for batch in dataset:
    input_ids = batch["input_ids"]
    labels = batch["labels"]
```

---

## Low-Level Functions

### `_rust_streaming_reader()`

Low-level Rust streaming reader (internal use).

```python
fast_axolotl._rust_streaming_reader(
    path: str,
    batch_size: int,
    columns: list[str] | None
) -> Iterator
```

!!! warning
    This is an internal function. Use `streaming_dataset_reader()` instead.

---

## Error Handling

### Common Exceptions

| Exception | Cause |
|-----------|-------|
| `FileNotFoundError` | File or pattern doesn't match any files |
| `ValueError` | Unsupported format or invalid parameters |
| `IOError` | File read error or corruption |

**Example**:
```python
from fast_axolotl import streaming_dataset_reader

try:
    for batch in streaming_dataset_reader("data/*.parquet"):
        process(batch)
except FileNotFoundError:
    print("No matching files found")
except ValueError as e:
    print(f"Format error: {e}")
```

---

## Performance Notes

### Batch Size Selection

| Dataset Size | Recommended Batch Size | Memory Usage |
|--------------|----------------------|--------------|
| < 100K rows | 1000-5000 | Low |
| 100K-1M rows | 500-2000 | Medium |
| > 1M rows | 100-1000 | High |

### Format Performance

| Format | Read Speed | Memory Efficiency |
|--------|-----------|-------------------|
| Parquet | Fastest | Excellent |
| Arrow | Very Fast | Excellent |
| JSONL | Fast | Good |
| CSV | Moderate | Good |
| JSON | Slower | Moderate |

### Compression Performance

| Compression | Decompression Speed | Compression Ratio |
|-------------|--------------------|--------------------|
| None | Fastest | 1x |
| ZSTD | Fast | 3-5x |
| Gzip | Moderate | 2-4x |

---

## See Also

- [Streaming Guide](../user-guide/streaming.md) - Usage patterns and examples
- [Core Functions](core.md) - Format detection utilities
- [Benchmarks](../performance/benchmarks.md) - Performance data
