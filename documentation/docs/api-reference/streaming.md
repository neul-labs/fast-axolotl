# Streaming API

This page covers the streaming reader, its HuggingFace-compatible wrapper,
and the helper functions Axolotl uses to decide whether to switch streaming
on.

All names are importable from `fast_axolotl`.

---

## `streaming_dataset_reader`

```python
def streaming_dataset_reader(
    file_path: str,
    dataset_type: str,
    batch_size: int = 1000,
    num_threads: int = 4,
) -> Iterator[Dict[str, Any]]
```

Stream batches from a dataset file. Yields `Dict[str, List[Any]]` where the
keys are column names and the values are batch-sized columns.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `file_path` | `str` | required | file or directory path |
| `dataset_type` | `str` | required | one of `parquet`, `arrow`, `feather`, `csv`, `json`, `jsonl`, `text` |
| `batch_size` | `int` | `1000` | rows per yielded batch |
| `num_threads` | `int` | `4` | worker threads for I/O / decode |

```python
from fast_axolotl import streaming_dataset_reader

for batch in streaming_dataset_reader("data.parquet", "parquet", batch_size=5000):
    texts = batch["text"]
    labels = batch["label"]
```

Raises:

- `ImportError` - Rust extension is not available
- `FileNotFoundError` - path does not exist
- `PermissionError` - access denied
- `ValueError` - invalid `dataset_type` or bad arguments
- `RuntimeError` - underlying Arrow / Parquet / JSON / CSV error

---

## `RustStreamingDataset`

```python
class RustStreamingDataset:
    def __init__(
        self,
        file_path: str,
        dataset_type: str,
        batch_size: int = 1000,
        num_threads: int = 4,
        dataset_keep_in_memory: bool = False,
    )
```

HuggingFace-Dataset-compatible iterable wrapper around
[`streaming_dataset_reader`](#streaming_dataset_reader). Iteration yields
the same `Dict[str, List[Any]]` batches.

```python
from fast_axolotl import RustStreamingDataset

dataset = RustStreamingDataset(
    file_path="data.parquet",
    dataset_type="parquet",
    batch_size=1000,
)

for batch in dataset:
    process(batch)
```

---

## `create_rust_streaming_dataset`

```python
def create_rust_streaming_dataset(
    cfg: dict,
    file_path: str,
    dataset_type: str,
) -> Optional[RustStreamingDataset]
```

Config-driven helper. Returns a `RustStreamingDataset` configured from an
Axolotl-style config dict (`batch_size`, `num_threads`, etc.) when the Rust
extension is available and the config opts in
(`dataset_use_rust_streaming: true`); returns `None` otherwise.

---

## `should_use_rust_streaming`

```python
def should_use_rust_streaming(
    cfg: dict,
    file_size_bytes: Optional[int] = None,
) -> bool
```

Decide whether streaming is worth turning on for a given config and file
size. Returns `True` when any of the following hold (and the Rust extension
is available):

- `cfg["dataset_use_rust_streaming"]` is truthy
- `cfg["sequence_len"]` exceeds 10,000
- `file_size_bytes` exceeds roughly 1 GB

Used by `create_rust_streaming_dataset` to avoid streaming overhead on
small datasets where the in-memory path is fine.

---

## See also

- [Streaming Data Loading guide](../user-guide/streaming.md)
- [Auto-Shimming](../user-guide/shimming.md) - how Axolotl picks the reader up automatically
