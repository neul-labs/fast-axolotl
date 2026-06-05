# Streaming Data Loading

`fast-axolotl`'s streaming reader is the largest single win in the package:
**77x faster** Parquet loading than the Python baseline in the project
benchmark. This guide explains the API and how to get the most out of it.

## Why streaming?

The Rust reader avoids the per-row Python object overhead of `datasets` and
streams record batches straight from the file:

- Native Parquet / Arrow / CSV / JSON parsers (`arrow`, `parquet`, `csv`,
  `arrow-json` crates)
- Multi-threaded I/O via Tokio
- Transparent ZSTD and Gzip decompression
- HuggingFace Arrow-dataset directories supported out of the box

## Basic Usage

```python
from fast_axolotl import streaming_dataset_reader

for batch in streaming_dataset_reader(
    file_path="/path/to/train.parquet",
    dataset_type="parquet",
    batch_size=1000,
    num_threads=4,
):
    process(batch)
```

Each yielded `batch` is a `Dict[str, List[Any]]` keyed by column name.

### Parameters

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `file_path` | `str` | required | path to file or directory |
| `dataset_type` | `str` | required | one of `parquet`, `arrow`, `feather`, `csv`, `json`, `jsonl`, `text` |
| `batch_size` | `int` | `1000` | rows per yielded batch |
| `num_threads` | `int` | `4` | worker threads for I/O / decode |

!!! tip "Choosing batch size"
    Larger batches improve throughput but use more memory. 1,000-10,000 is a
    good starting range for typical LLM training rows.

## Format Detection

If you don't already know the format, `detect_format` returns `(format, compression)`:

```python
from fast_axolotl import detect_format, list_supported_formats

detect_format("data.parquet")        # ("parquet", None)
detect_format("data.jsonl.zst")      # ("jsonl", "zstd")
detect_format("data.csv.gz")         # ("csv", "gzip")
detect_format("/path/to/hf_dir/")    # ("hf_dataset", None)

list_supported_formats()
# ['parquet', 'arrow', 'feather', 'csv', 'json', 'jsonl', 'text',
#  'parquet.zst', 'parquet.gz', 'arrow.zst', 'arrow.gz',
#  'json.zst', 'json.gz', 'jsonl.zst', 'jsonl.gz',
#  'csv.zst', 'csv.gz', 'text.zst', 'text.gz', 'hf_dataset']
```

## Supported Formats

| Format | Extensions | Notes |
|---|---|---|
| Parquet | `.parquet` | Columnar, recommended |
| Arrow IPC | `.arrow`, `.ipc` | Zero-copy capable |
| Feather | `.feather` | Arrow IPC v2 |
| JSON | `.json` | Array of objects |
| JSONL | `.jsonl`, `.ndjson` | Line-delimited |
| CSV / TSV | `.csv`, `.tsv` | Comma/tab |
| Text | `.txt` | One record per line |
| HuggingFace dataset | directory with `dataset_info.json` | auto-detected |

All file formats may be transparently `.zst`- or `.gz`-compressed.

## HuggingFace-Compatible Wrapper

`RustStreamingDataset` exposes the same reader as an iterable class - useful
for slotting into existing dataset pipelines:

```python
from fast_axolotl import RustStreamingDataset

dataset = RustStreamingDataset(
    file_path="/data/train.parquet",
    dataset_type="parquet",
    batch_size=1000,
    num_threads=4,
)

for batch in dataset:
    train_step(batch)
```

For config-driven plumbing inside Axolotl, `create_rust_streaming_dataset`
and `should_use_rust_streaming` accept the YAML config dict directly. The
default rule of thumb is: enable streaming when
`dataset_use_rust_streaming: true`, or when sequence lengths exceed 10,000
or files are larger than 1 GB.

## Performance Tips

### Prefer Parquet

Parquet's columnar layout plays best with the Rust reader. Convert other
formats when possible:

```python
import pandas as pd

pd.read_json("data.jsonl", lines=True).to_parquet("data.parquet")
```

### Use ZSTD compression

ZSTD strikes a good balance between compression ratio and decode speed:

```python
import pyarrow.parquet as pq

pq.write_table(table, "data.parquet", compression="zstd")
```

### Right-size `batch_size`

| Dataset size | Suggested `batch_size` |
|---|---|
| < 100K rows | 1,000-5,000 |
| 100K - 1M rows | 500-2,000 |
| > 1M rows | 100-1,000 |

### Let the reader manage threads

`num_threads=4` is a sensible default; raise it on machines with many cores
and fast storage. The reader uses a Tokio multi-thread runtime under the
hood, so Python-level worker pools usually aren't needed.

## Error Handling

The reader raises standard Python exceptions:

- `FileNotFoundError` - the path does not exist
- `PermissionError` - access denied
- `ValueError` - invalid argument (bad `dataset_type`, bad batch_size, ...)
- `RuntimeError` - underlying Arrow / Parquet / CSV / JSON error

```python
try:
    for batch in streaming_dataset_reader(path, "parquet"):
        process(batch)
except FileNotFoundError:
    log.error("missing dataset: %s", path)
except ValueError as e:
    log.error("bad streaming config: %s", e)
```

## See also

- [Streaming API](../api-reference/streaming.md)
- [Benchmarks](../performance/benchmarks.md)
- [Auto-Shimming](shimming.md) - how Axolotl picks up the Rust reader automatically
