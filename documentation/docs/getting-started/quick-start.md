# Quick Start

This page walks through the two ways you'll use `fast-axolotl`: as a
transparent shim for an existing Axolotl install, and as a direct Rust API
you call from your own code.

## Option 1: Drop-in Acceleration

Import order matters. Bring in `fast_axolotl` **before** `axolotl` so the
shim has a chance to install itself:

```python
import fast_axolotl  # installs the shim on import (when Rust ext is available)
import axolotl       # now resolves to the accelerated implementations
```

The shim is installed automatically at import time. You can drive it
manually if you need to:

```python
import fast_axolotl

fast_axolotl.is_available()   # True if the Rust extension loaded
fast_axolotl.install()        # re-install (idempotent)
fast_axolotl.uninstall()      # remove the shim
```

See [Auto-Shimming](../user-guide/shimming.md) for the full list of patched modules.

## Option 2: Direct API

The Rust functions are also exported directly from `fast_axolotl`. Use them
wherever you have a hot loop.

### Streaming a dataset

```python
from fast_axolotl import streaming_dataset_reader

for batch in streaming_dataset_reader(
    file_path="/path/to/data.parquet",
    dataset_type="parquet",
    batch_size=1000,
    num_threads=4,
):
    texts  = batch.get("text", [])
    labels = batch.get("label", [])
    train_step(texts, labels)
```

`dataset_type` accepts `"parquet"`, `"arrow"`, `"feather"`, `"json"`,
`"jsonl"`, `"csv"`, or `"text"`. Compression (`.zst`, `.gz`) is detected from
the filename automatically.

### Packing token sequences

```python
from fast_axolotl import pack_sequences

result = pack_sequences(
    sequences=[[1, 2, 3], [4, 5], [6, 7, 8, 9]],
    max_length=10,
    pad_token_id=0,
    eos_token_id=2,
    label_pad_id=-100,
)
# result == {"input_ids": [...], "labels": [...], "attention_mask": [...]}
```

### Parallel deduplication

```python
from fast_axolotl import parallel_hash_rows, deduplicate_indices

rows = [str(row) for row in dataset]
hashes = parallel_hash_rows(rows, num_threads=0)   # 0 = auto-detect cores

unique_indices, all_hashes = deduplicate_indices(rows)
deduped = dataset.select(unique_indices)
```

### Batch padding

```python
from fast_axolotl import pad_sequences

padded = pad_sequences(
    [[1, 2, 3], [4, 5]],
    target_length=8,
    pad_value=0,
    padding_side="right",
)
# [[1, 2, 3, 0, 0, 0, 0, 0],
#  [4, 5, 0, 0, 0, 0, 0, 0]]
```

## Option 3: HuggingFace-Compatible Streaming

`RustStreamingDataset` is a thin HuggingFace-compatible wrapper that yields
batches as Python dicts:

```python
from fast_axolotl import RustStreamingDataset

dataset = RustStreamingDataset(
    file_path="/path/to/data.parquet",
    dataset_type="parquet",
    batch_size=1000,
    num_threads=4,
)

for batch in dataset:
    train_step(batch)
```

There is also a config-driven helper that decides for you whether streaming
should kick in based on an Axolotl config dict:

```python
from fast_axolotl import should_use_rust_streaming, create_rust_streaming_dataset

cfg = {"dataset_use_rust_streaming": True, "sequence_len": 32768}

if should_use_rust_streaming(cfg, file_size_bytes=2 * 1024**3):
    ds = create_rust_streaming_dataset(cfg, "/path/to/data.parquet", "parquet")
```

## Axolotl YAML Config

Enable the same features inside an Axolotl YAML config:

```yaml
# Force-enable Rust streaming
dataset_use_rust_streaming: true

# Auto-enables when files are >1GB or sequence_len > 10000
sequence_len: 32768

# Deduplication automatically uses parallel hashing once the shim is in place
dedupe: true
```

## Next Steps

- [Streaming Data Loading](../user-guide/streaming.md) - in-depth streaming guide
- [Token Packing](../user-guide/token-packing.md) - how packing works under the hood
- [API Reference](../api-reference/core.md) - every exported function
