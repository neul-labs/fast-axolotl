# Usage Guide

This guide covers how to use fast-axolotl to accelerate your Axolotl workflows.

## Quick Start

The simplest way to use fast-axolotl is automatic shimming:

```python
import fast_axolotl  # Auto-installs acceleration shim

# Now use axolotl normally
import axolotl
```

That's it! The shim automatically patches axolotl modules to use Rust-accelerated implementations.

## How the Shim Works

When you import `fast_axolotl`, it:

1. Loads the Rust extension
2. Creates virtual modules in `sys.modules` that shadow axolotl's modules
3. Binds Rust functions to these modules

This means axolotl code that imports from these modules automatically gets the accelerated versions.

### Shimmed Modules

| Module | Accelerated Functions |
|--------|----------------------|
| `axolotl.utils.data.rust_streaming` | `streaming_dataset_reader` |
| `axolotl.utils.data` | `fast_parallel_hash_rows`, `fast_deduplicate_indices` |
| `axolotl.utils.collators` | `fast_pad_sequences`, `fast_create_padding_mask` |

### Manual Shim Control

```python
import fast_axolotl

# Check if shim is active
print(fast_axolotl.is_available())  # True if Rust extension loaded

# Uninstall the shim
fast_axolotl.uninstall()

# Re-install the shim
fast_axolotl.install()
```

## Direct API Usage

You can also use fast-axolotl functions directly without axolotl.

### Streaming Data Loading

Load large datasets efficiently without loading everything into memory:

```python
from fast_axolotl import streaming_dataset_reader

# Stream from a Parquet file
for batch in streaming_dataset_reader(
    file_path="/path/to/data.parquet",
    dataset_type="parquet",
    batch_size=1000,
    num_threads=4
):
    # batch is a dict with column names as keys
    texts = batch.get("text", [])
    labels = batch.get("label", [])
    process(texts, labels)
```

#### Supported Formats

```python
from fast_axolotl import list_supported_formats, detect_format

# List all formats
print(list_supported_formats())
# ['parquet', 'arrow', 'json', 'jsonl', 'csv', 'text', ...]

# Auto-detect format
format, compression = detect_format("data.jsonl.zst")
print(format, compression)  # ('jsonl', 'zstd')
```

| Format | Extensions | Notes |
|--------|------------|-------|
| Parquet | `.parquet` | Columnar, recommended for large datasets |
| Arrow | `.arrow`, `.ipc` | Arrow IPC format |
| JSON | `.json` | Array of objects |
| JSONL | `.jsonl`, `.ndjson` | JSON Lines, one object per line |
| CSV | `.csv`, `.tsv` | Comma/tab separated |
| Text | `.txt` | Plain text, one line per record |

All formats support `.zst` (ZSTD) and `.gz` (Gzip) compression.

### Token Packing

Pack variable-length sequences into fixed-length chunks for efficient training:

```python
from fast_axolotl import pack_sequences

sequences = [
    [101, 2054, 2003, 2023, 102],      # "What is this"
    [101, 1037, 3231, 102],             # "A test"
    [101, 2178, 6251, 102],             # "Another sentence"
]

result = pack_sequences(
    sequences=sequences,
    max_length=20,
    pad_token_id=0,
    eos_token_id=102,
    label_pad_id=-100
)

print(result.keys())
# dict_keys(['input_ids', 'labels', 'attention_mask'])

# Each output sequence is exactly max_length
for seq in result['input_ids']:
    assert len(seq) == 20
```

#### Lower-Level Packing

If you have separate input_ids, labels, and attention_masks:

```python
from fast_axolotl import concatenate_and_pack

result = concatenate_and_pack(
    input_ids=[[1, 2, 3], [4, 5]],
    labels=[[1, 2, 3], [4, 5]],
    attention_masks=[[1, 1, 1], [1, 1]],
    max_length=10,
    pad_token_id=0,
    label_pad_id=-100
)
```

### Parallel Hashing

Hash strings in parallel for deduplication:

```python
from fast_axolotl import parallel_hash_rows, deduplicate_indices

# Hash rows using all CPU cores
rows = [str(row) for row in dataset]
hashes = parallel_hash_rows(rows, num_threads=0)  # 0 = auto-detect cores

# Or deduplicate directly
unique_indices, all_hashes = deduplicate_indices(rows)
deduplicated_dataset = dataset.select(unique_indices)
```

#### With Existing Hashes

Filter against previously seen data:

```python
existing_hashes = load_previous_hashes()
unique_indices, new_hashes = deduplicate_indices(
    rows,
    existing_hashes=existing_hashes,
    num_threads=8
)
```

### Batch Padding

Pad sequences to uniform length:

```python
from fast_axolotl import pad_sequences

sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]]

# Right padding (default)
padded = pad_sequences(
    sequences,
    target_length=8,
    pad_value=0,
    padding_side="right"
)
# [[1, 2, 3, 0, 0, 0, 0, 0],
#  [4, 5, 0, 0, 0, 0, 0, 0],
#  [6, 7, 8, 9, 10, 0, 0, 0]]

# Left padding
padded_left = pad_sequences(
    sequences,
    target_length=8,
    pad_value=0,
    padding_side="left"
)
# [[0, 0, 0, 0, 0, 1, 2, 3],
#  [0, 0, 0, 0, 0, 0, 4, 5],
#  [0, 0, 0, 6, 7, 8, 9, 10]]
```

#### Pad to Multiple

Useful for hardware alignment:

```python
padded = pad_sequences(
    sequences,
    pad_value=0,
    pad_to_multiple_of=8  # Pad to nearest multiple of 8
)
```

## Axolotl Configuration

Enable fast-axolotl features in your Axolotl YAML config:

```yaml
# Enable Rust-based streaming for large datasets
dataset_use_rust_streaming: true

# Streaming is auto-enabled for:
# - Files > 1GB
# - sequence_len > 10000
sequence_len: 32768

# Deduplication automatically uses parallel hashing
dedupe: true
```

## Best Practices

### 1. Import Order

Always import fast_axolotl before axolotl:

```python
import fast_axolotl  # First!
import axolotl       # Then axolotl
```

### 2. Large Datasets

For datasets over 1GB, use streaming:

```python
from fast_axolotl import streaming_dataset_reader

# Don't load everything at once
for batch in streaming_dataset_reader(path, "parquet", batch_size=10000):
    process_batch(batch)
```

### 3. Deduplication

Use parallel hashing for large-scale deduplication:

```python
# Convert rows to strings for hashing
row_strings = [json.dumps(row, sort_keys=True) for row in dataset]
unique_idx, _ = deduplicate_indices(row_strings)
```

### 4. Thread Count

Let fast-axolotl auto-detect the optimal thread count:

```python
# num_threads=0 auto-detects CPU cores
hashes = parallel_hash_rows(rows, num_threads=0)
```

## Next Steps

- [API Reference](api.md) - Complete function documentation
- [Benchmarks](benchmarks.md) - Performance comparisons
- [Compatibility](compatibility.md) - Tested configurations
