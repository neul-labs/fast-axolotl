# Streaming Data Loading

fast-axolotl's streaming data loader is one of its most powerful features, providing up to **77x faster** data loading compared to Python-based solutions.

## Overview

The streaming reader efficiently loads large datasets that don't fit in memory by:

- Reading data in configurable batches
- Supporting multiple file formats natively
- Handling compression transparently
- Using memory-mapped I/O where possible

## Basic Usage

```python
from fast_axolotl import streaming_dataset_reader

# Simple streaming from a single file
for batch in streaming_dataset_reader("data/train.parquet"):
    process(batch)
```

## Configuration Options

### Batch Size

Control how many rows are loaded per iteration:

```python
# Load 1000 rows at a time
reader = streaming_dataset_reader(
    "data/train.parquet",
    batch_size=1000
)
```

!!! tip "Choosing batch size"
    Larger batch sizes improve throughput but use more memory. Start with 1000 and adjust based on your memory constraints.

### Column Selection

Load only the columns you need:

```python
reader = streaming_dataset_reader(
    "data/train.parquet",
    columns=["input_ids", "attention_mask", "labels"]
)
```

This reduces memory usage and improves performance when your dataset has many columns.

### Multiple Files

Use glob patterns to stream from multiple files:

```python
# All parquet files in a directory
reader = streaming_dataset_reader("data/*.parquet")

# Recursive glob
reader = streaming_dataset_reader("data/**/*.parquet")

# Multiple patterns
reader = streaming_dataset_reader("data/train_*.parquet")
```

## Supported Formats

fast-axolotl automatically detects file formats:

| Format | Extensions | Description |
|--------|-----------|-------------|
| Parquet | `.parquet` | Columnar format, best performance |
| Arrow | `.arrow` | Zero-copy memory mapping |
| Feather | `.feather` | Fast binary format |
| JSON | `.json` | Standard JSON arrays |
| JSONL | `.jsonl`, `.ndjson` | Line-delimited JSON |
| CSV | `.csv` | Comma-separated values |
| Text | `.txt` | Plain text (one row per line) |

### Format Detection

```python
from fast_axolotl import detect_format

# Automatic detection
format_info = detect_format("data/train.parquet.zst")
print(format_info)
# {'format': 'parquet', 'compression': 'zstd'}
```

### Compression Support

ZSTD and Gzip compression are automatically handled:

```python
# These all work automatically
reader = streaming_dataset_reader("data/train.parquet.zst")  # ZSTD
reader = streaming_dataset_reader("data/train.json.gz")      # Gzip
reader = streaming_dataset_reader("data/train.csv.zstd")     # ZSTD
```

## Advanced Usage

### Custom Iteration

```python
reader = streaming_dataset_reader("data/train.parquet", batch_size=500)

# Manual iteration
batch = next(iter(reader))

# Check batch contents
print(batch.keys())      # Column names
print(len(batch["input_ids"]))  # Batch size
```

### Memory-Efficient Processing

For very large datasets, process and discard batches to minimize memory:

```python
def process_large_dataset(path):
    total_rows = 0

    for batch in streaming_dataset_reader(path, batch_size=1000):
        # Process batch
        total_rows += len(batch["input_ids"])

        # Batch is automatically freed when loop continues

    return total_rows
```

### Combining with PyTorch DataLoader

```python
from fast_axolotl import create_rust_streaming_dataset
from torch.utils.data import DataLoader

# Create HF-compatible dataset
dataset = create_rust_streaming_dataset(
    "data/train.parquet",
    batch_size=32
)

# Use with DataLoader (batch_size=None since dataset handles batching)
loader = DataLoader(
    dataset,
    batch_size=None,
    num_workers=0  # Rust handles parallelism
)

for batch in loader:
    model.train_step(batch)
```

## Performance Tips

### 1. Use Parquet Format

Parquet offers the best performance due to columnar storage and efficient compression:

```python
# Convert other formats to Parquet for best streaming performance
import pandas as pd

df = pd.read_json("data.json")
df.to_parquet("data.parquet")
```

### 2. Select Only Needed Columns

```python
# Faster - only loads needed columns
reader = streaming_dataset_reader(
    "data/train.parquet",
    columns=["input_ids", "labels"]
)

# Slower - loads all columns
reader = streaming_dataset_reader("data/train.parquet")
```

### 3. Use ZSTD Compression

ZSTD offers excellent compression with fast decompression:

```python
# Create ZSTD-compressed Parquet
import pyarrow.parquet as pq

pq.write_table(
    table,
    "data.parquet",
    compression="zstd"
)
```

### 4. Batch Size Tuning

| Dataset Size | Recommended Batch Size |
|--------------|----------------------|
| < 100K rows | 1000-5000 |
| 100K - 1M rows | 500-2000 |
| > 1M rows | 100-1000 |

## Error Handling

```python
from fast_axolotl import streaming_dataset_reader

try:
    for batch in streaming_dataset_reader("data/train.parquet"):
        process(batch)
except FileNotFoundError:
    print("Data file not found")
except ValueError as e:
    print(f"Format error: {e}")
```

## Comparison with Alternatives

| Feature | fast-axolotl | HuggingFace datasets | pandas |
|---------|--------------|---------------------|--------|
| Memory efficiency | Excellent | Good | Poor |
| Speed | 77x faster | 1x baseline | 0.5x |
| Format support | 7 formats | Many | Many |
| Compression | Auto | Manual | Manual |

## Next Steps

- [Token Packing](token-packing.md) - Efficiently pack streamed sequences
- [API Reference](../api-reference/streaming.md) - Complete streaming API docs
- [Benchmarks](../performance/benchmarks.md) - Detailed performance data
