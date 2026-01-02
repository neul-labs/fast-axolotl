# Quick Start

Get up and running with fast-axolotl in minutes.

## Method 1: Auto-Shimming (Recommended)

The easiest way to use fast-axolotl is with automatic shimming. This transparently accelerates Axolotl without any code changes:

```python
import fast_axolotl

# Install acceleration shims
fast_axolotl.install()

# Now use Axolotl as normal - it's automatically accelerated!
from axolotl.utils.data import load_tokenized_prepared_datasets
# ... your training code
```

That's it! All compatible Axolotl operations now use Rust-accelerated implementations.

### Disabling Shimming

To temporarily disable acceleration:

```python
fast_axolotl.uninstall()
```

## Method 2: Direct API Usage

For more control, use the fast-axolotl functions directly:

### Streaming Data Loading

```python
from fast_axolotl import streaming_dataset_reader

# Stream from a Parquet file
for batch in streaming_dataset_reader("data/train.parquet", batch_size=1000):
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    # Process batch...
```

### With Glob Patterns

```python
# Load from multiple files
reader = streaming_dataset_reader(
    "data/**/*.parquet",
    batch_size=500,
    columns=["input_ids", "attention_mask", "labels"]
)

for batch in reader:
    process(batch)
```

### Token Packing

```python
from fast_axolotl import pack_sequences
import torch

# Your tokenized sequences
sequences = [
    torch.tensor([1, 2, 3]),
    torch.tensor([4, 5]),
    torch.tensor([6, 7, 8, 9]),
]

# Pack into fixed-length chunks
packed = pack_sequences(
    sequences,
    max_length=8,
    pad_token_id=0
)
# Result: tensor([[1, 2, 3, 4, 5, 0, 0, 0], [6, 7, 8, 9, 0, 0, 0, 0]])
```

### Parallel Deduplication

```python
from fast_axolotl import parallel_hash_rows, deduplicate_indices

# Your dataset rows (as bytes)
rows = [b"row1", b"row2", b"row1", b"row3", b"row2"]

# Get unique indices
unique_idx = deduplicate_indices(rows)
# Result: [0, 1, 3] - indices of unique rows
```

### Batch Padding

```python
from fast_axolotl import pad_sequences

sequences = [
    [1, 2, 3],
    [4, 5],
    [6, 7, 8, 9, 10],
]

# Pad to uniform length
padded = pad_sequences(
    sequences,
    pad_value=0,
    padding_side="right"
)
# All sequences now have length 5
```

## Method 3: HuggingFace Dataset Compatible

fast-axolotl provides a HuggingFace-compatible streaming dataset:

```python
from fast_axolotl import create_rust_streaming_dataset

# Create HF-compatible streaming dataset
dataset = create_rust_streaming_dataset(
    "data/train.parquet",
    batch_size=32
)

# Works with DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=None)  # batch_size handled by dataset

for batch in loader:
    model(batch)
```

## Complete Training Example

Here's a complete example integrating fast-axolotl with a training loop:

```python
import fast_axolotl
from fast_axolotl import (
    streaming_dataset_reader,
    pack_sequences,
    pad_sequences,
)
import torch

# Enable shimming for any axolotl code
fast_axolotl.install()

def train():
    # Stream training data
    for batch in streaming_dataset_reader("data/train.parquet", batch_size=32):
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # Pack sequences for efficient training
        packed_inputs = pack_sequences(input_ids, max_length=2048, pad_token_id=0)
        packed_labels = pack_sequences(labels, max_length=2048, pad_token_id=-100)

        # Your training step here
        loss = model(packed_inputs, labels=packed_labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

if __name__ == "__main__":
    train()
```

## Checking What's Accelerated

To see what functions are using Rust acceleration:

```python
import fast_axolotl

# Check if Rust is available
print(f"Rust available: {fast_axolotl.rust_available()}")

# List supported formats
print(f"Supported formats: {fast_axolotl.list_supported_formats()}")

# After installing shims
fast_axolotl.install()
print("Shims installed - Axolotl is now accelerated")
```

## Next Steps

- [Streaming Data Guide](../user-guide/streaming.md) - Deep dive into streaming
- [Token Packing Guide](../user-guide/token-packing.md) - Efficient sequence packing
- [API Reference](../api-reference/core.md) - Complete function reference
- [Benchmarks](../performance/benchmarks.md) - Performance comparisons
