# Parallel Hashing & Deduplication

fast-axolotl provides multi-threaded SHA256 hashing for efficient dataset deduplication, achieving **1.9x speedup** over Python's hashlib.

## Why Parallel Hashing?

Dataset deduplication is crucial for LLM training:

- Removes exact duplicates that can cause overfitting
- Reduces training time and costs
- Improves model generalization

The bottleneck is usually hashing millions of rows - fast-axolotl parallelizes this across all CPU cores.

## Basic Usage

### Computing Hashes

```python
from fast_axolotl import parallel_hash_rows

# Your data rows (as bytes)
rows = [
    b"This is the first document",
    b"This is the second document",
    b"This is the first document",  # duplicate
    b"This is the third document",
]

# Compute SHA256 hashes in parallel
hashes = parallel_hash_rows(rows)

print(hashes)
# ['a1b2c3...', 'd4e5f6...', 'a1b2c3...', 'g7h8i9...']
```

### Finding Unique Indices

```python
from fast_axolotl import deduplicate_indices

rows = [
    b"row1",
    b"row2",
    b"row1",  # duplicate of index 0
    b"row3",
    b"row2",  # duplicate of index 1
]

# Get indices of unique rows
unique_idx = deduplicate_indices(rows)

print(unique_idx)
# [0, 1, 3] - first occurrence of each unique row
```

## Working with Datasets

### Deduplicating a HuggingFace Dataset

```python
from datasets import load_dataset
from fast_axolotl import deduplicate_indices

# Load dataset
dataset = load_dataset("your_dataset")

# Convert rows to bytes for hashing
rows = [str(row).encode() for row in dataset["train"]]

# Find unique indices
unique_idx = deduplicate_indices(rows)

# Filter dataset
deduped_dataset = dataset["train"].select(unique_idx)

print(f"Original: {len(dataset['train'])}, Deduped: {len(deduped_dataset)}")
```

### Deduplicating by Specific Columns

```python
from fast_axolotl import deduplicate_indices
import json

def deduplicate_by_columns(dataset, columns):
    """Deduplicate based on specific columns only."""

    # Create hash keys from selected columns
    rows = []
    for item in dataset:
        key = json.dumps({col: item[col] for col in columns}).encode()
        rows.append(key)

    unique_idx = deduplicate_indices(rows)
    return dataset.select(unique_idx)

# Deduplicate by 'text' column only
deduped = deduplicate_by_columns(dataset, ["text"])
```

### Streaming Deduplication

For very large datasets that don't fit in memory:

```python
from fast_axolotl import streaming_dataset_reader, parallel_hash_rows

def streaming_deduplicate(data_path, output_path):
    seen_hashes = set()
    unique_rows = []

    for batch in streaming_dataset_reader(data_path, batch_size=10000):
        # Convert batch to bytes
        rows = [str(row).encode() for row in batch["text"]]

        # Hash the batch
        hashes = parallel_hash_rows(rows)

        # Keep only new unique rows
        for i, h in enumerate(hashes):
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_rows.append(batch[i])

    return unique_rows
```

## Advanced Usage

### Custom Hash Function

While fast-axolotl uses SHA256 by default, you can preprocess data for different deduplication strategies:

```python
from fast_axolotl import deduplicate_indices

def normalize_and_dedupe(texts):
    """Deduplicate with normalization."""

    # Normalize: lowercase, strip whitespace
    normalized = [
        text.lower().strip().encode()
        for text in texts
    ]

    return deduplicate_indices(normalized)
```

### Fuzzy Deduplication Preparation

For near-duplicate detection, use hashing as a first pass:

```python
from fast_axolotl import parallel_hash_rows

def find_candidate_duplicates(texts, n_shingles=3):
    """Find candidate near-duplicates using shingling."""

    all_shingles = []
    for text in texts:
        # Create character n-grams
        words = text.split()
        shingles = [
            " ".join(words[i:i+n_shingles]).encode()
            for i in range(len(words) - n_shingles + 1)
        ]
        all_shingles.append(shingles)

    # Hash all shingles in parallel
    flat_shingles = [s for shingles in all_shingles for s in shingles]
    hashes = parallel_hash_rows(flat_shingles)

    # Group by common shingles for further analysis
    # ...
```

## Performance Benchmarks

| Dataset Size | Python hashlib | fast-axolotl | Speedup |
|--------------|---------------|--------------|---------|
| 10,000 rows | 0.5s | 0.3s | 1.7x |
| 100,000 rows | 5.2s | 2.7s | 1.9x |
| 1,000,000 rows | 52s | 27s | 1.9x |

### Thread Scaling

fast-axolotl automatically uses all available CPU cores:

| Cores | Speedup vs Single Thread |
|-------|-------------------------|
| 4 | 3.2x |
| 8 | 5.8x |
| 16 | 9.1x |
| 32 | 14.2x |

## Integration with Axolotl

When shimming is enabled, Axolotl's deduplication automatically uses fast-axolotl:

```python
import fast_axolotl
fast_axolotl.install()

# Axolotl's dedupe now uses Rust-accelerated hashing
from axolotl.utils.data import deduplicate_dataset
```

## Memory Considerations

### Hashes

Each SHA256 hash is 64 characters (hex string). For 1M rows:
- Memory for hashes: ~64MB

### Indices

The `deduplicate_indices` function returns a list of integers:
- Memory for indices: ~8 bytes per unique row

### Tips for Large Datasets

```python
# Process in chunks to limit memory
def chunked_dedupe(rows, chunk_size=100000):
    all_unique = []
    seen_hashes = set()

    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i+chunk_size]
        hashes = parallel_hash_rows(chunk)

        for j, h in enumerate(hashes):
            if h not in seen_hashes:
                seen_hashes.add(h)
                all_unique.append(i + j)

    return all_unique
```

## Error Handling

```python
from fast_axolotl import parallel_hash_rows, deduplicate_indices

# Empty input handling
hashes = parallel_hash_rows([])  # Returns []
indices = deduplicate_indices([])  # Returns []

# Invalid input
try:
    hashes = parallel_hash_rows(["string", "not", "bytes"])
except TypeError as e:
    print(f"Error: {e}")  # Rows must be bytes
```

## Next Steps

- [Streaming Data](streaming.md) - Combine with streaming for large datasets
- [API Reference](../api-reference/data-processing.md) - Complete API docs
- [Benchmarks](../performance/benchmarks.md) - Detailed performance data
