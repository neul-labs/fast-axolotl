# Best Practices

This guide covers optimization strategies for getting the most performance from fast-axolotl.

## Data Format Selection

### Use Parquet for Best Performance

Parquet provides the best streaming performance due to:

- Columnar storage (only read needed columns)
- Efficient compression (ZSTD recommended)
- Row group organization for batch reading

```python
# Convert to Parquet with optimal settings
import pyarrow as pa
import pyarrow.parquet as pq

table = pa.Table.from_pandas(df)
pq.write_table(
    table,
    "data.parquet",
    compression="zstd",
    row_group_size=10000  # Tune based on batch_size
)
```

### Format Decision Tree

```
Is your data structured?
├── Yes → Use Parquet
└── No (raw text) → Use JSONL with ZSTD compression
```

---

## Streaming Optimization

### Batch Size Tuning

| Memory Available | Recommended Batch Size |
|-----------------|----------------------|
| < 8 GB | 100-500 |
| 8-16 GB | 500-2000 |
| 16-32 GB | 1000-5000 |
| > 32 GB | 2000-10000 |

```python
# Start conservative, increase if memory allows
for batch_size in [500, 1000, 2000, 5000]:
    try:
        for batch in streaming_dataset_reader(path, batch_size=batch_size):
            process(batch)
        print(f"batch_size={batch_size} works")
        break
    except MemoryError:
        print(f"batch_size={batch_size} too large")
```

### Column Selection

Always specify only the columns you need:

```python
# Good - only loads needed columns
reader = streaming_dataset_reader(
    "data.parquet",
    columns=["input_ids", "attention_mask", "labels"]
)

# Bad - loads all columns including unused ones
reader = streaming_dataset_reader("data.parquet")
```

### File Organization

For large datasets, split into multiple files:

```
data/
├── train_00000.parquet  # ~100MB each
├── train_00001.parquet
├── train_00002.parquet
└── ...
```

Benefits:

- Parallel file processing
- Better memory management
- Easier data versioning

---

## Token Packing Optimization

### When to Use Packing

Use packing when:

- Average sequence length < 50% of max length
- High variance in sequence lengths
- Training on concatenated documents

Skip packing when:

- Sequences are already near max length
- Uniform sequence lengths
- Very small batch sizes

### Packing Strategy

```python
from fast_axolotl import pack_sequences

def efficient_packing(sequences, max_length, pad_token_id):
    # Sort by length for better packing efficiency
    sorted_seqs = sorted(sequences, key=len)

    # Pack
    packed = pack_sequences(sorted_seqs, max_length, pad_token_id)

    return packed
```

### Memory-Efficient Packing

For large datasets, pack in chunks:

```python
def chunked_packing(sequences, max_length, pad_token_id, chunk_size=10000):
    all_packed = []

    for i in range(0, len(sequences), chunk_size):
        chunk = sequences[i:i+chunk_size]
        packed = pack_sequences(chunk, max_length, pad_token_id)
        all_packed.append(packed)

    return torch.cat(all_packed, dim=0)
```

---

## Parallel Hashing Optimization

### Optimal Row Size

Parallel hashing works best with:

- Row sizes between 100 bytes and 10 KB
- Large number of rows (1000+)

```python
# For very small rows, batch them
def batch_hash(items, items_per_row=10):
    batched = [
        b"".join(items[i:i+items_per_row])
        for i in range(0, len(items), items_per_row)
    ]
    return parallel_hash_rows(batched)
```

### Streaming Deduplication

For datasets too large for memory:

```python
def streaming_dedupe(path, output_path, chunk_size=100000):
    seen_hashes = set()

    with open(output_path, "w") as out:
        for batch in streaming_dataset_reader(path, batch_size=chunk_size):
            rows = [str(r).encode() for r in batch["text"]]
            hashes = parallel_hash_rows(rows)

            for i, h in enumerate(hashes):
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    out.write(batch["text"][i] + "\n")
```

---

## Batch Padding Optimization

### Dynamic vs Static Batching

**Dynamic batching** (pad to longest in batch):

- Less wasted compute
- Variable memory usage
- Slightly more overhead

```python
# Dynamic - default behavior
padded = pad_sequences(batch, pad_value=0)
```

**Static batching** (pad to fixed length):

- Consistent memory usage
- Better for caching
- May waste compute on short sequences

```python
# Static - fixed max_length
padded = pad_sequences(batch, pad_value=0, max_length=2048)
```

### Length-Sorted Batching

Minimize padding by sorting sequences by length:

```python
def create_sorted_batches(sequences, batch_size, pad_value):
    # Sort by length
    sorted_idx = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    sorted_seqs = [sequences[i] for i in sorted_idx]

    # Create batches
    batches = []
    for i in range(0, len(sorted_seqs), batch_size):
        batch = sorted_seqs[i:i+batch_size]
        padded = pad_sequences(batch, pad_value=pad_value)
        batches.append(padded)

    return batches
```

---

## Memory Management

### Reduce Peak Memory

```python
import gc

def process_large_dataset(path):
    for batch in streaming_dataset_reader(path, batch_size=1000):
        result = process(batch)
        save(result)

        # Explicitly free batch memory
        del batch
        gc.collect()
```

### Monitor Memory Usage

```python
import psutil

def memory_efficient_processing(path):
    process = psutil.Process()

    for batch in streaming_dataset_reader(path):
        mem_before = process.memory_info().rss / 1024 / 1024

        result = process(batch)

        mem_after = process.memory_info().rss / 1024 / 1024
        if mem_after > mem_before + 100:  # More than 100MB growth
            gc.collect()
```

---

## Integration Patterns

### With PyTorch DataLoader

```python
from torch.utils.data import DataLoader
from fast_axolotl import create_rust_streaming_dataset

# Optimal DataLoader settings for fast-axolotl
loader = DataLoader(
    create_rust_streaming_dataset("data.parquet", batch_size=32),
    batch_size=None,      # Dataset handles batching
    num_workers=0,        # Rust handles parallelism
    pin_memory=True,      # Fast GPU transfer
    prefetch_factor=None  # Not needed with streaming
)
```

### With HuggingFace Trainer

```python
import fast_axolotl
from transformers import Trainer

# Enable shimming before creating trainer
fast_axolotl.install()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    # fast-axolotl accelerates internal operations
)
```

---

## Profiling

### Identify Bottlenecks

```python
import time

def profile_pipeline(path):
    timings = {}

    # Profile streaming
    start = time.time()
    data = list(streaming_dataset_reader(path, batch_size=1000))
    timings["streaming"] = time.time() - start

    # Profile packing
    start = time.time()
    packed = pack_sequences(data, max_length=2048, pad_token_id=0)
    timings["packing"] = time.time() - start

    # Profile padding
    start = time.time()
    padded = pad_sequences(data, pad_value=0)
    timings["padding"] = time.time() - start

    return timings
```

### Compare With and Without fast-axolotl

```python
def compare_performance():
    import fast_axolotl

    # With fast-axolotl
    fast_axolotl.install()
    start = time.time()
    run_pipeline()
    with_fa = time.time() - start

    # Without fast-axolotl
    fast_axolotl.uninstall()
    start = time.time()
    run_pipeline()
    without_fa = time.time() - start

    print(f"Speedup: {without_fa / with_fa:.1f}x")
```

---

## Common Pitfalls

### 1. Not Using Shimming

```python
# Wrong - import axolotl before install()
import axolotl
import fast_axolotl
fast_axolotl.install()  # Too late!

# Right - install before importing axolotl
import fast_axolotl
fast_axolotl.install()
import axolotl
```

### 2. Loading All Columns

```python
# Wrong - loads unused columns
reader = streaming_dataset_reader("data.parquet")

# Right - specify needed columns
reader = streaming_dataset_reader("data.parquet", columns=["input_ids", "labels"])
```

### 3. Small Batch Sizes

```python
# Wrong - too small, high overhead per batch
reader = streaming_dataset_reader("data.parquet", batch_size=10)

# Right - larger batches for better throughput
reader = streaming_dataset_reader("data.parquet", batch_size=1000)
```

---

## See Also

- [Benchmarks](benchmarks.md) - Performance comparisons
- [Streaming Guide](../user-guide/streaming.md) - Detailed streaming usage
- [API Reference](../api-reference/core.md) - Complete API documentation
