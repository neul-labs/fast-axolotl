# Best Practices

How to get the most out of `fast-axolotl` in real pipelines. The
guidance below comes from the project benchmarks and the structure of the
Rust extension itself.

## When to reach for fast-axolotl

The Rust accelerations win in proportion to two things: the amount of data
processed per call, and how many of those calls happen in your hot loop.

| Symptom | Reach for |
|---|---|
| `datasets.load_dataset` dominates training time | streaming reader |
| Deduping >100K rows in Python | `parallel_hash_rows` / `deduplicate_indices` |
| Custom collator with hand-rolled packing | `pack_sequences` / `concatenate_and_pack` |
| Long-sequence padding shows up in profiler | `pad_sequences` |
| You don't want to touch Axolotl source | install the shim, you're done |

For small data (<10K rows, <1K sequence length), the Python baselines are
often fine. The shim has no overhead in that case - it just makes the
faster paths available when Axolotl chooses them.

## Streaming

### Prefer Parquet with ZSTD

The Rust reader is fastest on Parquet thanks to columnar layout and
predicate-friendly compression:

```python
import pyarrow.parquet as pq
pq.write_table(table, "data.parquet", compression="zstd")
```

ZSTD decodes quickly and shrinks the dataset enough that I/O is rarely the
bottleneck.

### Tune `batch_size` to your memory budget

| Dataset rows | Suggested batch_size |
|---|---|
| <100K | 1,000-5,000 |
| 100K-1M | 500-2,000 |
| >1M | 100-1,000 |

Larger batches improve throughput but each one is held in Python memory
between yields.

### Let `num_threads` default to 4

The reader uses a Tokio multi-thread runtime; `num_threads=4` is a good
baseline. Raise it when you have many fast cores and fast storage; lower
it when you're competing with other workers.

### Use the config-driven helper inside Axolotl

`should_use_rust_streaming(cfg, file_size_bytes)` lets the package decide
whether streaming is worth it for a given file. Plumb it in once and avoid
hand-rolling the "is this big enough to stream?" check.

## Deduplication

### Pick a stable row encoding

The hash treats your row as a UTF-8 string, so equality is byte-equality:

```python
import json
rows = [
    json.dumps(r, sort_keys=True, separators=(",", ":"))
    for r in dataset
]
```

`sort_keys=True` is the part that matters - without it semantically equal
dicts can hash differently.

### Cache prior hashes across runs

Use the `existing_hashes` argument to `deduplicate_indices` to filter out
rows that match anything from a previous dataset version:

```python
unique_idx, new_hashes = deduplicate_indices(
    rows,
    existing_hashes=previously_seen,
    num_threads=0,
)
```

Persist `new_hashes` next to your dataset and you have an incremental
dedupe pipeline.

## Token packing and padding

### Reach for them at scale, not at toy sizes

The benchmark shows token packing and batch padding underperforming the
Python baseline at 10K sequences (0.4x and 0.5x). The Rust paths win when
your batches are large and your sequences are long; otherwise the FFI
overhead dominates.

If you're not sure, profile first.

### Align to hardware multiples

Tensor cores and FlashAttention prefer lengths divisible by 8, 16, or 64.
Pass `pad_to_multiple_of` to `pad_sequences` to bake that in:

```python
padded = pad_sequences(seqs, target_length=2048, pad_to_multiple_of=8)
```

### Keep `label_pad_id` at `-100`

PyTorch's `CrossEntropyLoss` defaults to ignoring `-100`. Override only if
you have a custom loss function.

## Import order matters

The shim only patches modules that haven't been imported yet:

```python
import fast_axolotl   # do this first
import axolotl        # then this
```

If you must import in the other order, call `fast_axolotl.install()`
afterwards to overwrite the existing entries.

## Verify the shim is active

```python
import fast_axolotl, sys
assert fast_axolotl.is_available()
assert getattr(
    sys.modules["axolotl.utils.data.rust_streaming"],
    "__fast_axolotl_shimmed__",
    False,
)
```

Wrap that in a startup check so you catch unaccelerated runs in CI before
they hit production.

## See also

- [Benchmarks](benchmarks.md)
- [Auto-Shimming](../user-guide/shimming.md)
- [Streaming guide](../user-guide/streaming.md)
