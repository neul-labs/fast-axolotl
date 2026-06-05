# Parallel Hashing and Deduplication

`fast-axolotl` ships two Rust functions for multi-threaded SHA256
deduplication:

- `parallel_hash_rows` - hash a list of strings in parallel
- `deduplicate_indices` - return the indices of the first occurrence of
  each unique row

The repo benchmark shows a **1.9x speedup** over Python's `hashlib` at
100,000 rows on 16 cores.

## `parallel_hash_rows`

```python
from fast_axolotl import parallel_hash_rows

rows = [str(row) for row in dataset]
hashes = parallel_hash_rows(rows, num_threads=0)  # 0 = auto-detect cores
# ['d4735e3a265e16...', 'b2d2226c48a9bd...', ...]
```

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `rows` | `List[str]` | required | string-encoded rows to hash |
| `num_threads` | `int` | `0` | worker thread count, `0` = auto |

Output order matches input order. Each hash is the lowercase hex SHA256
digest of the row's UTF-8 bytes - byte-identical to Python's
`hashlib.sha256(row.encode()).hexdigest()`.

## `deduplicate_indices`

```python
from fast_axolotl import deduplicate_indices

rows = ["a", "b", "a", "c"]
unique_indices, all_hashes = deduplicate_indices(rows)
# unique_indices == [0, 1, 3]
# all_hashes is the per-row SHA256 list, same order as `rows`
```

### Filtering against previously seen data

Pass the hashes you've already seen as `existing_hashes` to skip rows that
match anything in your prior set:

```python
existing = load_previous_hashes()
unique_idx, new_hashes = deduplicate_indices(
    rows,
    existing_hashes=existing,
    num_threads=8,
)
```

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `rows` | `List[str]` | required | rows to dedupe |
| `existing_hashes` | `Optional[List[str]]` | `None` | hashes to filter against |
| `num_threads` | `int` | `0` | `0` = auto-detect cores |

## Encoding rows correctly

The hash treats your row as a string, so the encoding decides equality.
Pick a stable, canonical form:

```python
import json

row_strings = [
    json.dumps(row, sort_keys=True, separators=(",", ":"))
    for row in dataset
]
unique_idx, _ = deduplicate_indices(row_strings)
```

`sort_keys=True` is the part that matters - without it two semantically
identical dicts can hash differently because of insertion order.

## Axolotl integration

When the shim is installed (`import fast_axolotl` runs first), the
following names are installed on `axolotl.utils.data` as drop-in
replacements:

| Shimmed attribute | Backed by |
|---|---|
| `axolotl.utils.data.fast_parallel_hash_rows` | `parallel_hash_rows` |
| `axolotl.utils.data.fast_deduplicate_indices` | `deduplicate_indices` |

Enable Axolotl's dedupe in your YAML config and it will use the Rust path
automatically:

```yaml
dedupe: true
```

## See also

- [Auto-Shimming](shimming.md)
- [Data Processing API](../api-reference/data-processing.md)
- [Benchmarks](../performance/benchmarks.md)
