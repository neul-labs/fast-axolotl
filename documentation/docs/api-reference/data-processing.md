# Data Processing API

This page documents the token packing, parallel hashing, and batch padding
functions. All are importable from `fast_axolotl` and require the Rust
extension - they raise `ImportError` when called without it.

---

## Token Packing

### `pack_sequences`

```python
def pack_sequences(
    sequences: List[List[int]],
    max_length: int,
    pad_token_id: int,
    eos_token_id: int,
    label_pad_id: int = -100,
) -> Dict[str, List[List[int]]]
```

Pack a list of token-ID sequences into fixed-length chunks. Returns a dict
with `input_ids`, `labels`, and `attention_mask` keys; every inner list has
length `max_length`.

```python
from fast_axolotl import pack_sequences

result = pack_sequences(
    sequences=[[1, 2, 3], [4, 5], [6, 7, 8, 9]],
    max_length=10,
    pad_token_id=0,
    eos_token_id=2,
)
```

### `concatenate_and_pack`

```python
def concatenate_and_pack(
    input_ids: List[List[int]],
    labels: List[List[int]],
    attention_masks: List[List[int]],
    max_length: int,
    pad_token_id: int,
    label_pad_id: int = -100,
) -> Dict[str, List[List[int]]]
```

Lower-level packing for when you already maintain `input_ids`, `labels`,
and `attention_masks` separately. Returns the same shape of dict as
`pack_sequences`.

---

## Parallel Hashing

### `parallel_hash_rows`

```python
def parallel_hash_rows(
    rows: List[str],
    num_threads: int = 0,
) -> List[str]
```

Compute SHA256 hashes for a list of rows in parallel. Returns lower-case
hex digests in the same order as the input.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `rows` | `List[str]` | required | rows to hash (UTF-8 bytes are hashed) |
| `num_threads` | `int` | `0` | `0` = auto-detect cores |

```python
hashes = parallel_hash_rows(["row1", "row2", "row3"], num_threads=0)
```

### `deduplicate_indices`

```python
def deduplicate_indices(
    rows: List[str],
    existing_hashes: Optional[List[str]] = None,
    num_threads: int = 0,
) -> Tuple[List[int], List[str]]
```

Return the indices of the first occurrence of each unique row, plus the
full hash list for `rows`. When `existing_hashes` is given, rows whose
hashes appear in it are also dropped.

```python
unique_idx, all_hashes = deduplicate_indices(["a", "b", "a", "c"])
# unique_idx == [0, 1, 3]
```

---

## Batch Padding

### `pad_sequences`

```python
def pad_sequences(
    sequences: List[List[int]],
    target_length: Optional[int] = None,
    pad_value: int = 0,
    padding_side: str = "right",
    pad_to_multiple_of: Optional[int] = None,
) -> List[List[int]]
```

Pad a batch of sequences to a uniform length.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `sequences` | `List[List[int]]` | required | sequences to pad |
| `target_length` | `Optional[int]` | `None` | pad to this length; `None` = batch max |
| `pad_value` | `int` | `0` | fill value |
| `padding_side` | `str` | `"right"` | `"right"` or `"left"` |
| `pad_to_multiple_of` | `Optional[int]` | `None` | round final length up to multiple |

```python
pad_sequences([[1, 2, 3], [4, 5]], target_length=5)
# [[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]]
```

### `create_padding_mask`

```python
def create_padding_mask(
    current_length: int,
    target_length: int,
) -> List[int]
```

Return the position IDs `[0, 1, 2, ...]` needed to pad a single sequence
from `current_length` up to `target_length`. The returned list has length
`target_length - current_length`.

```python
create_padding_mask(5, 8)   # [0, 1, 2]
```

---

## See also

- [Token Packing guide](../user-guide/token-packing.md)
- [Parallel Hashing guide](../user-guide/parallel-hashing.md)
- [Batch Padding guide](../user-guide/batch-padding.md)
