# Batch Padding

`fast-axolotl` provides two Rust-backed padding utilities:

- `pad_sequences` - pad a batch of variable-length sequences to a uniform length
- `create_padding_mask` - return the position IDs needed to extend a single
  sequence

## `pad_sequences`

```python
from fast_axolotl import pad_sequences

padded = pad_sequences(
    [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]],
    target_length=8,
    pad_value=0,
    padding_side="right",
)
# [[1, 2, 3, 0, 0, 0, 0, 0],
#  [4, 5, 0, 0, 0, 0, 0, 0],
#  [6, 7, 8, 9, 10, 0, 0, 0]]
```

### Signature

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `sequences` | `List[List[int]]` | required | sequences to pad |
| `target_length` | `Optional[int]` | `None` | pad to this length; `None` = max length in batch |
| `pad_value` | `int` | `0` | value used to fill |
| `padding_side` | `str` | `"right"` | `"right"` or `"left"` |
| `pad_to_multiple_of` | `Optional[int]` | `None` | round target length up to a multiple of this value (useful for tensor-core alignment) |

### Left-side padding

```python
pad_sequences(
    [[1, 2, 3], [4, 5]],
    target_length=8,
    pad_value=0,
    padding_side="left",
)
# [[0, 0, 0, 0, 0, 1, 2, 3],
#  [0, 0, 0, 0, 0, 0, 4, 5]]
```

### Pad to a multiple

Hardware kernels (FlashAttention, tensor cores) often prefer sequence
lengths that are multiples of 8, 16, or 64:

```python
pad_sequences(
    sequences,
    pad_value=0,
    pad_to_multiple_of=8,
)
```

If `target_length` is also given, the final length is
`max(target_length, ceil(target_length / multiple) * multiple)`.

## `create_padding_mask`

Helper that returns the `[0, 1, 2, ...]` position IDs you need when
extending one sequence:

```python
from fast_axolotl import create_padding_mask

mask = create_padding_mask(current_length=5, target_length=8)
# [0, 1, 2]   # three padding positions
```

## Axolotl integration

The shim installs the following on `axolotl.utils.collators`:

| Shimmed attribute | Backed by |
|---|---|
| `axolotl.utils.collators.fast_pad_sequences` | `pad_sequences` |
| `axolotl.utils.collators.fast_create_padding_mask` | `create_padding_mask` |

Axolotl collators that look up these names will use the Rust
implementations automatically once `fast_axolotl` has been imported.

## When to reach for it

At small batch sizes plain Python list padding can be faster - the
benchmark shows a 0.53x ratio on 10,000 sequences because the FFI cost
dominates. The Rust path becomes worthwhile when:

- batches are large (hundreds of sequences)
- sequences are long (thousands of tokens)
- padding is on the hot training loop

See [Best Practices](../performance/best-practices.md) for guidance.

## See also

- [Token Packing](token-packing.md)
- [Data Processing API](../api-reference/data-processing.md)
