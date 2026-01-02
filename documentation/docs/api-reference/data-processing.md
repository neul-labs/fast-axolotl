# Data Processing API

This page documents the data processing functions: token packing, parallel hashing, and batch padding.

---

## Token Packing

### `pack_sequences()`

Pack multiple sequences into fixed-length chunks.

```python
fast_axolotl.pack_sequences(
    sequences: list[list[int]] | list[torch.Tensor],
    max_length: int,
    pad_token_id: int,
    return_boundaries: bool = False
) -> torch.Tensor | tuple[torch.Tensor, list]
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sequences` | `list` | required | List of token sequences |
| `max_length` | `int` | required | Target length for packed sequences |
| `pad_token_id` | `int` | required | Token ID used for padding |
| `return_boundaries` | `bool` | `False` | Also return sequence boundaries |

**Returns**:

- If `return_boundaries=False`: A tensor of shape `(num_packed, max_length)`
- If `return_boundaries=True`: A tuple of (tensor, boundaries list)

**Example**:
```python
from fast_axolotl import pack_sequences
import torch

sequences = [
    torch.tensor([1, 2, 3]),
    torch.tensor([4, 5]),
    torch.tensor([6, 7, 8, 9]),
]

packed = pack_sequences(sequences, max_length=8, pad_token_id=0)
print(packed)
# tensor([[1, 2, 3, 4, 5, 0, 0, 0],
#         [6, 7, 8, 9, 0, 0, 0, 0]])
```

---

### `concatenate_and_pack()`

Pack with separate input, label, and attention mask sequences.

```python
fast_axolotl.concatenate_and_pack(
    input_ids: list[list[int]],
    labels: list[list[int]],
    attention_masks: list[list[int]],
    max_length: int,
    pad_token_id: int,
    label_pad_id: int = -100
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_ids` | `list[list[int]]` | required | Input token sequences |
| `labels` | `list[list[int]]` | required | Label sequences |
| `attention_masks` | `list[list[int]]` | required | Attention mask sequences |
| `max_length` | `int` | required | Target length |
| `pad_token_id` | `int` | required | Padding token for inputs |
| `label_pad_id` | `int` | `-100` | Padding token for labels |

**Returns**: Tuple of (packed_inputs, packed_labels, packed_masks) tensors.

**Example**:
```python
from fast_axolotl import concatenate_and_pack

inputs = [[1, 2, 3], [4, 5, 6]]
labels = [[-100, 2, 3], [-100, -100, 6]]
masks = [[1, 1, 1], [1, 1, 1]]

packed_in, packed_lab, packed_mask = concatenate_and_pack(
    inputs, labels, masks,
    max_length=8,
    pad_token_id=0,
    label_pad_id=-100
)
```

---

## Parallel Hashing

### `parallel_hash_rows()`

Compute SHA256 hashes of byte sequences in parallel.

```python
fast_axolotl.parallel_hash_rows(
    rows: list[bytes]
) -> list[str]
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `rows` | `list[bytes]` | List of byte sequences to hash |

**Returns**: List of hex-encoded SHA256 hash strings.

**Example**:
```python
from fast_axolotl import parallel_hash_rows

rows = [
    b"First document content",
    b"Second document content",
    b"Third document content",
]

hashes = parallel_hash_rows(rows)
print(hashes[0])  # 64-character hex string
```

**Performance**: Uses all available CPU cores for parallel hashing.

---

### `deduplicate_indices()`

Find indices of unique rows using parallel hashing.

```python
fast_axolotl.deduplicate_indices(
    rows: list[bytes]
) -> list[int]
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `rows` | `list[bytes]` | List of byte sequences |

**Returns**: List of indices pointing to the first occurrence of each unique row.

**Example**:
```python
from fast_axolotl import deduplicate_indices

rows = [
    b"document A",
    b"document B",
    b"document A",  # duplicate
    b"document C",
    b"document B",  # duplicate
]

unique_idx = deduplicate_indices(rows)
print(unique_idx)  # [0, 1, 3]
```

---

## Batch Padding

### `pad_sequences()`

Pad sequences to uniform length.

```python
fast_axolotl.pad_sequences(
    sequences: list[list[int]],
    pad_value: int,
    padding_side: str = "right",
    max_length: int | None = None
) -> list[list[int]]
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sequences` | `list[list[int]]` | required | Sequences to pad |
| `pad_value` | `int` | required | Value used for padding |
| `padding_side` | `str` | `"right"` | `"left"` or `"right"` |
| `max_length` | `int` | `None` | Target length (None = longest) |

**Returns**: List of padded sequences (all same length).

**Example**:
```python
from fast_axolotl import pad_sequences

sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]

# Pad to longest
padded = pad_sequences(sequences, pad_value=0)
print(padded)
# [[1, 2, 3, 0], [4, 5, 0, 0], [6, 7, 8, 9]]

# Pad with specific length
padded = pad_sequences(sequences, pad_value=0, max_length=6)
print(padded)
# [[1, 2, 3, 0, 0, 0], [4, 5, 0, 0, 0, 0], [6, 7, 8, 9, 0, 0]]

# Left padding
padded = pad_sequences(sequences, pad_value=0, padding_side="left")
print(padded)
# [[0, 1, 2, 3], [0, 0, 4, 5], [6, 7, 8, 9]]
```

---

### `create_padding_mask()`

Create attention masks for padded sequences.

```python
fast_axolotl.create_padding_mask(
    padded_sequences: list[list[int]],
    pad_value: int
) -> list[list[int]]
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `padded_sequences` | `list[list[int]]` | Padded sequences |
| `pad_value` | `int` | The padding value to detect |

**Returns**: Attention masks (1 for real tokens, 0 for padding).

**Example**:
```python
from fast_axolotl import pad_sequences, create_padding_mask

sequences = [[1, 2, 3], [4, 5]]
padded = pad_sequences(sequences, pad_value=0)

mask = create_padding_mask(padded, pad_value=0)
print(mask)
# [[1, 1, 1], [1, 1, 0]]
```

---

## Quick Reference

### Token Packing

| Function | Description |
|----------|-------------|
| `pack_sequences()` | Pack sequences into fixed-length chunks |
| `concatenate_and_pack()` | Pack with inputs, labels, masks |

### Hashing & Deduplication

| Function | Description |
|----------|-------------|
| `parallel_hash_rows()` | Parallel SHA256 hashing |
| `deduplicate_indices()` | Find unique row indices |

### Padding

| Function | Description |
|----------|-------------|
| `pad_sequences()` | Pad to uniform length |
| `create_padding_mask()` | Create attention masks |

---

## Type Annotations

```python
from typing import List, Tuple, Optional, Union
import torch

# Token packing
def pack_sequences(
    sequences: Union[List[List[int]], List[torch.Tensor]],
    max_length: int,
    pad_token_id: int,
    return_boundaries: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]: ...

# Hashing
def parallel_hash_rows(rows: List[bytes]) -> List[str]: ...
def deduplicate_indices(rows: List[bytes]) -> List[int]: ...

# Padding
def pad_sequences(
    sequences: List[List[int]],
    pad_value: int,
    padding_side: str = "right",
    max_length: Optional[int] = None
) -> List[List[int]]: ...
```

---

## See Also

- [Token Packing Guide](../user-guide/token-packing.md) - Usage patterns
- [Parallel Hashing Guide](../user-guide/parallel-hashing.md) - Deduplication workflows
- [Batch Padding Guide](../user-guide/batch-padding.md) - Padding strategies
