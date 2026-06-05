# Token Packing

Token packing concatenates variable-length sequences into fixed-length
chunks so the GPU sees fewer (but fuller) examples per step. `fast-axolotl`
provides two pack functions, both implemented in Rust:

- [`pack_sequences`](#pack_sequences) - the common case: turn a list of
  token-ID lists into `input_ids` / `labels` / `attention_mask`
- [`concatenate_and_pack`](#concatenate_and_pack) - lower-level form when
  you already have separate `input_ids`, `labels`, and `attention_masks`

Both functions are exported directly from `fast_axolotl`.

## `pack_sequences`

```python
from fast_axolotl import pack_sequences

result = pack_sequences(
    sequences=[[1, 2, 3], [4, 5], [6, 7, 8, 9]],
    max_length=10,
    pad_token_id=0,
    eos_token_id=2,
    label_pad_id=-100,   # default
)

result.keys()
# dict_keys(['input_ids', 'labels', 'attention_mask'])
```

### Signature

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `sequences` | `List[List[int]]` | required | token-ID lists to pack |
| `max_length` | `int` | required | length of each packed chunk |
| `pad_token_id` | `int` | required | pad value for `input_ids` |
| `eos_token_id` | `int` | required | end-of-sequence marker between concatenated sequences |
| `label_pad_id` | `int` | `-100` | pad value for `labels` (kept out of the loss) |

The return value is a dict of three equal-length `List[List[int]]`s where
every inner list has length exactly `max_length`.

## `concatenate_and_pack`

Use this when you already have parallel `input_ids`, `labels`, and
`attention_masks` (for example after a custom tokenizer step):

```python
from fast_axolotl import concatenate_and_pack

packed = concatenate_and_pack(
    input_ids=[[1, 2, 3], [4, 5]],
    labels=[[1, 2, 3], [4, 5]],
    attention_masks=[[1, 1, 1], [1, 1]],
    max_length=10,
    pad_token_id=0,
    label_pad_id=-100,
)
```

The output has the same `input_ids` / `labels` / `attention_mask` keys as
`pack_sequences`.

## When packing helps

Packing is a wash on small toy datasets - the benchmark in the repo shows
roughly 0.4x speedup on 10,000 sequences because the Python <-> Rust
boundary cost dominates. The wins show up at production scale:

- millions of tokens
- sequence lengths in the thousands
- pre-allocated buffers and cache-friendly memory layout matter

See [Benchmarks](../performance/benchmarks.md) for the underlying numbers
and [Best Practices](../performance/best-practices.md) for guidance on when
to reach for packing vs plain padding.

## Integration with Axolotl

Axolotl drives its own packing decisions, so `pack_sequences` is not
shimmed automatically. Use it directly inside custom collators or data
prep scripts:

```python
from fast_axolotl import pack_sequences

def collate(batch):
    return pack_sequences(
        sequences=[ex["input_ids"] for ex in batch],
        max_length=2048,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
```

## See also

- [Batch Padding](batch-padding.md) - simpler padding without concatenation
- [Data Processing API](../api-reference/data-processing.md)
