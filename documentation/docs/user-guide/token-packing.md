# Token Packing

Token packing efficiently combines multiple short sequences into longer ones, maximizing GPU utilization during training.

## Why Token Packing?

Without packing, short sequences waste compute:

```
Sequence 1: [token, token, token, PAD, PAD, PAD, PAD, PAD]  # 62% padding
Sequence 2: [token, token, PAD, PAD, PAD, PAD, PAD, PAD]    # 75% padding
```

With packing:

```
Packed: [token, token, token, SEP, token, token, PAD, PAD]  # 25% padding
```

## Basic Usage

```python
from fast_axolotl import pack_sequences
import torch

sequences = [
    torch.tensor([1, 2, 3]),
    torch.tensor([4, 5]),
    torch.tensor([6, 7, 8, 9]),
    torch.tensor([10, 11, 12]),
]

packed = pack_sequences(
    sequences,
    max_length=8,
    pad_token_id=0
)

print(packed)
# tensor([[ 1,  2,  3,  4,  5,  0,  0,  0],
#         [ 6,  7,  8,  9, 10, 11, 12,  0]])
```

## Configuration Options

### Max Length

Set the target sequence length:

```python
# For models with 2048 context
packed = pack_sequences(sequences, max_length=2048, pad_token_id=0)

# For models with 4096 context
packed = pack_sequences(sequences, max_length=4096, pad_token_id=0)
```

### Pad Token ID

Specify the padding token for your tokenizer:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

packed = pack_sequences(
    sequences,
    max_length=2048,
    pad_token_id=tokenizer.pad_token_id
)
```

## Advanced: Concatenate and Pack

For more control, use `concatenate_and_pack` with separate inputs, labels, and masks:

```python
from fast_axolotl import concatenate_and_pack

# Separate input_ids and labels
input_sequences = [
    [1, 2, 3],
    [4, 5, 6, 7],
]

label_sequences = [
    [-100, 2, 3],      # -100 = ignore in loss
    [-100, -100, 6, 7],
]

attention_masks = [
    [1, 1, 1],
    [1, 1, 1, 1],
]

packed_inputs, packed_labels, packed_masks = concatenate_and_pack(
    input_sequences,
    label_sequences,
    attention_masks,
    max_length=8,
    pad_token_id=0,
    label_pad_id=-100
)
```

## Packing Strategies

### Greedy First-Fit

The default strategy packs sequences greedily:

```python
# Sequences are packed in order, fitting as many as possible
packed = pack_sequences(sequences, max_length=2048, pad_token_id=0)
```

### With Sequence Boundaries

To preserve sequence boundaries for causal attention:

```python
from fast_axolotl import pack_sequences

packed, boundaries = pack_sequences(
    sequences,
    max_length=2048,
    pad_token_id=0,
    return_boundaries=True
)

# boundaries contains start/end indices for each original sequence
```

## Integration with Training

### With Streaming Data

```python
from fast_axolotl import streaming_dataset_reader, pack_sequences

def create_packed_batches(data_path, max_length, batch_size):
    buffer = []

    for batch in streaming_dataset_reader(data_path, batch_size=100):
        buffer.extend(batch["input_ids"])

        while len(buffer) >= batch_size:
            to_pack = buffer[:batch_size]
            buffer = buffer[batch_size:]

            packed = pack_sequences(
                to_pack,
                max_length=max_length,
                pad_token_id=0
            )
            yield packed
```

### Complete Training Loop

```python
import torch
from fast_axolotl import pack_sequences

def train_with_packing(model, sequences, max_length=2048):
    optimizer = torch.optim.AdamW(model.parameters())

    # Pack all sequences
    packed = pack_sequences(
        sequences,
        max_length=max_length,
        pad_token_id=model.config.pad_token_id
    )

    # Training loop
    for i in range(0, len(packed), batch_size):
        batch = packed[i:i+batch_size]

        outputs = model(batch, labels=batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## Performance Considerations

### When Packing Helps

- Short sequences (< 25% of max length)
- Variable-length datasets
- High padding ratios

### When Packing May Not Help

- Already long sequences
- Uniform sequence lengths
- Very small batch sizes

### Benchmarks

| Scenario | Without Packing | With Packing | Improvement |
|----------|----------------|--------------|-------------|
| Avg length 256, max 2048 | 12.5% utilization | 80%+ utilization | 6.4x |
| Avg length 512, max 2048 | 25% utilization | 85%+ utilization | 3.4x |
| Avg length 1024, max 2048 | 50% utilization | 90%+ utilization | 1.8x |

## Common Patterns

### Packing with Labels

```python
# Pack both inputs and labels together
packed_inputs = pack_sequences(input_ids, max_length=2048, pad_token_id=0)
packed_labels = pack_sequences(labels, max_length=2048, pad_token_id=-100)
```

### Packing for Different Model Sizes

```python
# Adjust max_length based on model
model_configs = {
    "7b": 4096,
    "13b": 4096,
    "70b": 8192,
}

max_length = model_configs[model_size]
packed = pack_sequences(sequences, max_length=max_length, pad_token_id=0)
```

## Troubleshooting

### Sequences Longer Than Max Length

```python
# Filter or truncate long sequences first
sequences = [s[:max_length] for s in sequences if len(s) > 0]
packed = pack_sequences(sequences, max_length=max_length, pad_token_id=0)
```

### Memory Issues

```python
# Process in smaller chunks
chunk_size = 10000
all_packed = []

for i in range(0, len(sequences), chunk_size):
    chunk = sequences[i:i+chunk_size]
    packed = pack_sequences(chunk, max_length=2048, pad_token_id=0)
    all_packed.append(packed)

final = torch.cat(all_packed, dim=0)
```

## Next Steps

- [Batch Padding](batch-padding.md) - Efficient batch preprocessing
- [API Reference](../api-reference/data-processing.md) - Complete API docs
- [Best Practices](../performance/best-practices.md) - Optimization tips
