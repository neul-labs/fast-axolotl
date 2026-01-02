# Batch Padding

fast-axolotl provides optimized batch padding for efficient model training, automatically handling variable-length sequences.

## Overview

Batch padding ensures all sequences in a batch have the same length, which is required for efficient tensor operations. fast-axolotl's Rust implementation is optimized for:

- Fast memory allocation
- Vectorized operations
- Minimal Python overhead

## Basic Usage

### Padding Sequences

```python
from fast_axolotl import pad_sequences

sequences = [
    [1, 2, 3],
    [4, 5],
    [6, 7, 8, 9, 10],
]

# Pad to longest sequence
padded = pad_sequences(sequences, pad_value=0)

print(padded)
# [[1, 2, 3, 0, 0],
#  [4, 5, 0, 0, 0],
#  [6, 7, 8, 9, 10]]
```

### Padding Side

Control whether padding is added to the left or right:

```python
# Right padding (default) - for causal/decoder models
padded_right = pad_sequences(sequences, pad_value=0, padding_side="right")
# [[1, 2, 3, 0, 0], ...]

# Left padding - for encoder models or specific use cases
padded_left = pad_sequences(sequences, pad_value=0, padding_side="left")
# [[0, 0, 1, 2, 3], ...]
```

## Creating Attention Masks

### Basic Mask Creation

```python
from fast_axolotl import create_padding_mask

sequences = [
    [1, 2, 3],
    [4, 5],
]

# Pad sequences
padded = pad_sequences(sequences, pad_value=0)

# Create attention mask (1 for real tokens, 0 for padding)
mask = create_padding_mask(padded, pad_value=0)

print(mask)
# [[1, 1, 1, 0, 0],
#  [1, 1, 0, 0, 0]]
```

### Position IDs

Generate position IDs that account for padding:

```python
from fast_axolotl import create_padding_mask

def create_position_ids(attention_mask):
    """Create position IDs from attention mask."""
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)
    return position_ids

mask = create_padding_mask(padded, pad_value=0)
positions = create_position_ids(mask)
```

## Padding for Training

### Complete Batch Preparation

```python
from fast_axolotl import pad_sequences, create_padding_mask
import torch

def prepare_batch(input_ids_list, label_ids_list, tokenizer):
    """Prepare a batch for training."""

    # Pad inputs
    input_ids = pad_sequences(
        input_ids_list,
        pad_value=tokenizer.pad_token_id,
        padding_side="right"
    )

    # Pad labels (use -100 to ignore in loss)
    labels = pad_sequences(
        label_ids_list,
        pad_value=-100,
        padding_side="right"
    )

    # Create attention mask
    attention_mask = create_padding_mask(input_ids, pad_value=tokenizer.pad_token_id)

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
    }
```

### With Max Length

Truncate and pad to a specific length:

```python
def pad_to_max_length(sequences, max_length, pad_value):
    """Pad or truncate sequences to exact length."""

    # Truncate if needed
    truncated = [seq[:max_length] for seq in sequences]

    # Pad to max_length
    padded = pad_sequences(
        truncated,
        pad_value=pad_value,
        max_length=max_length
    )

    return padded

# Ensure all sequences are exactly 2048 tokens
padded = pad_to_max_length(sequences, max_length=2048, pad_value=0)
```

## Dynamic vs Static Batching

### Dynamic Batching (Default)

Pad to the longest sequence in each batch:

```python
# Batch 1: max length might be 256
batch1 = pad_sequences(sequences_batch1, pad_value=0)

# Batch 2: max length might be 512
batch2 = pad_sequences(sequences_batch2, pad_value=0)
```

### Static Batching

Pad all batches to the same length for consistent memory usage:

```python
MAX_LENGTH = 2048

batch1 = pad_sequences(sequences_batch1, pad_value=0, max_length=MAX_LENGTH)
batch2 = pad_sequences(sequences_batch2, pad_value=0, max_length=MAX_LENGTH)
```

## Integration with Data Loaders

### Custom Collate Function

```python
from torch.utils.data import DataLoader
from fast_axolotl import pad_sequences, create_padding_mask
import torch

def collate_fn(batch):
    """Custom collate with fast-axolotl padding."""

    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Fast padding
    padded_inputs = pad_sequences(input_ids, pad_value=0)
    padded_labels = pad_sequences(labels, pad_value=-100)
    attention_mask = create_padding_mask(padded_inputs, pad_value=0)

    return {
        "input_ids": torch.tensor(padded_inputs),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(padded_labels),
    }

# Use in DataLoader
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```

### With HuggingFace Trainer

```python
from transformers import Trainer, TrainingArguments
from fast_axolotl import pad_sequences

class FastCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        padded_inputs = pad_sequences(input_ids, pad_value=self.tokenizer.pad_token_id)
        padded_labels = pad_sequences(labels, pad_value=-100)

        return {
            "input_ids": torch.tensor(padded_inputs),
            "labels": torch.tensor(padded_labels),
        }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=FastCollator(tokenizer),
)
```

## Performance Tips

### 1. Batch Similar Lengths Together

Sorting by length reduces padding waste:

```python
def create_length_sorted_batches(sequences, batch_size):
    """Create batches sorted by length to minimize padding."""

    # Sort by length
    sorted_idx = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    sorted_sequences = [sequences[i] for i in sorted_idx]

    # Create batches
    batches = []
    for i in range(0, len(sorted_sequences), batch_size):
        batch = sorted_sequences[i:i+batch_size]
        padded = pad_sequences(batch, pad_value=0)
        batches.append(padded)

    return batches
```

### 2. Use Appropriate Batch Sizes

| Sequence Length | Recommended Batch Size |
|-----------------|----------------------|
| < 512 | 32-64 |
| 512-2048 | 8-32 |
| 2048-4096 | 4-16 |
| > 4096 | 1-8 |

### 3. Avoid Excessive Padding

Monitor your padding ratio:

```python
def padding_ratio(padded_batch, pad_value):
    """Calculate percentage of padding in batch."""
    total = padded_batch.numel()
    padding = (padded_batch == pad_value).sum().item()
    return padding / total

ratio = padding_ratio(batch["input_ids"], pad_value=0)
if ratio > 0.5:
    print(f"Warning: {ratio:.1%} padding - consider smaller max_length")
```

## Next Steps

- [Token Packing](token-packing.md) - Reduce padding with sequence packing
- [API Reference](../api-reference/data-processing.md) - Complete API docs
- [Best Practices](../performance/best-practices.md) - Optimization strategies
