# API Reference

Complete API documentation for fast-axolotl.

## Core Functions

### is_available

```python
def is_available() -> bool
```

Check if the Rust extension is available.

**Returns:**
- `bool`: `True` if Rust extension loaded successfully

**Example:**
```python
import fast_axolotl

if fast_axolotl.is_available():
    print("Rust acceleration available!")
```

---

### get_version

```python
def get_version() -> str
```

Get the fast-axolotl version string.

**Returns:**
- `str`: Version string including Rust extension version

**Example:**
```python
print(fast_axolotl.get_version())
# "0.2.0 (rust: 0.2.0)"
```

---

### install

```python
def install() -> bool
```

Install the fast-axolotl shim into the axolotl namespace.

Patches axolotl modules to use Rust-based implementations. Called automatically on import.

**Returns:**
- `bool`: `True` if shim was installed, `False` if already installed or failed

**Example:**
```python
fast_axolotl.uninstall()
fast_axolotl.install()  # Re-install
```

---

### uninstall

```python
def uninstall() -> bool
```

Remove the fast-axolotl shim from the axolotl namespace.

**Returns:**
- `bool`: `True` if shim was removed, `False` if not installed

---

## Format Detection

### list_supported_formats

```python
def list_supported_formats() -> List[str]
```

List all supported file formats.

**Returns:**
- `List[str]`: Format strings including base formats and compressed variants

**Example:**
```python
formats = fast_axolotl.list_supported_formats()
# ['parquet', 'arrow', 'json', 'jsonl', 'csv', 'text',
#  'parquet.zst', 'json.gz', 'hf_dataset', ...]
```

---

### detect_format

```python
def detect_format(file_path: str) -> Tuple[str, Optional[str]]
```

Auto-detect file format and compression from path.

**Parameters:**
- `file_path`: Path to the file or directory

**Returns:**
- `Tuple[str, Optional[str]]`: (base_format, compression)

**Examples:**
```python
detect_format("data.parquet")      # ("parquet", None)
detect_format("data.jsonl.zst")    # ("jsonl", "zstd")
detect_format("data.csv.gz")       # ("csv", "gzip")
detect_format("/path/to/hf_dir/")  # ("hf_dataset", None)
```

---

## Streaming

### streaming_dataset_reader

```python
def streaming_dataset_reader(
    file_path: str,
    dataset_type: str,
    batch_size: int = 1000,
    num_threads: int = 4
) -> Iterator[Dict[str, Any]]
```

Stream data from a dataset file.

**Parameters:**
- `file_path`: Path to the dataset file
- `dataset_type`: Type of dataset (`'parquet'`, `'arrow'`, `'csv'`, `'json'`, `'jsonl'`, `'text'`)
- `batch_size`: Number of rows per batch (default: 1000)
- `num_threads`: Number of threads for processing (default: 4)

**Yields:**
- `Dict[str, Any]`: Dictionary with column names as keys and lists as values

**Example:**
```python
for batch in streaming_dataset_reader("data.parquet", "parquet", batch_size=5000):
    texts = batch["text"]
    labels = batch["label"]
```

---

### RustStreamingDataset

```python
class RustStreamingDataset:
    def __init__(
        self,
        file_path: str,
        dataset_type: str,
        batch_size: int = 1000,
        num_threads: int = 4,
        dataset_keep_in_memory: bool = False
    )
```

HuggingFace Dataset-compatible wrapper for Rust-based streaming.

**Example:**
```python
dataset = RustStreamingDataset("data.parquet", "parquet")
for batch in dataset:
    process(batch)
```

---

## Token Packing

### pack_sequences

```python
def pack_sequences(
    sequences: List[List[int]],
    max_length: int,
    pad_token_id: int,
    eos_token_id: int,
    label_pad_id: int = -100
) -> Dict[str, List[List[int]]]
```

Pack multiple sequences into fixed-length chunks.

**Parameters:**
- `sequences`: List of token ID lists to pack
- `max_length`: Maximum length for each packed sequence
- `pad_token_id`: Token ID to use for padding
- `eos_token_id`: End-of-sequence token ID
- `label_pad_id`: Padding value for labels (default: -100)

**Returns:**
- `Dict` with keys:
  - `'input_ids'`: List of packed input sequences
  - `'labels'`: List of packed label sequences
  - `'attention_mask'`: List of attention masks

**Example:**
```python
result = pack_sequences(
    sequences=[[1, 2, 3], [4, 5], [6, 7, 8, 9]],
    max_length=10,
    pad_token_id=0,
    eos_token_id=2
)
```

---

### concatenate_and_pack

```python
def concatenate_and_pack(
    input_ids: List[List[int]],
    labels: List[List[int]],
    attention_masks: List[List[int]],
    max_length: int,
    pad_token_id: int,
    label_pad_id: int = -100
) -> Dict[str, List[List[int]]]
```

Concatenate sequences and pack into max_length chunks.

Lower-level function when you have separate input_ids, labels, and attention_masks.

**Parameters:**
- `input_ids`: List of input token ID lists
- `labels`: List of label token ID lists
- `attention_masks`: List of attention mask lists
- `max_length`: Maximum length for each packed sequence
- `pad_token_id`: Token ID for padding input_ids
- `label_pad_id`: Value for padding labels (default: -100)

**Returns:**
- `Dict` with `'input_ids'`, `'labels'`, `'attention_mask'`

---

## Parallel Hashing

### parallel_hash_rows

```python
def parallel_hash_rows(
    rows: List[str],
    num_threads: int = 0
) -> List[str]
```

Compute SHA256 hashes for multiple rows in parallel.

**Parameters:**
- `rows`: List of string representations of rows
- `num_threads`: Number of threads (0 = auto-detect CPU cores)

**Returns:**
- `List[str]`: Hex-encoded SHA256 hashes in same order as input

**Example:**
```python
rows = ["row1", "row2", "row3"]
hashes = parallel_hash_rows(rows, num_threads=0)
# ['d4735e3a265e16...', 'b2d2226c48a9bd...', ...]
```

---

### deduplicate_indices

```python
def deduplicate_indices(
    rows: List[str],
    existing_hashes: Optional[List[str]] = None,
    num_threads: int = 0
) -> Tuple[List[int], List[str]]
```

Find indices of unique rows using parallel hashing.

**Parameters:**
- `rows`: List of string representations of rows
- `existing_hashes`: Optional list of already-seen hashes to filter against
- `num_threads`: Number of threads (0 = auto-detect)

**Returns:**
- `Tuple[List[int], List[str]]`: (unique_indices, all_hashes)

**Example:**
```python
rows = ["a", "b", "a", "c"]
unique_idx, hashes = deduplicate_indices(rows)
# unique_idx = [0, 1, 3]  # Indices of unique rows
```

---

## Batch Padding

### pad_sequences

```python
def pad_sequences(
    sequences: List[List[int]],
    target_length: Optional[int] = None,
    pad_value: int = 0,
    padding_side: str = "right",
    pad_to_multiple_of: Optional[int] = None
) -> List[List[int]]
```

Pad a batch of sequences to a target length.

**Parameters:**
- `sequences`: List of token ID lists
- `target_length`: Length to pad to (None = max in batch)
- `pad_value`: Value to use for padding (default: 0)
- `padding_side`: `"right"` or `"left"`
- `pad_to_multiple_of`: Optionally pad to multiple of this value

**Returns:**
- `List[List[int]]`: Padded sequences

**Example:**
```python
sequences = [[1, 2, 3], [4, 5]]
padded = pad_sequences(sequences, target_length=5, pad_value=0)
# [[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]]
```

---

### create_padding_mask

```python
def create_padding_mask(
    current_length: int,
    target_length: int
) -> List[int]
```

Create position IDs for padding.

**Parameters:**
- `current_length`: Current sequence length
- `target_length`: Target length after padding

**Returns:**
- `List[int]`: Position IDs (0, 1, 2, ...)

**Example:**
```python
mask = create_padding_mask(5, 8)
# [0, 1, 2]  # 3 padding positions needed
```

---

## Constants

### RUST_AVAILABLE

```python
RUST_AVAILABLE: bool
```

Boolean indicating if the Rust extension was successfully imported.

---

## Module Exports

All public functions are available via:

```python
from fast_axolotl import (
    # Core
    is_available,
    get_version,
    install,
    uninstall,
    RUST_AVAILABLE,

    # Format Detection
    list_supported_formats,
    detect_format,

    # Streaming
    streaming_dataset_reader,
    RustStreamingDataset,
    create_rust_streaming_dataset,
    should_use_rust_streaming,

    # Token Packing
    pack_sequences,
    concatenate_and_pack,

    # Parallel Hashing
    parallel_hash_rows,
    deduplicate_indices,

    # Batch Padding
    pad_sequences,
    create_padding_mask,
)
```
