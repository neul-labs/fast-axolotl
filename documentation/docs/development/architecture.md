# Architecture

This document describes the internal architecture of fast-axolotl.

## Overview

fast-axolotl is a hybrid Python-Rust package that accelerates LLM training data pipelines:

```
┌─────────────────────────────────────────────────────────────┐
│                      Python API                             │
│  (fast_axolotl/__init__.py, streaming.py)                   │
├─────────────────────────────────────────────────────────────┤
│                      PyO3 Bindings                          │
│  (Python ↔ Rust interface)                                  │
├─────────────────────────────────────────────────────────────┤
│                      Rust Core                              │
│  (src/lib.rs - streaming, hashing, packing, padding)        │
├─────────────────────────────────────────────────────────────┤
│                   Rust Dependencies                         │
│  (arrow, parquet, tokio, rayon, sha2, pyo3)                │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Details

### Python Layer

**`src/fast_axolotl/__init__.py`**

The main Python API that:

- Exposes Rust functions to users
- Implements the shimming system
- Provides high-level wrappers with validation
- Handles fallbacks when Rust is unavailable

Key sections:

```python
# Rust extension import
from ._fast_axolotl import (
    streaming_dataset_reader,
    pack_sequences,
    parallel_hash_rows,
    pad_sequences,
    # ... more functions
)

# Shimming system
def install():
    """Install shims into axolotl namespace."""
    _install_streaming_shim()
    _install_hashing_shim()
    _install_collator_shim()
```

### PyO3 Bindings

**`src/lib.rs`**

PyO3 provides the Python-Rust interface:

```rust
use pyo3::prelude::*;

#[pyfunction]
fn streaming_dataset_reader(
    py: Python<'_>,
    path: &str,
    batch_size: usize,
    columns: Option<Vec<String>>,
) -> PyResult<PyObject> {
    // Rust implementation
    // Returns Python-compatible objects
}

#[pymodule]
fn _fast_axolotl(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(streaming_dataset_reader, m)?)?;
    // Register more functions...
    Ok(())
}
```

### Rust Core

The core Rust implementation provides:

#### Streaming Reader

```rust
// File format detection
fn detect_format(path: &Path) -> (FileFormat, Compression)

// Parquet streaming
fn stream_parquet(
    path: &Path,
    batch_size: usize,
    columns: Option<&[String]>,
) -> impl Iterator<Item = RecordBatch>

// Multi-format support
fn stream_file(path: &Path, ...) -> Box<dyn Iterator<Item = RecordBatch>>
```

#### Parallel Hashing

```rust
use rayon::prelude::*;
use sha2::{Sha256, Digest};

fn parallel_hash_rows(rows: &[Vec<u8>]) -> Vec<String> {
    rows.par_iter()
        .map(|row| {
            let mut hasher = Sha256::new();
            hasher.update(row);
            hex::encode(hasher.finalize())
        })
        .collect()
}
```

#### Token Packing

```rust
fn pack_sequences(
    sequences: Vec<Vec<i64>>,
    max_length: usize,
    pad_token_id: i64,
) -> Vec<Vec<i64>> {
    // Greedy first-fit packing algorithm
    let mut packed = Vec::new();
    let mut current = Vec::with_capacity(max_length);

    for seq in sequences {
        if current.len() + seq.len() <= max_length {
            current.extend(seq);
        } else {
            // Pad and save current, start new
            current.resize(max_length, pad_token_id);
            packed.push(std::mem::take(&mut current));
            current = seq;
        }
    }
    // Handle remaining
    if !current.is_empty() {
        current.resize(max_length, pad_token_id);
        packed.push(current);
    }

    packed
}
```

---

## Shimming System

The shimming system transparently replaces Axolotl functions:

```python
def _install_streaming_shim():
    """Replace axolotl.utils.data.rust_streaming with fast-axolotl."""
    import sys

    # Create shim module
    class StreamingShim:
        def __init__(self):
            self.__fast_axolotl_shimmed__ = True

        streaming_dataset_reader = staticmethod(streaming_dataset_reader)
        RustStreamingDataset = RustStreamingDataset
        create_rust_streaming_dataset = staticmethod(create_rust_streaming_dataset)

    # Install in sys.modules
    sys.modules["axolotl.utils.data.rust_streaming"] = StreamingShim()
```

### Shim Architecture

```
┌────────────────────────────────────────────────────────────┐
│                   User Code                                │
│  from axolotl.utils.data import streaming_dataset_reader   │
├────────────────────────────────────────────────────────────┤
│                   sys.modules                              │
│  "axolotl.utils.data.rust_streaming" → StreamingShim       │
├────────────────────────────────────────────────────────────┤
│                   StreamingShim                            │
│  streaming_dataset_reader → fast_axolotl.streaming_reader  │
├────────────────────────────────────────────────────────────┤
│                   fast-axolotl Rust                        │
│  High-performance implementation                           │
└────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Streaming Read Flow

```
1. User calls streaming_dataset_reader("data.parquet", batch_size=1000)
   │
2. Python wrapper validates parameters
   │
3. PyO3 converts Python types to Rust types
   │
4. Rust detects format (parquet) and compression (none)
   │
5. Rust opens ParquetReader with Arrow
   │
6. For each batch:
   ├─ Rust reads row_group(s) to fill batch_size
   ├─ Converts Arrow RecordBatch to Python dict
   └─ Yields to Python iterator
   │
7. User processes batch in Python
```

### Parallel Hash Flow

```
1. User calls parallel_hash_rows([b"row1", b"row2", ...])
   │
2. PyO3 extracts bytes from Python list
   │
3. Rust uses rayon to parallel iterate:
   ├─ Thread 1: hash rows 0-999
   ├─ Thread 2: hash rows 1000-1999
   ├─ Thread 3: hash rows 2000-2999
   └─ ...
   │
4. Collect results in order
   │
5. Convert to Python list of hex strings
```

---

## Dependencies

### Rust Dependencies (`Cargo.toml`)

```toml
[dependencies]
pyo3 = { version = "0.23", features = ["extension-module"] }
arrow = "53"           # Arrow/Parquet support
parquet = "53"         # Parquet format
tokio = "1"            # Async runtime
rayon = "1.8"          # Parallel iteration
sha2 = "0.10"          # SHA256 hashing
zstd = "0.13"          # ZSTD compression
flate2 = "1.0"         # Gzip compression
```

### Python Dependencies (`pyproject.toml`)

```toml
[project]
dependencies = []  # No required dependencies!

[project.optional-dependencies]
dev = [
    "pytest",
    "maturin",
    "ruff",
    "mypy",
]
```

---

## Performance Optimizations

### Memory Efficiency

- **Zero-copy where possible**: Arrow arrays avoid copying
- **Streaming**: Never load full dataset into memory
- **Batch processing**: Process data in chunks

### CPU Efficiency

- **SIMD**: Arrow uses vectorized operations
- **Parallel**: Rayon for multi-threaded processing
- **Native code**: Rust compiled to machine code

### I/O Efficiency

- **Memory mapping**: Arrow supports mmap for large files
- **Columnar reads**: Only read needed columns from Parquet
- **Compression**: ZSTD provides fast decompression

---

## Extension Points

### Adding a New File Format

1. Add format detection in `detect_format()`:

```rust
fn detect_format(path: &Path) -> FileFormat {
    match path.extension().and_then(|e| e.to_str()) {
        Some("newformat") => FileFormat::NewFormat,
        // ...
    }
}
```

2. Implement reader:

```rust
fn stream_newformat(path: &Path, batch_size: usize) -> impl Iterator<Item = RecordBatch> {
    // Implementation
}
```

3. Add to main dispatch:

```rust
fn stream_file(path: &Path, ...) -> Box<dyn Iterator<...>> {
    match detect_format(path) {
        FileFormat::NewFormat => Box::new(stream_newformat(path, batch_size)),
        // ...
    }
}
```

### Adding a New Processing Function

1. Implement in Rust:

```rust
#[pyfunction]
fn new_function(data: Vec<i64>) -> PyResult<Vec<i64>> {
    Ok(data.iter().map(|x| x * 2).collect())
}
```

2. Register in module:

```rust
m.add_function(wrap_pyfunction!(new_function, m)?)?;
```

3. Expose in Python:

```python
from ._fast_axolotl import new_function

__all__ = [..., "new_function"]
```

---

## See Also

- [Contributing](contributing.md) - Development setup
- [API Reference](../api-reference/core.md) - Function documentation
- [Benchmarks](../performance/benchmarks.md) - Performance data
