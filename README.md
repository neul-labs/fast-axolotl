# Fast-Axolotl

[![CI](https://github.com/axolotl-ai-cloud/fast-axolotl/actions/workflows/ci.yml/badge.svg)](https://github.com/axolotl-ai-cloud/fast-axolotl/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fast-axolotl.svg)](https://pypi.org/project/fast-axolotl/)
[![Python](https://img.shields.io/pypi/pyversions/fast-axolotl.svg)](https://pypi.org/project/fast-axolotl/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

High-performance Rust extensions for [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) - drop-in acceleration for existing installations.

## Highlights

- **Zero-config acceleration** - Just `import fast_axolotl` before axolotl
- **77x faster streaming** - Rust-based data loading vs HuggingFace datasets
- **Parallel hashing** - Multi-threaded SHA256 for deduplication
- **Cross-platform** - Linux, macOS, Windows with Python 3.10-3.12

## Quick Start

```bash
pip install fast-axolotl
```

```python
import fast_axolotl  # Auto-installs acceleration shim

# Now use axolotl normally - accelerations are active
import axolotl
```

## Benchmark Results

Tested on Linux x86_64, Python 3.11, 16 CPU cores:

| Operation | Data Size | Rust | Python | Speedup |
|-----------|-----------|------|--------|---------|
| Streaming Data Loading | 50,000 rows | 0.009s | 0.724s | **77x** |
| Parallel Hashing (SHA256) | 100,000 rows | 0.027s | 0.052s | **1.9x** |
| Token Packing | 10,000 sequences | 0.079s | 0.033s | 0.4x* |
| Batch Padding | 10,000 sequences | 0.200s | 0.105s | 0.5x* |

*Token packing and batch padding show overhead for small datasets due to FFI costs. Performance gains are realized with larger datasets typical in LLM training.

See [BENCHMARK.md](BENCHMARK.md) for detailed results.

## Compatibility

All features tested and working:

| Feature | Status |
|---------|--------|
| Rust Extension Loading | Tested |
| Module Shimming | Tested |
| Streaming (Parquet, JSON, CSV, Arrow) | Tested |
| Token Packing | Tested |
| Parallel Hashing | Tested |
| Batch Padding | Tested |
| Axolotl Integration | Tested |

See [COMPATIBILITY.md](COMPATIBILITY.md) for full test results.

## Features

### 1. Streaming Data Loading

Memory-efficient streaming for large datasets:

```python
from fast_axolotl import streaming_dataset_reader

for batch in streaming_dataset_reader(
    "/path/to/large_dataset.parquet",
    dataset_type="parquet",
    batch_size=1000,
    num_threads=4
):
    process(batch)
```

Supports: Parquet, Arrow, JSON, JSONL, CSV, Text (with ZSTD/Gzip compression)

### 2. Token Packing

Replace inefficient `torch.cat()` loops:

```python
from fast_axolotl import pack_sequences

result = pack_sequences(
    sequences=[[1, 2, 3], [4, 5], [6, 7, 8, 9]],
    max_length=2048,
    pad_token_id=0,
    eos_token_id=2
)
# Returns: {'input_ids': [...], 'labels': [...], 'attention_mask': [...]}
```

### 3. Parallel Hashing

Multi-threaded SHA256 for deduplication:

```python
from fast_axolotl import parallel_hash_rows, deduplicate_indices

hashes = parallel_hash_rows(rows, num_threads=0)  # 0 = auto

# Or get unique indices directly
unique_indices, new_hashes = deduplicate_indices(rows)
```

### 4. Batch Padding

Efficient sequence padding:

```python
from fast_axolotl import pad_sequences

padded = pad_sequences(
    [[1, 2, 3], [4, 5]],
    target_length=8,
    pad_value=0,
    padding_side="right"
)
```

## Installation

### From PyPI

```bash
pip install fast-axolotl
```

### From Source

```bash
git clone https://github.com/axolotl-ai-cloud/fast-axolotl
cd fast-axolotl

# Using uv (recommended)
uv pip install -e .

# Or with pip + maturin
pip install maturin
maturin develop --release
```

## Documentation

- [Installation Guide](docs/installation.md)
- [Usage Guide](docs/usage.md)
- [API Reference](docs/api.md)
- [Benchmarks](docs/benchmarks.md)
- [Compatibility](docs/compatibility.md)
- [Contributing](docs/contributing.md)

## Configuration

Enable features in your Axolotl config:

```yaml
# Enable Rust streaming for large datasets
dataset_use_rust_streaming: true
sequence_len: 32768

# Deduplication uses parallel hashing automatically
dedupe: true
```

## Development

```bash
git clone https://github.com/axolotl-ai-cloud/fast-axolotl
cd fast-axolotl

uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
maturin develop

# Run tests
pytest -v

# Run benchmarks
python scripts/benchmark.py

# Run compatibility tests
python scripts/compatibility_test.py
```

## License

MIT
