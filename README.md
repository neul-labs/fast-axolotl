# Fast-Axolotl

[![CI](https://github.com/neul-labs/fast-axolotl/actions/workflows/ci.yml/badge.svg)](https://github.com/neul-labs/fast-axolotl/actions/workflows/ci.yml)
[![Compatibility](https://github.com/neul-labs/fast-axolotl/actions/workflows/compatibility-tests.yml/badge.svg)](https://github.com/neul-labs/fast-axolotl/actions/workflows/compatibility-tests.yml)
[![PyPI](https://img.shields.io/pypi/v/fast-axolotl.svg)](https://pypi.org/project/fast-axolotl/)
[![Python](https://img.shields.io/pypi/pyversions/fast-axolotl.svg)](https://pypi.org/project/fast-axolotl/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Rust extensions for [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) fine-tuning — no OOM on large datasets.** Drop-in acceleration for existing installations, with zero config.

**Links:** [Website](https://fast-axolotl.neullabs.com) · [Docs](https://docs.neullabs.com/fast-axolotl) · [GitHub](https://github.com/neul-labs/fast-axolotl)

## Highlights

- **Zero-config acceleration** - Just `import fast_axolotl` before axolotl
- **77x faster streaming** - Rust-based data loading vs HuggingFace datasets
- **Parallel hashing** - Multi-threaded SHA256 for deduplication
- **Cross-platform** - Linux, macOS, Windows with Python 3.10-3.13

## Quick Start

```bash
uv add fast-axolotl
```

or

```bash
pip install fast-axolotl
```

`fast-axolotl` is also published to [crates.io](https://crates.io/crates/fast-axolotl) as the PyO3 extension module that powers the Python wheel. The Python package above is the intended interface for most users; to build against the Rust crate directly, run `cargo add fast-axolotl`.

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
uv pip install fast-axolotl
```

### From Source

```bash
git clone https://github.com/neul-labs/fast-axolotl
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
git clone https://github.com/neul-labs/fast-axolotl
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

## Support

Questions or bugs? Reach out via:

- GitHub Issues: https://github.com/neul-labs/fast-axolotl/issues
- GitHub Discussions: https://github.com/neul-labs/fast-axolotl/discussions

## Maintainers

Fast-Axolotl is built and maintained by [Neul Labs](https://www.neullabs.com) (<contact@neullabs.com>).

## Part of the Neul Labs toolchain

Fast-Axolotl is part of the Neul Labs accelerators family. Explore the rest of the toolchain from [Neul Labs](https://www.neullabs.com):

| Project | What it does |
|---------|--------------|
| [fast-litellm](https://fast-litellm.neullabs.com) | Drop-in Rust acceleration for LiteLLM. |
| [fast-langgraph](https://fast-langgraph.neullabs.com) | Rust accelerators for LangGraph — up to 700x faster checkpoints. |
| [fast-crewai](https://fast-crewai.neullabs.com) | Drop-in Rust acceleration for CrewAI. |
| [fastagentic](https://fastagentic.neullabs.com) | Build agents with any framework; ship them with FastAgentic. |

## License

MIT
