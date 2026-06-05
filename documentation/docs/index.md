# fast-axolotl

**High-performance Rust extensions for Axolotl - drop-in acceleration for LLM training**

`fast-axolotl` provides blazing-fast Rust implementations of the data-processing operations used by the [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) LLM fine-tuning framework. A single `import fast_axolotl` automatically installs a shim into the `axolotl` namespace so your existing pipelines pick up the accelerations transparently.

---

## Highlights

- **Zero-config acceleration** - Just `import fast_axolotl` before `axolotl`
- **77x faster streaming** - Native Rust readers for Parquet, Arrow, JSON, JSONL, CSV, Text
- **Parallel hashing** - Multi-threaded SHA256 for dataset deduplication
- **Cross-platform** - Pre-built wheels for Linux, macOS, and Windows on Python 3.10-3.12

---

## Quick Start

```bash
pip install fast-axolotl
# or
uv add fast-axolotl
```

```python
import fast_axolotl  # auto-installs the acceleration shim
import axolotl       # now uses the Rust-backed implementations
```

---

## Benchmark Headlines

Measured on Linux x86_64, Python 3.11, 16 CPU cores (full results in
[BENCHMARK.md](https://github.com/neul-labs/fast-axolotl/blob/main/BENCHMARK.md)):

| Operation | Data Size | Rust | Python | Speedup |
|-----------|-----------|------|--------|---------|
| Streaming Data Loading (Parquet) | 50,000 rows | 0.009s | 0.724s | **77.26x** |
| Parallel Hashing (SHA256) | 100,000 rows | 0.027s | 0.052s | **1.90x** |
| Token Packing | 10,000 sequences | 0.079s | 0.033s | 0.42x* |
| Batch Padding | 10,000 sequences | 0.200s | 0.105s | 0.53x* |

\* Token packing and batch padding show FFI overhead at small sizes; gains
appear with the larger sequence counts typical of real LLM training runs.

---

## What Gets Accelerated

When `fast_axolotl.install()` runs (called automatically on import), the
following modules are shimmed into `sys.modules`:

| Shimmed module | Function | Purpose |
|---|---|---|
| `axolotl.utils.data.rust_streaming` | `streaming_dataset_reader` | Native streaming reader |
| `axolotl.utils.data` | `fast_parallel_hash_rows`, `fast_deduplicate_indices` | Parallel SHA256 dedupe |
| `axolotl.utils.collators` | `fast_pad_sequences`, `fast_create_padding_mask` | Batch padding |

See [Auto-Shimming](user-guide/shimming.md) for the full module list and how
to control the shim.

---

## Supported Formats

| Format | Extensions |
|---|---|
| Parquet | `.parquet` |
| Arrow IPC | `.arrow`, `.ipc` |
| Feather | `.feather` |
| JSON | `.json` |
| JSONL | `.jsonl`, `.ndjson` |
| CSV / TSV | `.csv`, `.tsv` |
| Text | `.txt` |
| HuggingFace Arrow dataset | directory with `dataset_info.json` |

All formats transparently support **ZSTD** (`.zst`) and **Gzip** (`.gz`) compression.

---

## Requirements

- Python 3.10, 3.11, or 3.12
- Linux (x86_64 / aarch64), macOS (x86_64 / arm64), or Windows (x86_64)
- No Rust toolchain required when installing from PyPI

---

## Project Status

`fast-axolotl` is authored by Dipankar Sarkar (`me@dipankar.name`) and maintained by the team at Neul Labs. The package is MIT-licensed and currently at version **0.2.0**.

- Issues: [github.com/neul-labs/fast-axolotl/issues](https://github.com/neul-labs/fast-axolotl/issues)
- Discussions: [github.com/neul-labs/fast-axolotl/discussions](https://github.com/neul-labs/fast-axolotl/discussions)

---

## Where to Next

- [Installation](getting-started/installation.md) - PyPI and from-source install paths
- [Quick Start](getting-started/quick-start.md) - first working example
- [API Reference](api-reference/core.md) - every public symbol
- [Benchmarks](performance/benchmarks.md) - methodology and full results
