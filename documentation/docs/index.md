# fast-axolotl

**High-performance Rust extensions for Axolotl - drop-in acceleration for LLM training**

fast-axolotl provides blazing-fast Rust implementations of common data processing operations used in the [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) LLM fine-tuning framework. With a single import, you can accelerate your training data pipelines by up to **77x**.

---

## Key Features

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } **77x Faster Streaming**

    ---

    High-performance streaming data loading with native Parquet, Arrow, JSON, and CSV support.

-   :material-package-variant:{ .lg .middle } **Token Packing**

    ---

    Efficient sequence concatenation and packing into fixed-length chunks.

-   :material-content-duplicate:{ .lg .middle } **Parallel Deduplication**

    ---

    Multi-threaded SHA256 hashing for fast dataset deduplication.

-   :material-table-row:{ .lg .middle } **Batch Padding**

    ---

    Optimized sequence padding and attention mask generation.

</div>

---

## Quick Start

### Installation

```bash
pip install fast-axolotl
```

### Basic Usage

```python
import fast_axolotl

# Enable automatic acceleration for Axolotl
fast_axolotl.install()

# That's it! Your Axolotl training is now accelerated
```

### Direct API Usage

```python
from fast_axolotl import streaming_dataset_reader

# Stream data from Parquet files
for batch in streaming_dataset_reader("data/*.parquet", batch_size=1000):
    process(batch)
```

---

## Performance Highlights

| Feature | Speedup | Use Case |
|---------|---------|----------|
| Streaming Data Loading | **77x** | Loading large datasets |
| Parallel Hashing | **1.9x** | Dataset deduplication |
| Token Packing | Variable | Sequence concatenation |
| Batch Padding | Variable | Batch preprocessing |

---

## How It Works

fast-axolotl uses a transparent **shimming system** that automatically replaces slow Python implementations with optimized Rust code:

```python
import fast_axolotl

# Install shims into axolotl namespace
fast_axolotl.install()

# Now axolotl automatically uses Rust-accelerated functions
import axolotl
# ... your normal axolotl training code
```

No changes to your existing Axolotl configuration or code are required.

---

## Supported Formats

fast-axolotl supports a wide range of data formats:

- **Parquet** - Columnar format with efficient compression
- **Arrow/Feather** - Zero-copy memory-mapped reading
- **JSON/JSONL** - Line-delimited JSON for streaming
- **CSV** - Standard tabular data
- **Text** - Plain text files

All formats support **ZSTD** and **Gzip** compression automatically.

---

## Requirements

- Python 3.10, 3.11, or 3.12
- Linux, macOS, or Windows
- No Rust toolchain required (pre-built wheels available)

---

## Next Steps

- [Installation Guide](getting-started/installation.md) - Detailed installation instructions
- [Quick Start Tutorial](getting-started/quick-start.md) - Get up and running in minutes
- [API Reference](api-reference/core.md) - Complete API documentation
- [Performance Benchmarks](performance/benchmarks.md) - Detailed performance analysis

---

## License

fast-axolotl is released under the MIT License.
