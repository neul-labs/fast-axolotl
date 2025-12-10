# Fast-Axolotl

High-performance Rust extensions for [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) - drop-in acceleration for existing installations.

## Overview

Fast-Axolotl provides Rust-based streaming dataset loading that can be shimmed into existing Axolotl installations. This helps prevent RAM OOM errors when working with large long-context datasets by streaming data directly from disk in batches.

## Features

- **Memory-efficient streaming**: Process large datasets without loading everything into RAM
- **Multiple format support**: Parquet, Arrow, CSV, JSON/JSONL
- **Drop-in replacement**: Simply `pip install fast-axolotl` alongside your existing Axolotl installation
- **Auto-shimming**: Automatically patches Axolotl's data loading when imported

## Installation

### Using uv (recommended)

```bash
# Install from source
cd ~/fast-axolotl
uv pip install -e .

# Or build a wheel
uv build
```

### Using pip

```bash
pip install fast-axolotl
```

### Building from source

Requires Rust toolchain:

```bash
# Install Rust if needed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build with maturin
cd ~/fast-axolotl
pip install maturin
maturin develop --release
```

## Usage

### Automatic shimming (recommended)

Simply import `fast_axolotl` before using Axolotl:

```python
import fast_axolotl  # This auto-installs the shim

# Now use axolotl normally - it will use Rust streaming when beneficial
import axolotl
```

### Direct usage

You can also use the streaming functionality directly:

```python
from fast_axolotl.streaming import streaming_dataset_reader, is_available

if is_available():
    for batch in streaming_dataset_reader(
        file_path="/path/to/large_dataset.parquet",
        dataset_type="parquet",
        batch_size=1000,
        num_threads=4
    ):
        # Process each batch
        process(batch)
```

### Configuration

Enable Rust streaming in your Axolotl config:

```yaml
# axolotl config
dataset_use_rust_streaming: true
sequence_len: 32768  # Rust streaming is used for sequence_len > 10K
```

## Supported Dataset Types

| Type | Extension | Description |
|------|-----------|-------------|
| parquet | .parquet | Columnar storage, most efficient |
| arrow | .arrow | Arrow IPC format |
| csv | .csv | Text-based tabular data |
| json | .json, .jsonl | JSON Lines format |

## API Reference

### fast_axolotl.is_available()
Check if the Rust extension is available.

### fast_axolotl.streaming_dataset_reader(file_path, dataset_type, batch_size=1000, num_threads=4)
Stream data from a dataset file using the Rust extension.

### fast_axolotl.RustStreamingDataset
HuggingFace Dataset-compatible wrapper for Rust-based streaming.

### fast_axolotl.install() / fast_axolotl.uninstall()
Manually install or remove the axolotl shim.

## Development

```bash
# Clone the repo
git clone https://github.com/axolotl-ai-cloud/fast-axolotl
cd fast-axolotl

# Create virtual environment with uv
uv venv
source .venv/bin/activate

# Install dev dependencies
uv pip install -e ".[dev]"

# Build Rust extension in development mode
maturin develop

# Run tests
pytest
```

## License

Apache-2.0
