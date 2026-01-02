# Installation

This guide covers all installation methods for fast-axolotl.

## Requirements

- **Python**: 3.10, 3.11, or 3.12
- **Platforms**: Linux (x86_64, aarch64), macOS (x86_64, Apple Silicon), Windows (x86_64)

## Install from PyPI (Recommended)

The simplest way to install fast-axolotl is via pip:

```bash
pip install fast-axolotl
```

Pre-built wheels are available for all supported platforms, so no compilation is required.

## Install from Source

If you need to build from source (e.g., for development or unsupported platforms):

### Prerequisites

1. **Rust toolchain** (1.70 or later):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **maturin** (Rust-Python build tool):
   ```bash
   pip install maturin
   ```

### Build and Install

```bash
# Clone the repository
git clone https://github.com/dipankar/fast-axolotl.git
cd fast-axolotl

# Build and install in development mode
maturin develop --release

# Or build a wheel
maturin build --release
pip install target/wheels/fast_axolotl-*.whl
```

## Verify Installation

After installation, verify that fast-axolotl is working correctly:

```python
import fast_axolotl

# Check version
print(f"Version: {fast_axolotl.__version__}")

# Check Rust extension
print(f"Rust available: {fast_axolotl.rust_available()}")

# List supported formats
print(f"Formats: {fast_axolotl.list_supported_formats()}")
```

Expected output:

```
Version: 0.1.x
Rust available: True
Formats: ['parquet', 'arrow', 'feather', 'json', 'jsonl', 'csv', 'text']
```

## Installation with Axolotl

fast-axolotl is designed to work alongside Axolotl. Install both:

```bash
pip install axolotl fast-axolotl
```

Then enable acceleration in your training script:

```python
import fast_axolotl
fast_axolotl.install()

# Continue with normal axolotl usage
```

## Troubleshooting

### ImportError: Rust extension not found

If you see this error, the Rust extension failed to load. Try:

1. **Reinstall the package**:
   ```bash
   pip uninstall fast-axolotl
   pip install fast-axolotl --no-cache-dir
   ```

2. **Check Python version**: Ensure you're using Python 3.10, 3.11, or 3.12.

3. **Check platform**: Verify your platform is supported (see Requirements above).

### Build errors from source

If building from source fails:

1. **Update Rust**:
   ```bash
   rustup update
   ```

2. **Install build dependencies** (Linux):
   ```bash
   sudo apt-get install build-essential python3-dev
   ```

3. **Install build dependencies** (macOS):
   ```bash
   xcode-select --install
   ```

### Performance issues

If performance is worse than expected:

1. Ensure you're using the release build (not debug)
2. Check that the Rust extension is loaded: `fast_axolotl.rust_available()`
3. See [Best Practices](../performance/best-practices.md) for optimization tips

## Virtual Environments

We recommend using a virtual environment:

=== "venv"

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    .venv\Scripts\activate     # Windows
    pip install fast-axolotl
    ```

=== "conda"

    ```bash
    conda create -n fast-axolotl python=3.11
    conda activate fast-axolotl
    pip install fast-axolotl
    ```

=== "uv"

    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install fast-axolotl
    ```

## Next Steps

- [Quick Start](quick-start.md) - Get started with your first example
- [Streaming Data Guide](../user-guide/streaming.md) - Learn about streaming data loading
