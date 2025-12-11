# Installation Guide

This guide covers all methods for installing fast-axolotl.

## Requirements

- Python 3.10, 3.11, or 3.12
- Rust toolchain (for building from source)

## From PyPI (Recommended)

The simplest way to install fast-axolotl:

```bash
pip install fast-axolotl
```

Pre-built wheels are available for:
- Linux (x86_64, aarch64)
- macOS (x86_64, arm64)
- Windows (x86_64)

## Using uv

[uv](https://github.com/astral-sh/uv) is a fast Python package manager:

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install fast-axolotl
uv pip install fast-axolotl
```

## From Source

Building from source requires the Rust toolchain.

### Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

### Clone and Build

```bash
git clone https://github.com/axolotl-ai-cloud/fast-axolotl
cd fast-axolotl
```

#### Using uv (recommended)

```bash
uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
uv pip install -e .
```

#### Using pip + maturin

```bash
pip install maturin
maturin develop --release
```

### Development Installation

For development with test dependencies:

```bash
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

## Verifying Installation

```python
import fast_axolotl

print(fast_axolotl.get_version())
# Output: 0.2.0 (rust: 0.2.0)

print(fast_axolotl.is_available())
# Output: True
```

## Troubleshooting

### Rust Extension Not Available

If `is_available()` returns `False`:

1. Ensure you built with `--release` flag:
   ```bash
   maturin develop --release
   ```

2. Check that the Rust toolchain is installed:
   ```bash
   rustc --version
   cargo --version
   ```

3. Try rebuilding:
   ```bash
   cargo clean
   maturin develop --release
   ```

### Import Errors

If you get import errors, ensure the package is installed in your active environment:

```bash
pip list | grep fast-axolotl
```

### Platform-Specific Issues

#### Linux

On some Linux distributions, you may need to install development headers:

```bash
# Ubuntu/Debian
sudo apt-get install python3-dev

# Fedora/RHEL
sudo dnf install python3-devel
```

#### macOS

Ensure Xcode command line tools are installed:

```bash
xcode-select --install
```

#### Windows

Ensure you have the Visual Studio Build Tools installed with the "Desktop development with C++" workload.

## Next Steps

- [Usage Guide](usage.md) - Learn how to use fast-axolotl
- [API Reference](api.md) - Detailed function documentation
