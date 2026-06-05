# Installation

`fast-axolotl` ships pre-built wheels for Linux, macOS, and Windows on Python
3.10-3.12. For most users a single `pip install` is enough.

## Requirements

- **Python**: 3.10, 3.11, or 3.12 (3.9 is not supported, 3.13+ is not yet tested)
- **Operating system**: Linux (x86_64, aarch64), macOS (x86_64, arm64), or Windows (x86_64)
- **Runtime deps**: `datasets >= 2.14.0`, `numpy >= 1.24.0` (installed automatically)

No Rust toolchain is required when installing from PyPI.

## Install from PyPI

```bash
pip install fast-axolotl
```

Or using [uv](https://github.com/astral-sh/uv):

```bash
uv add fast-axolotl
# or, inside an existing project
uv pip install fast-axolotl
```

## Install Alongside Axolotl

`fast-axolotl` is designed to live next to Axolotl; install both and let the
shim do the rest:

```bash
pip install axolotl fast-axolotl
```

```python
import fast_axolotl  # auto-installs the shim before axolotl is imported
import axolotl
```

## Install from Source

You only need this path for unsupported platforms or to hack on the Rust
extension.

### Prerequisites

```bash
# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# maturin (Rust-Python build tool)
pip install maturin
```

### Clone and build

```bash
git clone https://github.com/neul-labs/fast-axolotl.git
cd fast-axolotl

# uv path (recommended)
uv venv && source .venv/bin/activate
uv pip install -e .

# or pip + maturin
maturin develop --release
```

For development with tests and linters:

```bash
uv pip install -e ".[dev]"
maturin develop
```

## Verify the Install

```python
import fast_axolotl

print(fast_axolotl.get_version())
# 0.2.0 (rust: 0.2.0)

print(fast_axolotl.is_available())
# True

print(fast_axolotl.list_supported_formats())
# ['parquet', 'arrow', 'feather', 'csv', 'json', 'jsonl', 'text',
#  'parquet.zst', 'parquet.gz', ..., 'hf_dataset']
```

If `is_available()` returns `False`, the Rust extension failed to load - see
[Troubleshooting](#troubleshooting) below.

## Troubleshooting

### `is_available()` returns `False`

The pure-Python wrapper loaded but the compiled `_rust_ext` module did not.
Typical causes:

1. You built from source without `--release`. Rebuild with:
   ```bash
   maturin develop --release
   ```
2. Your Python version is outside the 3.10-3.12 range.
3. The wheel for your platform is missing - reinstall with `--no-cache-dir`:
   ```bash
   pip install --force-reinstall --no-cache-dir fast-axolotl
   ```

### Platform-specific notes

**Linux** - requires glibc 2.17+ (any modern distribution). musl libc is not
currently supported. On Debian/Ubuntu you may need `python3-dev` when
building from source.

**macOS** - First import may be slow because of code-signing verification;
subsequent imports are fast. Building from source requires
`xcode-select --install`.

**Windows** - Use forward slashes in paths for consistency. Paths longer
than 260 characters can fail without long-path support enabled. Building
from source requires the Visual Studio "Desktop development with C++"
workload.

## Next Steps

- [Quick Start](quick-start.md) - run your first accelerated example
- [Auto-Shimming](../user-guide/shimming.md) - what the shim actually patches
