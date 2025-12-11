# Contributing

Thank you for your interest in contributing to fast-axolotl! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.10, 3.11, or 3.12
- Rust toolchain (stable)
- Git

### Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

### Clone and Setup

```bash
git clone https://github.com/axolotl-ai-cloud/fast-axolotl
cd fast-axolotl

# Create virtual environment
uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install development dependencies
uv pip install -e ".[dev]"

# Build Rust extension
maturin develop
```

### Verify Setup

```bash
# Run tests
pytest -v

# Run linter
ruff check .

# Check Rust
cargo clippy
cargo fmt --check
```

## Project Structure

```
fast-axolotl/
├── src/
│   ├── lib.rs                 # Rust extension code
│   └── fast_axolotl/
│       ├── __init__.py        # Python API and shimming
│       └── streaming.py       # Streaming utilities
├── tests/
│   └── test_fast_axolotl.py   # Test suite
├── scripts/
│   ├── benchmark.py           # Benchmark script
│   └── compatibility_test.py  # Compatibility tests
├── docs/                      # Documentation
├── .github/workflows/         # CI/CD
├── Cargo.toml                 # Rust dependencies
└── pyproject.toml             # Python configuration
```

## Making Changes

### Rust Code

The Rust extension is in `src/lib.rs`. Key components:

- **PyO3 bindings** - Python-Rust interface
- **Streaming readers** - Parquet, Arrow, JSON, CSV, Text
- **Token operations** - Packing, padding
- **Parallel hashing** - Multi-threaded SHA256

After changing Rust code, rebuild:

```bash
maturin develop --release
```

### Python Code

Python code is in `src/fast_axolotl/`. The main file `__init__.py` contains:

- API wrappers for Rust functions
- Shim installation logic
- Fallback implementations

### Tests

Tests are in `tests/test_fast_axolotl.py`. Add tests for new features:

```python
def test_new_feature(self):
    import fast_axolotl
    result = fast_axolotl.new_function(...)
    assert result == expected
```

Run tests:

```bash
pytest -v
pytest -v -k "test_name"  # Run specific test
```

## Code Style

### Python

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check
ruff check .
ruff format --check .

# Fix
ruff check --fix .
ruff format .
```

Configuration is in `pyproject.toml`.

### Rust

We use `rustfmt` and `clippy`:

```bash
# Format
cargo fmt

# Lint
cargo clippy --all-targets -- -D warnings
```

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make changes** and add tests
4. **Run checks**:
   ```bash
   pytest -v
   ruff check .
   cargo clippy
   ```
5. **Commit** with clear message:
   ```bash
   git commit -m "Add feature: description"
   ```
6. **Push** and create PR:
   ```bash
   git push origin feature/your-feature-name
   ```

### PR Guidelines

- One feature/fix per PR
- Include tests for new functionality
- Update documentation if needed
- Ensure CI passes

## Areas for Contribution

### Good First Issues

- Documentation improvements
- Test coverage expansion
- Error message improvements
- Code comments

### Feature Ideas

- New file format support
- Additional acceleration functions
- Performance optimizations
- Better error handling

### Performance

When contributing performance improvements:

1. Run benchmarks before changes:
   ```bash
   python scripts/benchmark.py
   mv BENCHMARK.md BENCHMARK_before.md
   ```

2. Make changes

3. Run benchmarks after:
   ```bash
   python scripts/benchmark.py
   ```

4. Include comparison in PR

## Release Process

Releases are automated via GitHub Actions:

1. Update version in `Cargo.toml` and `src/fast_axolotl/__init__.py`
2. Create GitHub Release
3. CI builds wheels for all platforms
4. Wheels published to PyPI via OIDC

## Getting Help

- **Issues**: https://github.com/axolotl-ai-cloud/fast-axolotl/issues
- **Discussions**: https://github.com/axolotl-ai-cloud/fast-axolotl/discussions

## License

By contributing, you agree that your contributions will be licensed under the MIT license.
