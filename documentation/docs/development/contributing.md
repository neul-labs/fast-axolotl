# Contributing

Thank you for your interest in contributing to fast-axolotl! This guide will help you get started.

## Development Setup

### Prerequisites

1. **Python 3.10+**
2. **Rust toolchain** (1.70 or later):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```
3. **maturin** (Rust-Python build tool):
   ```bash
   pip install maturin
   ```

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/dipankar/fast-axolotl.git
cd fast-axolotl

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install in development mode
maturin develop

# Install dev dependencies
pip install -e ".[dev]"
```

### Verify Setup

```bash
# Run tests
pytest tests/

# Run type checks
mypy src/fast_axolotl

# Run linting
ruff check src/
```

---

## Project Structure

```
fast-axolotl/
├── src/
│   ├── lib.rs                 # Rust extension code
│   └── fast_axolotl/
│       ├── __init__.py        # Python API
│       ├── streaming.py       # Streaming utilities
│       └── py.typed           # Type hints marker
├── tests/
│   └── test_fast_axolotl.py   # Test suite
├── scripts/
│   ├── benchmark.py           # Benchmarking script
│   └── compatibility_test.py  # Compatibility tests
├── documentation/             # MkDocs documentation
├── Cargo.toml                 # Rust dependencies
├── pyproject.toml             # Python project config
└── README.md
```

---

## Making Changes

### Rust Code

Edit `src/lib.rs` for Rust extension changes:

```rust
// Example: Adding a new function
#[pyfunction]
fn my_new_function(input: &str) -> PyResult<String> {
    Ok(format!("Processed: {}", input))
}

// Register in module
#[pymodule]
fn _fast_axolotl(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(my_new_function, m)?)?;
    Ok(())
}
```

Rebuild after changes:

```bash
maturin develop --release
```

### Python Code

Edit files in `src/fast_axolotl/`:

```python
# Example: Exposing Rust function in Python API
from ._fast_axolotl import my_new_function

def my_wrapper(input: str) -> str:
    """Process input string.

    Args:
        input: String to process

    Returns:
        Processed string
    """
    return my_new_function(input)
```

---

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Tests

```bash
# Run one test file
pytest tests/test_fast_axolotl.py -v

# Run one test function
pytest tests/test_fast_axolotl.py::test_streaming_reader -v

# Run with coverage
pytest tests/ --cov=fast_axolotl --cov-report=html
```

### Writing Tests

Add tests in `tests/test_fast_axolotl.py`:

```python
import pytest
from fast_axolotl import my_new_function

def test_my_new_function():
    """Test the new function."""
    result = my_new_function("hello")
    assert result == "Processed: hello"

def test_my_new_function_edge_cases():
    """Test edge cases."""
    assert my_new_function("") == "Processed: "

    with pytest.raises(TypeError):
        my_new_function(123)  # type: ignore
```

---

## Code Style

### Python

We use `ruff` for linting and formatting:

```bash
# Check code style
ruff check src/

# Auto-fix issues
ruff check src/ --fix

# Format code
ruff format src/
```

### Rust

We use `rustfmt` and `clippy`:

```bash
# Format Rust code
cargo fmt

# Run linter
cargo clippy
```

---

## Documentation

### Building Docs Locally

```bash
cd documentation
pip install mkdocs-material mkdocstrings[python]
mkdocs serve
```

Visit `http://localhost:8000` to preview.

### Documentation Style

- Use clear, concise language
- Include code examples for all functions
- Add type annotations
- Link to related pages

---

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/my-new-feature
```

### 2. Make Changes

- Write code
- Add tests
- Update documentation

### 3. Run Checks

```bash
# All checks must pass
pytest tests/
ruff check src/
cargo clippy
cargo test
```

### 4. Commit

```bash
git add .
git commit -m "Add my new feature

- Describe what was added
- Explain why it's useful"
```

### 5. Push and Create PR

```bash
git push origin feature/my-new-feature
```

Then create a Pull Request on GitHub.

### PR Checklist

- [ ] Tests pass
- [ ] Code is formatted
- [ ] Documentation updated
- [ ] Changelog entry added (if applicable)

---

## Benchmarking

### Run Benchmarks

```bash
python scripts/benchmark.py
```

### Adding New Benchmarks

Edit `scripts/benchmark.py`:

```python
def benchmark_my_function():
    """Benchmark my new function."""
    import time
    from fast_axolotl import my_new_function

    # Setup
    data = ["test"] * 10000

    # Benchmark
    start = time.time()
    for item in data:
        my_new_function(item)
    elapsed = time.time() - start

    return {
        "items": len(data),
        "time": elapsed,
        "throughput": len(data) / elapsed
    }
```

---

## Release Process

Releases are automated via GitHub Actions:

1. Update version in `pyproject.toml` and `Cargo.toml`
2. Update `CHANGELOG.md`
3. Create a git tag: `git tag v0.x.x`
4. Push tag: `git push origin v0.x.x`

CI will build wheels and publish to PyPI.

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/dipankar/fast-axolotl/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dipankar/fast-axolotl/discussions)

---

## Code of Conduct

Please be respectful and constructive in all interactions. We're all here to build great software together.
