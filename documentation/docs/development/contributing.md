# Contributing

Thanks for considering a contribution to `fast-axolotl`. The project mixes
Python and Rust; this page covers the dev workflow for both.

## Prerequisites

- Python 3.10, 3.11, or 3.12
- A Rust toolchain (stable is fine):

  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  source ~/.cargo/env
  ```

- Git

## Get a working dev environment

```bash
git clone https://github.com/neul-labs/fast-axolotl
cd fast-axolotl

uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Build the Rust extension in place
maturin develop
```

For the release-quality build (used when you're chasing performance
numbers):

```bash
maturin develop --release
```

### Verify

```bash
pytest -v
ruff check .
cargo clippy
cargo fmt --check
```

## Repository layout

```
fast-axolotl/
├── src/
│   ├── lib.rs                  # Rust extension (PyO3 module `_rust_ext`)
│   └── fast_axolotl/
│       ├── __init__.py         # Public Python API + shim install logic
│       ├── streaming.py        # Streaming helpers
│       └── py.typed
├── tests/
│   └── test_fast_axolotl.py    # pytest suite
├── scripts/
│   ├── benchmark.py            # writes BENCHMARK.md
│   └── compatibility_test.py   # writes COMPATIBILITY.md
├── docs/                       # high-level Markdown docs (this site is in documentation/)
├── documentation/              # MkDocs site (you are here)
├── .github/workflows/          # CI
├── Cargo.toml                  # Rust crate config
└── pyproject.toml              # Python + maturin config
```

## Making changes

### Rust code

The Rust extension lives in `src/lib.rs`. Key building blocks:

- **PyO3 bindings** - `#[pyfunction]` and `#[pymodule]` wire Rust into
  Python (`_rust_ext`)
- **Streaming readers** - Parquet, Arrow, Feather, JSON, JSONL, CSV, Text
- **Token operations** - `pack_sequences`, `concatenate_and_pack`,
  `pad_sequences`, `create_padding_mask`
- **Parallel hashing** - `parallel_hash_rows`, `deduplicate_indices`

After editing Rust you must rebuild:

```bash
maturin develop          # debug
maturin develop --release  # release / benchmarks
```

### Python code

The public Python API lives in `src/fast_axolotl/__init__.py`. It:

- imports the Rust functions from `_rust_ext`
- wraps them with validation and fallbacks
- installs the shim into `sys.modules`

`__all__` at the bottom of the file is the source of truth for the public
API surface.

### Tests

Add unit tests in `tests/test_fast_axolotl.py`:

```python
def test_new_feature():
    import fast_axolotl
    result = fast_axolotl.new_function(...)
    assert result == expected
```

Run them with:

```bash
pytest -v
pytest -v -k test_new_feature
```

## Code style

### Python

[Ruff](https://github.com/astral-sh/ruff) handles both linting and
formatting. Configuration is in `pyproject.toml`:

```bash
ruff check .
ruff check --fix .
ruff format .
```

The configured line length is 88 and the target is `py310`.

### Rust

```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
```

## Pull request flow

1. Fork the repo and create a branch:
   ```bash
   git checkout -b feature/your-thing
   ```
2. Make your change. Add tests.
3. Run the full check suite:
   ```bash
   pytest -v
   ruff check .
   cargo clippy
   ```
4. Commit and push:
   ```bash
   git push origin feature/your-thing
   ```
5. Open a PR. Keep it focused on one concern; CI must pass.

## Benchmarking changes

If you're touching a hot path, capture before/after numbers:

```bash
# baseline
maturin develop --release
python scripts/benchmark.py
mv BENCHMARK.md BENCHMARK_before.md

# your change
python scripts/benchmark.py
diff BENCHMARK_before.md BENCHMARK.md
```

Paste the comparison into the PR description.

## Release process

Releases are automated by GitHub Actions:

1. Bump `version` in `Cargo.toml` and `__version__` in
   `src/fast_axolotl/__init__.py`.
2. Cut a GitHub release.
3. CI builds wheels for all supported platforms and publishes to PyPI via
   OIDC.

## Getting help

- Issues: <https://github.com/neul-labs/fast-axolotl/issues>
- Discussions: <https://github.com/neul-labs/fast-axolotl/discussions>

## License

By contributing you agree your contributions will be licensed under the
project's MIT licence.
