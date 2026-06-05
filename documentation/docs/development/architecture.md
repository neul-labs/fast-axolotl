# Architecture

`fast-axolotl` is a hybrid Python + Rust package. This page describes the
moving parts and how they fit together.

## Stack overview

```
┌──────────────────────────────────────────────────────────────┐
│  User code                                                   │
│  import fast_axolotl ; import axolotl                        │
├──────────────────────────────────────────────────────────────┤
│  Python API                                                  │
│  src/fast_axolotl/__init__.py    (wrappers, shim, fallbacks) │
│  src/fast_axolotl/streaming.py   (streaming helpers)         │
├──────────────────────────────────────────────────────────────┤
│  PyO3 bindings                                               │
│  src/lib.rs  #[pyfunction] / #[pymodule] _rust_ext           │
├──────────────────────────────────────────────────────────────┤
│  Rust core                                                   │
│  streaming readers (parquet/arrow/csv/json/text)             │
│  pack_sequences / concatenate_and_pack                       │
│  parallel_hash_rows / deduplicate_indices                    │
│  pad_sequences / create_padding_mask                         │
├──────────────────────────────────────────────────────────────┤
│  Rust dependencies                                           │
│  arrow / parquet (54), tokio, sha2, zstd, flate2, ...        │
└──────────────────────────────────────────────────────────────┘
```

## Components

### Python layer

`src/fast_axolotl/__init__.py` contains the entire public Python API plus
the shim. Each public function is a thin wrapper around an `_rust_ext`
binding that:

1. Checks `RUST_AVAILABLE` and raises `ImportError` if the extension is
   missing.
2. Forwards arguments unchanged to Rust.
3. Returns whatever Rust returns - typically lists or dicts of native
   Python types.

Shim install is run automatically at the bottom of the module if the Rust
extension loaded.

`src/fast_axolotl/streaming.py` holds small Python-side streaming helpers
used by `should_use_rust_streaming` and `create_rust_streaming_dataset`.

### PyO3 bindings

`src/lib.rs` builds an extension module named `_rust_ext` (per
`pyproject.toml`'s `[tool.maturin]` section). It registers every public
function with `#[pyfunction]` and exposes them on the module:

```rust
#[pymodule]
fn _rust_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    m.add_function(wrap_pyfunction!(list_supported_formats, m)?)?;
    m.add_function(wrap_pyfunction!(detect_format, m)?)?;
    m.add_function(wrap_pyfunction!(streaming_dataset_reader, m)?)?;
    m.add_function(wrap_pyfunction!(pack_sequences, m)?)?;
    // ...
    Ok(())
}
```

A unified `FastAxolotlError` enum (using `thiserror`) maps Rust failures to
the most appropriate Python exception:

| Rust variant | Python exception |
|---|---|
| `FileNotFound` | `FileNotFoundError` |
| `PermissionDenied` | `PermissionError` |
| `InvalidArgument` | `ValueError` |
| everything else | `RuntimeError` |

### Rust core

The Rust implementations cluster around four concerns:

- **Streaming readers** - one async function per format
  (`read_parquet_streaming`, `read_arrow_streaming`,
  `read_feather_streaming`, `read_csv_streaming`,
  `read_json_streaming`, `read_jsonl_from_reader`,
  `read_hf_dataset_streaming`) driven through a Tokio runtime. Output
  flows back as Arrow `RecordBatch` values converted to Python dicts via
  `record_batch_to_hashmap`.
- **Token packing** - `pack_sequences` and `concatenate_and_pack` use
  pre-allocated buffers and avoid Python list churn.
- **Hashing** - `parallel_hash_rows` and `deduplicate_indices` use a
  manual thread pool (Rust threads) feeding SHA256 digests from the
  `sha2` crate.
- **Padding** - `pad_sequences` and `create_padding_mask` are simple
  buffer-fills with left/right side support and an optional
  `pad_to_multiple_of` knob.

### Streaming sub-readers

```
streaming_dataset_reader
  └─ detect_format()  →  (format, compression)
  └─ Tokio runtime
       └─ read_dataset_streaming
            ├─ read_parquet_streaming
            ├─ read_arrow_streaming
            ├─ read_feather_streaming
            ├─ read_csv_streaming
            ├─ read_json_streaming    →  read_json_from_reader
            ├─ read_jsonl_from_reader
            └─ read_hf_dataset_streaming
```

Compression (ZSTD via `zstd`, Gzip via `flate2`) is layered transparently
under the format readers through a `CompressedReader` wrapper.

## Shimming

The shim installs entries in `sys.modules` so subsequent
`import axolotl.utils...` statements resolve to fast-axolotl's wrappers:

| `sys.modules` key | Installed by |
|---|---|
| `axolotl.rust_ext` | `_install_rust_ext_shim` |
| `axolotl.rust_ext.axolotl_ext` | `_install_rust_ext_shim` |
| `axolotl.utils.data.rust_streaming` | `_install_rust_streaming_shim` |
| `axolotl.utils.data.rust_wrapper` | `_install_rust_wrapper_shim` |
| `axolotl.utils.data` | `_install_data_utils_shim` |
| `axolotl.utils.collators` | `_install_collators_shim` |

Each installer marks its module with `__fast_axolotl_shimmed__ = True` so
`install()` and `uninstall()` are idempotent.

## Data flow

### Streaming read

```
1. user calls streaming_dataset_reader(path, "parquet", batch_size=1000)
2. Python wrapper checks RUST_AVAILABLE
3. PyO3 marshals the call into Rust
4. Rust detects format/compression
5. Tokio runtime opens the file and starts emitting RecordBatches
6. each batch is converted into a Python dict and yielded
```

### Parallel hash

```
1. user calls parallel_hash_rows(["row1", "row2", ...], num_threads=0)
2. Rust spawns N worker threads (N = num_threads or available cores)
3. each worker hashes a slice with sha2::Sha256
4. results are reassembled in input order and returned as Vec<String>
5. PyO3 converts to a Python list of hex strings
```

## Build

- Build backend: `maturin` (declared in `pyproject.toml`)
- Module name: `fast_axolotl._rust_ext`
- Python source: `src/`
- Crate type: `cdylib`
- Release profile: `lto = true`, `codegen-units = 1`, `opt-level = 3`

## Rust dependencies (`Cargo.toml`)

Selected dependencies and what they buy us:

| Crate | Used for |
|---|---|
| `pyo3` 0.23 | Python interop, extension module |
| `arrow` / `parquet` 54 | columnar reading |
| `arrow-csv`, `arrow-json` | CSV / JSON Arrow integration |
| `tokio` 1 | async runtime for streaming readers |
| `futures` 0.3 | combinators on async streams |
| `sha2` 0.10 | SHA256 hashing |
| `hex` 0.4 | hex-encode digests |
| `zstd` 0.13 / `flate2` 1.0 | ZSTD / Gzip decompression |
| `csv` 1.3 | CSV parsing |
| `serde` / `serde_json` 1.0 | JSON parsing |
| `walkdir` 2.4 | HF dataset directory traversal |
| `regex` 1.10 | format-detection patterns |
| `thiserror` 2 | structured `FastAxolotlError` |

## Performance characteristics

### Memory

- Streaming: never materialises the full dataset
- Arrow batches use minimal copies when going Python-side
- Padding/packing pre-allocates output buffers

### CPU

- Native code, no GIL contention during heavy work
- Multi-threaded readers, hashing, decompression
- Release profile compiles with LTO and `opt-level = 3`

### I/O

- Tokio multi-thread runtime for parallel file reads
- Columnar Parquet reads can project only requested columns
- ZSTD and Gzip decompression streaming

## Extension points

To add a new file format:

1. Extend `detect_format_and_compression` in `lib.rs`.
2. Add a new `read_<format>_streaming` async function.
3. Wire it into `read_dataset_streaming`'s dispatch.
4. Update `list_supported_formats`.
5. Add a test in `tests/test_fast_axolotl.py`.

To add a new processing function:

1. Implement it in `lib.rs` with `#[pyfunction]`.
2. Register it in the `#[pymodule]` block.
3. Import it in `src/fast_axolotl/__init__.py` and expose via `__all__`.
4. If Axolotl already has a name you can override, add a shim installer.

## See also

- [Contributing](contributing.md) - dev workflow
- [Core API](../api-reference/core.md) - the surface this architecture supports
