# Compatibility

This document covers fast-axolotl's compatibility with different environments and configurations.

## Compatibility Status

All features tested and verified:

| Feature | Status | Notes |
|---------|--------|-------|
| Rust Extension Loading | Tested | Core extension loads correctly |
| Module Shimming | Tested | 8 modules shimmed successfully |
| Format Detection | Tested | 20 formats supported |
| Streaming (Parquet) | Tested | Full support |
| Streaming (JSON/JSONL) | Tested | Full support |
| Streaming (CSV) | Tested | Full support |
| Streaming (Arrow) | Tested | Full support |
| Token Packing | Tested | Correct output verified |
| Parallel Hashing | Tested | Matches Python hashlib |
| Batch Padding | Tested | Left and right padding |
| Axolotl Integration | Tested | Shim functions accessible |

## Supported Platforms

### Operating Systems

| OS | Architecture | Status |
|----|--------------|--------|
| Linux | x86_64 | Fully supported |
| Linux | aarch64 | Fully supported |
| macOS | x86_64 | Fully supported |
| macOS | arm64 (M1/M2) | Fully supported |
| Windows | x86_64 | Fully supported |

### Python Versions

| Version | Status |
|---------|--------|
| Python 3.10 | Fully supported |
| Python 3.11 | Fully supported |
| Python 3.12 | Fully supported |
| Python 3.9 | Not supported |
| Python 3.13+ | Not tested |

## Axolotl Compatibility

fast-axolotl is designed to work with axolotl without requiring changes to axolotl code.

### Shimmed Modules

When fast-axolotl is imported, these modules are created/patched:

```
axolotl
axolotl.rust_ext
axolotl.rust_ext.axolotl_ext
axolotl.utils
axolotl.utils.data
axolotl.utils.data.rust_streaming
axolotl.utils.data.rust_wrapper
axolotl.utils.collators
```

### Functions Provided

| Module | Function | Description |
|--------|----------|-------------|
| `rust_streaming` | `streaming_dataset_reader` | Rust streaming loader |
| `rust_streaming` | `RUST_EXTENSION_AVAILABLE` | Availability flag |
| `data` | `fast_parallel_hash_rows` | Parallel hashing |
| `data` | `fast_deduplicate_indices` | Deduplication |
| `collators` | `fast_pad_sequences` | Batch padding |
| `collators` | `fast_create_padding_mask` | Padding mask creation |

## File Format Compatibility

### Base Formats

| Format | Read | Notes |
|--------|------|-------|
| Parquet | Yes | Recommended for large datasets |
| Arrow IPC | Yes | `.arrow`, `.ipc` extensions |
| Feather | Yes | Arrow IPC v2 format |
| JSON | Yes | Array of objects |
| JSONL | Yes | JSON Lines (`.jsonl`, `.ndjson`) |
| CSV | Yes | Comma-separated values |
| TSV | Yes | Tab-separated values |
| Text | Yes | Plain text files |

### Compression Support

| Compression | Extensions | Status |
|-------------|------------|--------|
| ZSTD | `.zst`, `.zstd` | Full support |
| Gzip | `.gz`, `.gzip` | Full support |
| None | - | Full support |

### HuggingFace Datasets

Directories containing `dataset_info.json` are detected as HuggingFace Arrow datasets.

## Running Compatibility Tests

Generate a compatibility report for your environment:

```bash
cd fast-axolotl
uv run python scripts/compatibility_test.py
```

This creates `COMPATIBILITY.md` with detailed test results.

### Test Output Example

```
============================================================
fast-axolotl Compatibility Test Suite
============================================================

Gathering environment information...
  Platform: Linux
  Python Version: 3.11.13
  fast-axolotl Version: 0.2.0 (rust: 0.2.0)
  Rust Extension: Available

------------------------------------------------------------
Running compatibility tests...
------------------------------------------------------------

[Rust Extension]
  Tested PASS: Rust extension loaded successfully

[Shim Installation]
  Tested PASS: All expected modules shimmed correctly

[Format Detection]
  Tested PASS: All format detection tests passed

...

============================================================
Compatibility test complete: 8/8 tests passed
============================================================
```

## Known Issues

### Windows

- Long paths (>260 characters) may cause issues
- Use forward slashes in paths for consistency

### macOS

- First import may be slow due to code signing verification
- Subsequent imports are fast

### Linux

- glibc 2.17+ required (most modern distributions)
- musl libc not currently supported

## Dependency Compatibility

### Required Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | >=3.10 | Runtime |
| datasets | >=2.14.0 | HuggingFace datasets integration |
| numpy | >=1.24.0 | Array operations |

### Optional Dependencies

| Package | Purpose |
|---------|---------|
| axolotl | Full integration testing |
| torch | Training workflows |
| transformers | Tokenization |

## CI/CD Testing

fast-axolotl is tested on every push with:

- **CI**: All platforms (Linux, macOS, Windows) × Python (3.10, 3.11, 3.12)
- **Compatibility Tests**: Full integration tests on Linux

See `.github/workflows/` for CI configuration.

## Reporting Issues

If you encounter compatibility issues:

1. Run the compatibility test: `python scripts/compatibility_test.py`
2. Include the generated `COMPATIBILITY.md` in your issue
3. Report at: https://github.com/axolotl-ai-cloud/fast-axolotl/issues

Include:
- Operating system and version
- Python version
- fast-axolotl version (`fast_axolotl.get_version()`)
- Full error message/traceback
