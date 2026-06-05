# Auto-Shimming

`fast-axolotl` transparently replaces a handful of Axolotl modules and
functions with their Rust-backed equivalents. This page documents exactly
what the shim touches and how to control it.

## How it works

On `import fast_axolotl`, if the Rust extension loaded successfully,
`install()` is called automatically. The shim:

1. Creates virtual modules in `sys.modules` so subsequent
   `import axolotl.utils...` calls resolve to fast-axolotl's wrappers.
2. Binds the Rust functions onto those modules.
3. Leaves a `__fast_axolotl_shimmed__` attribute on each so the install
   path is idempotent.

This means existing Axolotl code that calls `from axolotl.utils.data import
fast_parallel_hash_rows` keeps working without changes.

## What gets shimmed

The shim installs the following entries (taken straight from
`src/fast_axolotl/__init__.py`):

| Module | Symbol(s) installed | Backed by |
|---|---|---|
| `axolotl.rust_ext` | sub-package marker | - |
| `axolotl.rust_ext.axolotl_ext` | the Rust extension itself | `_rust_ext` |
| `axolotl.utils` | sub-package marker | - |
| `axolotl.utils.data` | `fast_parallel_hash_rows`, `fast_deduplicate_indices` | `parallel_hash_rows`, `deduplicate_indices` |
| `axolotl.utils.data.rust_streaming` | `streaming_dataset_reader`, `RustStreamingDataset`, `create_rust_streaming_dataset`, `RUST_EXTENSION_AVAILABLE` | the streaming reader |
| `axolotl.utils.data.rust_wrapper` | wrapper helpers | streaming helpers |
| `axolotl.utils.collators` | `fast_pad_sequences`, `fast_create_padding_mask` | `pad_sequences`, `create_padding_mask` |

## Controlling the shim

```python
import fast_axolotl

fast_axolotl.is_available()   # True if the Rust extension loaded
fast_axolotl.install()        # idempotent: returns True if it installed, False otherwise
fast_axolotl.uninstall()      # removes shim entries from sys.modules
```

Auto-install happens at the very bottom of `__init__.py`:

```python
# from src/fast_axolotl/__init__.py
if RUST_AVAILABLE:
    install()
```

If you need a clean `axolotl` namespace (for example to compare against the
unaccelerated baseline) call `uninstall()` before importing `axolotl` for
the first time.

## Import order

The shim only takes effect if `fast_axolotl` is imported **before** the
Axolotl modules it patches. Once a real `axolotl.utils.data` has been
loaded into `sys.modules`, the shim will not override it.

```python
import fast_axolotl   # first
import axolotl        # then
```

If you must import `axolotl` first, call `fast_axolotl.install()`
explicitly afterward - it will overwrite the entries.

## Checking that the shim is active

```python
import sys
import fast_axolotl

assert fast_axolotl.is_available()
mod = sys.modules.get("axolotl.utils.data.rust_streaming")
assert getattr(mod, "__fast_axolotl_shimmed__", False)
```

## When the Rust extension is missing

If `_rust_ext` did not load (build failure, unsupported platform, etc.):

- `fast_axolotl.is_available()` returns `False`
- `install()` does nothing and returns `False`
- The direct Python API functions raise `ImportError` when called

In this state your code falls back to whatever Axolotl ships natively. The
package itself still imports cleanly.

## See also

- [Installation troubleshooting](../getting-started/installation.md#troubleshooting)
- [Core API](../api-reference/core.md)
