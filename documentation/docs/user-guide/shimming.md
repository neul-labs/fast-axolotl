# Auto-Shimming System

fast-axolotl's auto-shimming system transparently replaces slow Python implementations in Axolotl with optimized Rust code - no code changes required.

## How It Works

When you call `fast_axolotl.install()`, the library:

1. Detects installed Axolotl modules
2. Replaces specific functions with Rust-accelerated versions
3. Maintains full API compatibility

```python
import fast_axolotl

# Before: Axolotl uses pure Python implementations
# After: Axolotl uses Rust-accelerated implementations
fast_axolotl.install()

# Your existing Axolotl code works unchanged
from axolotl.utils.data import load_tokenized_prepared_datasets
```

## Basic Usage

### Installing Shims

```python
import fast_axolotl

# Install all available shims
fast_axolotl.install()
```

### Uninstalling Shims

```python
# Temporarily disable acceleration
fast_axolotl.uninstall()

# Re-enable
fast_axolotl.install()
```

### Checking Status

```python
import fast_axolotl

# Check if Rust extension is available
print(f"Rust available: {fast_axolotl.rust_available()}")

# Check if shims are installed
# (After install() is called)
```

## What Gets Shimmed

fast-axolotl automatically accelerates these Axolotl components:

### 1. Streaming Data Loading

**Module**: `axolotl.utils.data.rust_streaming`

```python
# Original (Python)
from axolotl.utils.data import streaming_dataset_reader

# With shim installed, this transparently uses Rust
for batch in streaming_dataset_reader("data.parquet"):
    process(batch)
```

**Speedup**: Up to 77x faster

### 2. Dataset Deduplication

**Module**: `axolotl.utils.data` (hash-based deduplication)

```python
# Deduplication operations automatically use parallel hashing
from axolotl.utils.data import deduplicate_dataset
```

**Speedup**: ~1.9x faster

### 3. Batch Collation

**Module**: `axolotl.utils.collators`

```python
# Collators automatically use optimized padding
from axolotl.utils.collators import DataCollatorForSeq2Seq
```

**Speedup**: Variable, best with large batches

## Selective Shimming

For fine-grained control, you can install specific shims:

```python
import fast_axolotl

# Install only streaming shims
fast_axolotl.install_streaming_shim()

# Install only hashing shims
fast_axolotl.install_hashing_shim()

# Install only collator shims
fast_axolotl.install_collator_shim()
```

## Integration Patterns

### Training Script

```python
#!/usr/bin/env python
import fast_axolotl

# Install shims at the very start
fast_axolotl.install()

# Now import and use Axolotl normally
from axolotl.cli import train

if __name__ == "__main__":
    train()
```

### Jupyter Notebook

```python
# Cell 1: Setup
import fast_axolotl
fast_axolotl.install()

# Cell 2: Your Axolotl training code
from axolotl.utils.data import load_tokenized_prepared_datasets
# ...
```

### Config-Based Training

No code changes needed - just import fast_axolotl first:

```python
import fast_axolotl
fast_axolotl.install()

# Standard axolotl CLI
import subprocess
subprocess.run(["axolotl", "train", "config.yaml"])
```

## Compatibility

### Axolotl Versions

fast-axolotl is tested with:

| Axolotl Version | Status |
|-----------------|--------|
| 0.4.x | Fully supported |
| 0.3.x | Fully supported |
| 0.2.x | Partial support |

### Fallback Behavior

If Axolotl isn't installed or a module can't be shimmed:

```python
import fast_axolotl

# install() is safe to call even without Axolotl
fast_axolotl.install()  # Silently skips unavailable shims

# You can still use fast-axolotl functions directly
from fast_axolotl import streaming_dataset_reader
```

## Debugging

### Check What's Shimmed

```python
import fast_axolotl

fast_axolotl.install()

# After installation, check shim status
import sys

# Check if modules are shimmed
if "axolotl.utils.data" in sys.modules:
    module = sys.modules["axolotl.utils.data"]
    print(f"Module shimmed: {hasattr(module, '__fast_axolotl_shimmed__')}")
```

### Verbose Mode

```python
import fast_axolotl
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# install() will log what it's doing
fast_axolotl.install()
```

### Testing Without Shims

Compare performance with and without shimming:

```python
import time
import fast_axolotl
from axolotl.utils.data import some_function

# Without shims
fast_axolotl.uninstall()
start = time.time()
result1 = some_function(data)
time_without = time.time() - start

# With shims
fast_axolotl.install()
start = time.time()
result2 = some_function(data)
time_with = time.time() - start

print(f"Speedup: {time_without / time_with:.1f}x")
```

## Troubleshooting

### Shims Not Working

1. **Ensure proper import order**:
   ```python
   import fast_axolotl
   fast_axolotl.install()

   # THEN import axolotl
   import axolotl
   ```

2. **Check Rust extension**:
   ```python
   print(fast_axolotl.rust_available())  # Should be True
   ```

3. **Check Axolotl installation**:
   ```python
   import axolotl
   print(axolotl.__version__)
   ```

### Import Errors

If you see import errors after installing shims:

```python
# Uninstall and report the issue
fast_axolotl.uninstall()

# Use direct API instead
from fast_axolotl import streaming_dataset_reader
```

### Performance Not Improved

Some operations may not show improvement:

- Very small datasets (overhead dominates)
- Already optimized code paths
- GPU-bound operations (CPU acceleration won't help)

Check [Benchmarks](../performance/benchmarks.md) for expected improvements.

## Next Steps

- [Quick Start](../getting-started/quick-start.md) - Get started with examples
- [API Reference](../api-reference/core.md) - Complete API documentation
- [Benchmarks](../performance/benchmarks.md) - Performance comparisons
