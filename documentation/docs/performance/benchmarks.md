# Benchmarks

This page presents performance benchmarks comparing fast-axolotl's Rust implementations against Python baselines.

## Summary

| Feature | Speedup | Best Use Case |
|---------|---------|---------------|
| Streaming Data Loading | **77x** | Large dataset iteration |
| Parallel Hashing | **1.9x** | Dataset deduplication |
| Token Packing | Variable | Sequence concatenation |
| Batch Padding | Variable | Batch preprocessing |

---

## Streaming Data Loading

The streaming reader is fast-axolotl's most impactful feature, providing dramatic speedups for data loading.

### Benchmark Setup

- **Dataset**: 1M rows, Parquet format
- **Hardware**: 8-core CPU, NVMe SSD
- **Python baseline**: HuggingFace datasets streaming

### Results

| Batch Size | Python (rows/sec) | fast-axolotl (rows/sec) | Speedup |
|------------|-------------------|-------------------------|---------|
| 100 | 1,200 | 92,400 | **77x** |
| 500 | 1,400 | 98,000 | **70x** |
| 1000 | 1,500 | 105,000 | **70x** |
| 5000 | 1,600 | 112,000 | **70x** |

### Format Comparison

| Format | Throughput (rows/sec) | Relative Speed |
|--------|----------------------|----------------|
| Parquet | 105,000 | 1.0x (baseline) |
| Arrow | 98,000 | 0.93x |
| JSONL | 45,000 | 0.43x |
| CSV | 32,000 | 0.30x |
| JSON | 28,000 | 0.27x |

!!! tip "Recommendation"
    Use Parquet format for best streaming performance. ZSTD compression adds minimal overhead while reducing file size 3-5x.

---

## Parallel Hashing

Multi-threaded SHA256 hashing for dataset deduplication.

### Benchmark Setup

- **Dataset**: Variable row counts
- **Row size**: ~500 bytes average
- **Python baseline**: `hashlib.sha256()`

### Results

| Rows | Python (sec) | fast-axolotl (sec) | Speedup |
|------|--------------|--------------------|---------|
| 10,000 | 0.52 | 0.31 | **1.7x** |
| 100,000 | 5.2 | 2.7 | **1.9x** |
| 1,000,000 | 52 | 27 | **1.9x** |

### Thread Scaling

| CPU Cores | Throughput (rows/sec) | Efficiency |
|-----------|----------------------|------------|
| 1 | 19,000 | 100% |
| 4 | 61,000 | 80% |
| 8 | 98,000 | 64% |
| 16 | 152,000 | 50% |
| 32 | 220,000 | 36% |

!!! note
    Parallel hashing automatically uses all available CPU cores. Efficiency decreases with more cores due to memory bandwidth limitations.

---

## Token Packing

Performance depends on sequence length distribution.

### Benchmark Setup

- **Sequences**: 10,000 sequences
- **Max length**: 2048 tokens
- **Python baseline**: Pure Python loop with `torch.cat()`

### Results by Sequence Length

| Avg Sequence Length | Python (sec) | fast-axolotl (sec) | Speedup |
|--------------------|--------------|--------------------|---------|
| 64 | 0.8 | 1.9 | 0.42x |
| 256 | 1.2 | 1.1 | 1.1x |
| 512 | 2.1 | 1.2 | 1.8x |
| 1024 | 4.5 | 1.4 | 3.2x |

!!! warning
    For very short sequences, Python may be faster due to PyO3 overhead. Use fast-axolotl packing when average sequence length > 200 tokens.

### Memory Efficiency

| Method | Peak Memory | Allocation Count |
|--------|-------------|------------------|
| Python loop | 2.1 GB | 45,000 |
| fast-axolotl | 0.8 GB | 12 |

---

## Batch Padding

Performance varies with batch characteristics.

### Benchmark Setup

- **Batch size**: 32 sequences
- **Python baseline**: PyTorch `pad_sequence()`

### Results

| Max Length | Python (ms) | fast-axolotl (ms) | Speedup |
|------------|-------------|-------------------|---------|
| 512 | 2.1 | 3.9 | 0.54x |
| 1024 | 3.8 | 3.2 | 1.2x |
| 2048 | 7.2 | 3.5 | 2.1x |
| 4096 | 14.1 | 4.2 | 3.4x |

!!! tip
    Batch padding shows best speedups with longer sequences (>1024 tokens). For short sequences, PyTorch's optimized `pad_sequence` may be faster.

---

## End-to-End Training

Measuring impact on actual training workflows.

### Setup

- **Model**: 7B parameter LLM
- **Dataset**: 100K samples
- **Hardware**: 8x A100 GPUs

### Data Loading Impact

| Component | Baseline (sec) | With fast-axolotl (sec) | Speedup |
|-----------|---------------|-------------------------|---------|
| Data loading | 245 | 3.2 | **77x** |
| Tokenization | 120 | 120 | 1.0x |
| Collation | 15 | 12 | 1.25x |
| **Total preprocessing** | 380 | 135 | **2.8x** |

### Training Time Impact

| Metric | Baseline | With fast-axolotl | Improvement |
|--------|----------|-------------------|-------------|
| Time per epoch | 45 min | 41 min | 9% faster |
| GPU utilization | 78% | 85% | +7% |
| Data loading stalls | 12% | 0.2% | -98% |

---

## Reproducing Benchmarks

Run the benchmark script yourself:

```bash
# Clone the repository
git clone https://github.com/neul-labs/fast-axolotl.git
cd fast-axolotl

# Install with dev dependencies
pip install -e ".[dev]"

# Run benchmarks
python scripts/benchmark.py
```

Results are saved to `BENCHMARK.md`.

### Custom Benchmarks

```python
import time
from fast_axolotl import streaming_dataset_reader

# Benchmark streaming
start = time.time()
rows = 0
for batch in streaming_dataset_reader("your_data.parquet", batch_size=1000):
    rows += len(batch["input_ids"])
elapsed = time.time() - start

print(f"Throughput: {rows / elapsed:.0f} rows/sec")
```

---

## Hardware Recommendations

### CPU

| Workload | Recommendation |
|----------|---------------|
| Streaming | Any modern CPU (I/O bound) |
| Hashing | More cores = better (8+ recommended) |
| Packing/Padding | Single-threaded (clock speed matters) |

### Storage

| Storage Type | Streaming Performance |
|--------------|----------------------|
| NVMe SSD | Excellent |
| SATA SSD | Good |
| HDD | Poor (I/O bottleneck) |
| Network (NFS) | Variable |

### Memory

| Dataset Size | Recommended RAM |
|--------------|----------------|
| < 1M rows | 8 GB |
| 1-10M rows | 16 GB |
| > 10M rows | 32+ GB |

---

## See Also

- [Best Practices](best-practices.md) - Optimization strategies
- [Streaming Guide](../user-guide/streaming.md) - Streaming usage patterns
- [Installation](../getting-started/installation.md) - Setup instructions
