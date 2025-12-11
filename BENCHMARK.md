# fast-axolotl Benchmark Results

Generated: 2025-12-11 10:12:22

## System Information

| Property | Value |
|----------|-------|
| Platform | Linux |
| Platform Release | 6.17.7-x64v3-xanmod1 |
| Architecture | x86_64 |
| Processor | x86_64 |
| CPU Cores | 16 |
| Python Version | 3.11.13 |
| Memory | 62 GB |
| fast-axolotl Version | 0.2.0 (rust: 0.2.0) |

## Benchmark Results

| Operation | Data Size | Rust (s) | Python (s) | Speedup |
|-----------|-----------|----------|------------|---------|
| Streaming Data Loading (Parquet) | 50,000 rows | 0.0094 | 0.7237 | **77.26x** |
| Token Packing | 10,000 sequences | 0.0786 | 0.0327 | **0.42x** |
| Parallel Hashing (SHA256) | 100,000 rows | 0.0273 | 0.0520 | **1.90x** |
| Batch Padding | 10,000 sequences | 0.1998 | 0.1051 | **0.53x** |

## Details

### Streaming Data Loading (Parquet)

- **Data size**: 50,000 rows
- **Iterations**: 5
- **Rust time**: 0.0094s (avg)
- **Python time**: 0.7237s (avg)
- **Speedup**: 77.26x faster

### Token Packing

- **Data size**: 10,000 sequences
- **Iterations**: 10
- **Rust time**: 0.0786s (avg)
- **Python time**: 0.0327s (avg)
- **Speedup**: 0.42x faster

### Parallel Hashing (SHA256)

- **Data size**: 100,000 rows
- **Iterations**: 10
- **Rust time**: 0.0273s (avg)
- **Python time**: 0.0520s (avg)
- **Speedup**: 1.90x faster

### Batch Padding

- **Data size**: 10,000 sequences
- **Iterations**: 10
- **Rust time**: 0.1998s (avg)
- **Python time**: 0.1051s (avg)
- **Speedup**: 0.53x faster

## Notes

- All times are averages over multiple iterations
- Rust implementations use the fast-axolotl native extension
- Python baselines use standard library implementations
- Speedup = Python time / Rust time
