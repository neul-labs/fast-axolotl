#!/usr/bin/env python3
"""
Benchmark script for fast-axolotl.

Compares performance of Rust-accelerated functions against pure Python baselines.
Generates BENCHMARK.md with results.
"""

import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""

    name: str
    rust_time: float
    python_time: float
    speedup: float
    iterations: int
    data_size: str


def get_system_info() -> dict[str, str]:
    """Gather system information for the report."""
    import multiprocessing

    info = {
        "Platform": platform.system(),
        "Platform Release": platform.release(),
        "Architecture": platform.machine(),
        "Processor": platform.processor() or "Unknown",
        "CPU Cores": str(multiprocessing.cpu_count()),
        "Python Version": platform.python_version(),
    }

    # Try to get memory info
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_kb = int(line.split()[1])
                        info["Memory"] = f"{mem_kb // 1024 // 1024} GB"
                        break
        elif platform.system() == "Darwin":
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
            )
            mem_bytes = int(result.stdout.strip())
            info["Memory"] = f"{mem_bytes // 1024 // 1024 // 1024} GB"
    except Exception:
        info["Memory"] = "Unknown"

    # Get fast-axolotl version
    try:
        import fast_axolotl

        info["fast-axolotl Version"] = fast_axolotl.get_version()
    except Exception:
        info["fast-axolotl Version"] = "Unknown"

    return info


def timeit(func: Callable, iterations: int = 10) -> float:
    """Time a function over multiple iterations."""
    # Warmup
    func()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)

    return sum(times) / len(times)


# =============================================================================
# Baseline Python Implementations
# =============================================================================


def python_pack_sequences(
    sequences: list[list[int]],
    max_length: int,
    pad_token_id: int,
    eos_token_id: int,
    label_pad_id: int = -100,
) -> dict[str, list[list[int]]]:
    """Pure Python implementation of sequence packing."""
    packed_input_ids = []
    packed_labels = []
    packed_attention_mask = []

    current_input_ids = []
    current_labels = []
    current_attention_mask = []

    for seq in sequences:
        if len(current_input_ids) + len(seq) + 1 > max_length:
            # Pad current batch
            pad_len = max_length - len(current_input_ids)
            current_input_ids.extend([pad_token_id] * pad_len)
            current_labels.extend([label_pad_id] * pad_len)
            current_attention_mask.extend([0] * pad_len)

            packed_input_ids.append(current_input_ids)
            packed_labels.append(current_labels)
            packed_attention_mask.append(current_attention_mask)

            current_input_ids = []
            current_labels = []
            current_attention_mask = []

        current_input_ids.extend(seq)
        current_input_ids.append(eos_token_id)
        current_labels.extend(seq)
        current_labels.append(eos_token_id)
        current_attention_mask.extend([1] * (len(seq) + 1))

    # Handle remaining
    if current_input_ids:
        pad_len = max_length - len(current_input_ids)
        current_input_ids.extend([pad_token_id] * pad_len)
        current_labels.extend([label_pad_id] * pad_len)
        current_attention_mask.extend([0] * pad_len)

        packed_input_ids.append(current_input_ids)
        packed_labels.append(current_labels)
        packed_attention_mask.append(current_attention_mask)

    return {
        "input_ids": packed_input_ids,
        "labels": packed_labels,
        "attention_mask": packed_attention_mask,
    }


def python_parallel_hash_rows(rows: list[str], num_threads: int = 0) -> list[str]:
    """Pure Python implementation of row hashing."""
    return [hashlib.sha256(row.encode()).hexdigest() for row in rows]


def python_pad_sequences(
    sequences: list[list[int]],
    target_length: int | None = None,
    pad_value: int = 0,
    padding_side: str = "right",
    pad_to_multiple_of: int | None = None,
) -> list[list[int]]:
    """Pure Python implementation of sequence padding."""
    if not sequences:
        return []

    max_len = max(len(seq) for seq in sequences)
    if target_length is not None:
        max_len = max(max_len, target_length)

    if pad_to_multiple_of is not None:
        max_len = (
            (max_len + pad_to_multiple_of - 1) // pad_to_multiple_of
        ) * pad_to_multiple_of

    result = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        if padding_side == "right":
            padded = list(seq) + [pad_value] * pad_len
        else:
            padded = [pad_value] * pad_len + list(seq)
        result.append(padded)

    return result


# =============================================================================
# Benchmarks
# =============================================================================


def benchmark_streaming_loading(
    iterations: int = 5,
) -> BenchmarkResult | None:
    """Benchmark streaming data loading."""
    try:
        import fast_axolotl
        from datasets import Dataset
    except ImportError as e:
        print(f"  Skipping streaming benchmark: {e}")
        return None

    if not fast_axolotl.is_available():
        print("  Skipping streaming benchmark: Rust extension not available")
        return None

    # Create test data
    num_rows = 50000
    data = {
        "text": [
            f"This is sample text number {i} with some content."
            for i in range(num_rows)
        ],
        "label": list(range(num_rows)),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = os.path.join(tmpdir, "test_data.parquet")

        # Save as parquet
        dataset = Dataset.from_dict(data)
        dataset.to_parquet(parquet_path)

        # Benchmark Rust streaming
        def rust_load():
            rows = list(
                fast_axolotl.streaming_dataset_reader(
                    parquet_path, "parquet", batch_size=1000, num_threads=4
                )
            )
            return rows

        # Benchmark HuggingFace datasets
        def hf_load():
            from datasets import load_dataset

            ds = load_dataset("parquet", data_files=parquet_path, split="train")
            # Force iteration
            for _ in ds:
                pass
            return ds

        rust_time = timeit(rust_load, iterations)
        python_time = timeit(hf_load, iterations)

        return BenchmarkResult(
            name="Streaming Data Loading (Parquet)",
            rust_time=rust_time,
            python_time=python_time,
            speedup=python_time / rust_time if rust_time > 0 else 0,
            iterations=iterations,
            data_size=f"{num_rows:,} rows",
        )


def benchmark_token_packing(
    num_sequences: int = 10000,
    avg_seq_length: int = 50,
    max_length: int = 512,
    iterations: int = 10,
) -> BenchmarkResult | None:
    """Benchmark token packing."""
    try:
        import fast_axolotl
    except ImportError:
        print("  Skipping token packing benchmark: fast_axolotl not available")
        return None

    if not fast_axolotl.is_available():
        print("  Skipping token packing benchmark: Rust extension not available")
        return None

    import random

    random.seed(42)

    # Generate test sequences
    sequences = [
        [
            random.randint(1, 30000)
            for _ in range(random.randint(10, avg_seq_length * 2))
        ]
        for _ in range(num_sequences)
    ]

    pad_token_id = 0
    eos_token_id = 2

    def rust_pack():
        return fast_axolotl.pack_sequences(
            sequences, max_length, pad_token_id, eos_token_id
        )

    def python_pack():
        return python_pack_sequences(sequences, max_length, pad_token_id, eos_token_id)

    rust_time = timeit(rust_pack, iterations)
    python_time = timeit(python_pack, iterations)

    return BenchmarkResult(
        name="Token Packing",
        rust_time=rust_time,
        python_time=python_time,
        speedup=python_time / rust_time if rust_time > 0 else 0,
        iterations=iterations,
        data_size=f"{num_sequences:,} sequences",
    )


def benchmark_parallel_hashing(
    num_rows: int = 100000,
    iterations: int = 10,
) -> BenchmarkResult | None:
    """Benchmark parallel hashing."""
    try:
        import fast_axolotl
    except ImportError:
        print("  Skipping hashing benchmark: fast_axolotl not available")
        return None

    if not fast_axolotl.is_available():
        print("  Skipping hashing benchmark: Rust extension not available")
        return None

    # Generate test data
    rows = [
        json.dumps({"id": i, "text": f"Sample text content {i}", "value": i * 1.5})
        for i in range(num_rows)
    ]

    def rust_hash():
        return fast_axolotl.parallel_hash_rows(rows, num_threads=0)

    def python_hash():
        return python_parallel_hash_rows(rows)

    rust_time = timeit(rust_hash, iterations)
    python_time = timeit(python_hash, iterations)

    return BenchmarkResult(
        name="Parallel Hashing (SHA256)",
        rust_time=rust_time,
        python_time=python_time,
        speedup=python_time / rust_time if rust_time > 0 else 0,
        iterations=iterations,
        data_size=f"{num_rows:,} rows",
    )


def benchmark_batch_padding(
    num_sequences: int = 10000,
    max_seq_length: int = 256,
    iterations: int = 10,
) -> BenchmarkResult | None:
    """Benchmark batch padding."""
    try:
        import fast_axolotl
    except ImportError:
        print("  Skipping padding benchmark: fast_axolotl not available")
        return None

    if not fast_axolotl.is_available():
        print("  Skipping padding benchmark: Rust extension not available")
        return None

    import random

    random.seed(42)

    # Generate sequences of varying lengths
    sequences = [
        [random.randint(1, 30000) for _ in range(random.randint(10, max_seq_length))]
        for _ in range(num_sequences)
    ]

    def rust_pad():
        return fast_axolotl.pad_sequences(
            sequences, target_length=max_seq_length, pad_value=0, padding_side="right"
        )

    def python_pad():
        return python_pad_sequences(
            sequences, target_length=max_seq_length, pad_value=0, padding_side="right"
        )

    rust_time = timeit(rust_pad, iterations)
    python_time = timeit(python_pad, iterations)

    return BenchmarkResult(
        name="Batch Padding",
        rust_time=rust_time,
        python_time=python_time,
        speedup=python_time / rust_time if rust_time > 0 else 0,
        iterations=iterations,
        data_size=f"{num_sequences:,} sequences",
    )


def generate_benchmark_report(
    results: list[BenchmarkResult],
    system_info: dict[str, str],
    output_path: str = "BENCHMARK.md",
) -> None:
    """Generate BENCHMARK.md report."""
    lines = [
        "# fast-axolotl Benchmark Results",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## System Information",
        "",
        "| Property | Value |",
        "|----------|-------|",
    ]

    for key, value in system_info.items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Benchmark Results",
            "",
            "| Operation | Data Size | Rust (s) | Python (s) | Speedup |",
            "|-----------|-----------|----------|------------|---------|",
        ]
    )

    for result in results:
        speedup_str = f"{result.speedup:.2f}x"
        lines.append(
            f"| {result.name} | {result.data_size} | "
            f"{result.rust_time:.4f} | {result.python_time:.4f} | **{speedup_str}** |"
        )

    lines.extend(
        [
            "",
            "## Details",
            "",
        ]
    )

    for result in results:
        lines.extend(
            [
                f"### {result.name}",
                "",
                f"- **Data size**: {result.data_size}",
                f"- **Iterations**: {result.iterations}",
                f"- **Rust time**: {result.rust_time:.4f}s (avg)",
                f"- **Python time**: {result.python_time:.4f}s (avg)",
                f"- **Speedup**: {result.speedup:.2f}x faster",
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            "",
            "- All times are averages over multiple iterations",
            "- Rust implementations use the fast-axolotl native extension",
            "- Python baselines use standard library implementations",
            "- Speedup = Python time / Rust time",
            "",
        ]
    )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nBenchmark report written to: {output_path}")


def main() -> int:
    """Run all benchmarks and generate report."""
    print("=" * 60)
    print("fast-axolotl Benchmark Suite")
    print("=" * 60)

    # Check if fast_axolotl is available
    try:
        import fast_axolotl

        print(f"\nfast-axolotl version: {fast_axolotl.get_version()}")
        print(f"Rust extension available: {fast_axolotl.is_available()}")
    except ImportError as e:
        print(f"\nError: Could not import fast_axolotl: {e}")
        print("Please install fast-axolotl first: uv pip install -e .")
        return 1

    if not fast_axolotl.is_available():
        print("\nError: Rust extension not available.")
        print("Please build with: uv run maturin develop --release")
        return 1

    print("\nGathering system information...")
    system_info = get_system_info()

    results: list[BenchmarkResult] = []

    print("\n" + "-" * 60)
    print("Running benchmarks...")
    print("-" * 60)

    # Run each benchmark
    benchmarks = [
        ("Streaming Data Loading", benchmark_streaming_loading),
        ("Token Packing", benchmark_token_packing),
        ("Parallel Hashing", benchmark_parallel_hashing),
        ("Batch Padding", benchmark_batch_padding),
    ]

    for name, benchmark_func in benchmarks:
        print(f"\n[{name}]")
        result = benchmark_func()
        if result:
            print(f"  Rust:   {result.rust_time:.4f}s")
            print(f"  Python: {result.python_time:.4f}s")
            print(f"  Speedup: {result.speedup:.2f}x")
            results.append(result)

    if not results:
        print("\nNo benchmarks completed successfully.")
        return 1

    # Generate report
    print("\n" + "-" * 60)
    print("Generating report...")
    print("-" * 60)

    output_path = Path(__file__).parent.parent / "BENCHMARK.md"
    generate_benchmark_report(results, system_info, str(output_path))

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
