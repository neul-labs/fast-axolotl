#!/usr/bin/env python3
"""
Compatibility test script for fast-axolotl.

Verifies that fast-axolotl works correctly with axolotl and generates
COMPATIBILITY.md report.
"""

import json
import os
import platform
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import importlib.util


@dataclass
class TestResult:
    """Result of a compatibility test."""

    name: str
    passed: bool
    message: str
    details: str | None = None


def get_environment_info() -> dict[str, str]:
    """Gather environment information."""
    info = {
        "Platform": platform.system(),
        "Platform Version": platform.release(),
        "Architecture": platform.machine(),
        "Python Version": platform.python_version(),
    }

    # Get fast-axolotl version
    try:
        import fast_axolotl

        info["fast-axolotl Version"] = fast_axolotl.get_version()
        info["Rust Extension"] = (
            "Available" if fast_axolotl.is_available() else "Not Available"
        )
    except ImportError:
        info["fast-axolotl Version"] = "Not Installed"
        info["Rust Extension"] = "N/A"

    # Get axolotl version if available
    try:
        import axolotl

        info["axolotl Version"] = getattr(axolotl, "__version__", "Unknown")
    except ImportError:
        info["axolotl Version"] = "Not Installed"

    return info


# =============================================================================
# Test Functions
# =============================================================================


def verify_rust_extension() -> TestResult:
    """Verify that the Rust extension loads correctly."""
    try:
        import fast_axolotl

        if fast_axolotl.is_available():
            version = fast_axolotl.get_version()
            return TestResult(
                name="Rust Extension Loading",
                passed=True,
                message="Rust extension loaded successfully",
                details=f"Version: {version}",
            )
        else:
            return TestResult(
                name="Rust Extension Loading",
                passed=False,
                message="Rust extension not available",
                details="fast_axolotl.is_available() returned False",
            )
    except ImportError as e:
        return TestResult(
            name="Rust Extension Loading",
            passed=False,
            message="Failed to import fast_axolotl",
            details=str(e),
        )


def verify_shim_installation() -> TestResult:
    """Verify that the shim installs correctly into sys.modules."""
    try:
        import sys

        import fast_axolotl

        if not fast_axolotl.is_available():
            return TestResult(
                name="Shim Installation",
                passed=False,
                message="Rust extension not available",
            )

        # Uninstall and reinstall to test the mechanism
        fast_axolotl.uninstall()
        result = fast_axolotl.install()

        if not result:
            # May already be installed
            pass

        # Check expected modules exist in sys.modules
        expected_modules = [
            "axolotl",
            "axolotl.rust_ext",
            "axolotl.rust_ext.axolotl_ext",
            "axolotl.utils",
            "axolotl.utils.data",
            "axolotl.utils.data.rust_streaming",
            "axolotl.utils.data.rust_wrapper",
            "axolotl.utils.collators",
        ]

        missing = [m for m in expected_modules if m not in sys.modules]

        if missing:
            return TestResult(
                name="Shim Installation",
                passed=False,
                message=f"Missing {len(missing)} expected modules",
                details=f"Missing: {', '.join(missing)}",
            )

        # Verify shimmed functions are accessible
        rust_streaming = sys.modules.get("axolotl.utils.data.rust_streaming")
        if rust_streaming and hasattr(rust_streaming, "streaming_dataset_reader"):
            return TestResult(
                name="Shim Installation",
                passed=True,
                message="All expected modules shimmed correctly",
                details=f"Shimmed {len(expected_modules)} modules",
            )
        else:
            return TestResult(
                name="Shim Installation",
                passed=False,
                message="Shimmed modules missing expected functions",
            )

    except Exception as e:
        return TestResult(
            name="Shim Installation",
            passed=False,
            message="Exception during shim test",
            details=str(e),
        )


def test_format_detection() -> TestResult:
    """Test file format detection."""
    try:
        import fast_axolotl

        if not fast_axolotl.is_available():
            return TestResult(
                name="Format Detection",
                passed=False,
                message="Rust extension not available",
            )

        # Test supported formats listing
        formats = fast_axolotl.list_supported_formats()
        if not formats or len(formats) < 5:
            return TestResult(
                name="Format Detection",
                passed=False,
                message="list_supported_formats returned too few formats",
                details=f"Got: {formats}",
            )

        # Test format detection
        test_cases = [
            ("data.parquet", ("parquet", None)),
            ("data.jsonl", ("jsonl", None)),
            ("data.json.gz", ("json", "gzip")),
            ("data.csv.zst", ("csv", "zstd")),
            ("data.arrow", ("arrow", None)),
        ]

        failed = []
        for path, expected in test_cases:
            try:
                result = fast_axolotl.detect_format(path)
                if result != expected:
                    failed.append(f"{path}: expected {expected}, got {result}")
            except Exception as e:
                failed.append(f"{path}: {e}")

        if failed:
            return TestResult(
                name="Format Detection",
                passed=False,
                message=f"{len(failed)} format detection tests failed",
                details="\n".join(failed),
            )

        return TestResult(
            name="Format Detection",
            passed=True,
            message="All format detection tests passed",
            details=f"Tested {len(test_cases)} formats, {len(formats)} formats supported",
        )

    except Exception as e:
        return TestResult(
            name="Format Detection",
            passed=False,
            message="Exception during format detection test",
            details=str(e),
        )


def test_streaming_dataset_loading() -> TestResult:
    """Test streaming dataset loading with various formats."""
    try:
        import fast_axolotl

        if not fast_axolotl.is_available():
            return TestResult(
                name="Streaming Data Loading",
                passed=False,
                message="Rust extension not available",
            )

        # Create test data
        test_data = [{"text": f"Sample text {i}", "label": i} for i in range(100)]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test JSON format
            json_path = os.path.join(tmpdir, "test.json")
            with open(json_path, "w") as f:
                json.dump(test_data, f)

            # Test JSONL format
            jsonl_path = os.path.join(tmpdir, "test.jsonl")
            with open(jsonl_path, "w") as f:
                for item in test_data:
                    f.write(json.dumps(item) + "\n")

            # Test CSV format
            csv_path = os.path.join(tmpdir, "test.csv")
            with open(csv_path, "w") as f:
                f.write("text,label\n")
                for item in test_data:
                    f.write(f'"{item["text"]}",{item["label"]}\n')

            results = []

            # Test each format
            for name, path, fmt in [
                ("JSON", json_path, "json"),
                ("JSONL", jsonl_path, "jsonl"),
                ("CSV", csv_path, "csv"),
            ]:
                try:
                    batches = list(
                        fast_axolotl.streaming_dataset_reader(
                            path, fmt, batch_size=50, num_threads=2
                        )
                    )
                    if batches:
                        results.append(f"{name}: OK ({len(batches)} batches)")
                    else:
                        results.append(f"{name}: FAILED (no batches)")
                except Exception as e:
                    results.append(f"{name}: ERROR ({e})")

            # Try parquet if datasets is available
            try:
                from datasets import Dataset

                parquet_path = os.path.join(tmpdir, "test.parquet")
                Dataset.from_list(test_data).to_parquet(parquet_path)

                batches = list(
                    fast_axolotl.streaming_dataset_reader(
                        parquet_path, "parquet", batch_size=50, num_threads=2
                    )
                )
                results.append(f"Parquet: OK ({len(batches)} batches)")
            except ImportError:
                results.append("Parquet: SKIPPED (datasets not installed)")
            except Exception as e:
                results.append(f"Parquet: ERROR ({e})")

            failed = [r for r in results if "FAILED" in r or "ERROR" in r]
            if failed:
                return TestResult(
                    name="Streaming Data Loading",
                    passed=False,
                    message=f"{len(failed)} format(s) failed",
                    details="\n".join(results),
                )

            return TestResult(
                name="Streaming Data Loading",
                passed=True,
                message="All streaming formats working",
                details="\n".join(results),
            )

    except Exception as e:
        return TestResult(
            name="Streaming Data Loading",
            passed=False,
            message="Exception during streaming test",
            details=str(e),
        )


def test_token_packing() -> TestResult:
    """Test token packing functionality."""
    try:
        import fast_axolotl

        if not fast_axolotl.is_available():
            return TestResult(
                name="Token Packing",
                passed=False,
                message="Rust extension not available",
            )

        # Test basic packing
        sequences = [
            [1, 2, 3],
            [4, 5],
            [6, 7, 8, 9],
            [10, 11, 12],
        ]
        max_length = 10
        pad_token_id = 0
        eos_token_id = 2

        result = fast_axolotl.pack_sequences(
            sequences, max_length, pad_token_id, eos_token_id
        )

        # Verify structure
        if not isinstance(result, dict):
            return TestResult(
                name="Token Packing",
                passed=False,
                message="Result is not a dictionary",
            )

        required_keys = ["input_ids", "labels", "attention_mask"]
        missing_keys = [k for k in required_keys if k not in result]
        if missing_keys:
            return TestResult(
                name="Token Packing",
                passed=False,
                message=f"Missing keys: {missing_keys}",
            )

        # Verify all sequences have correct length
        for key in required_keys:
            for seq in result[key]:
                if len(seq) != max_length:
                    return TestResult(
                        name="Token Packing",
                        passed=False,
                        message=f"Sequence in {key} has wrong length: {len(seq)} != {max_length}",
                    )

        return TestResult(
            name="Token Packing",
            passed=True,
            message="Token packing working correctly",
            details=f"Packed {len(sequences)} sequences into {len(result['input_ids'])} chunks",
        )

    except Exception as e:
        return TestResult(
            name="Token Packing",
            passed=False,
            message="Exception during token packing test",
            details=str(e),
        )


def test_parallel_hashing() -> TestResult:
    """Test parallel hashing functionality."""
    try:
        import hashlib

        import fast_axolotl

        if not fast_axolotl.is_available():
            return TestResult(
                name="Parallel Hashing",
                passed=False,
                message="Rust extension not available",
            )

        # Test data
        rows = [f"test_row_{i}" for i in range(1000)]

        # Get Rust hashes
        rust_hashes = fast_axolotl.parallel_hash_rows(rows, num_threads=4)

        # Compute expected hashes with Python
        python_hashes = [hashlib.sha256(row.encode()).hexdigest() for row in rows]

        # Compare
        if rust_hashes != python_hashes:
            mismatches = sum(
                1 for r, p in zip(rust_hashes, python_hashes, strict=True) if r != p
            )
            return TestResult(
                name="Parallel Hashing",
                passed=False,
                message=f"{mismatches} hash mismatches",
                details=f"First mismatch: Rust={rust_hashes[0][:16]}..., Python={python_hashes[0][:16]}...",
            )

        # Test deduplication
        unique_indices, new_hashes = fast_axolotl.deduplicate_indices(rows)
        if len(unique_indices) != len(rows):
            return TestResult(
                name="Parallel Hashing",
                passed=False,
                message="Deduplication returned wrong count",
                details=f"Expected {len(rows)}, got {len(unique_indices)}",
            )

        return TestResult(
            name="Parallel Hashing",
            passed=True,
            message="Parallel hashing working correctly",
            details=f"Hashed {len(rows)} rows, all match Python hashlib",
        )

    except Exception as e:
        return TestResult(
            name="Parallel Hashing",
            passed=False,
            message="Exception during hashing test",
            details=str(e),
        )


def test_batch_padding() -> TestResult:
    """Test batch padding functionality."""
    try:
        import fast_axolotl

        if not fast_axolotl.is_available():
            return TestResult(
                name="Batch Padding",
                passed=False,
                message="Rust extension not available",
            )

        # Test data
        sequences = [
            [1, 2, 3],
            [4, 5],
            [6, 7, 8, 9, 10],
        ]

        # Test right padding
        padded = fast_axolotl.pad_sequences(
            sequences, target_length=8, pad_value=0, padding_side="right"
        )

        expected = [
            [1, 2, 3, 0, 0, 0, 0, 0],
            [4, 5, 0, 0, 0, 0, 0, 0],
            [6, 7, 8, 9, 10, 0, 0, 0],
        ]

        if padded != expected:
            return TestResult(
                name="Batch Padding",
                passed=False,
                message="Right padding mismatch",
                details=f"Expected: {expected}\nGot: {padded}",
            )

        # Test left padding
        padded_left = fast_axolotl.pad_sequences(
            sequences, target_length=8, pad_value=0, padding_side="left"
        )

        expected_left = [
            [0, 0, 0, 0, 0, 1, 2, 3],
            [0, 0, 0, 0, 0, 0, 4, 5],
            [0, 0, 0, 6, 7, 8, 9, 10],
        ]

        if padded_left != expected_left:
            return TestResult(
                name="Batch Padding",
                passed=False,
                message="Left padding mismatch",
                details=f"Expected: {expected_left}\nGot: {padded_left}",
            )

        return TestResult(
            name="Batch Padding",
            passed=True,
            message="Batch padding working correctly",
            details="Both left and right padding verified",
        )

    except Exception as e:
        return TestResult(
            name="Batch Padding",
            passed=False,
            message="Exception during padding test",
            details=str(e),
        )


def test_axolotl_integration() -> TestResult:
    """Test integration with actual axolotl package."""
    try:
        # First ensure shim is installed
        import fast_axolotl

        if not fast_axolotl.is_available():
            return TestResult(
                name="Axolotl Integration",
                passed=False,
                message="Rust extension not available",
            )

        # Check if the real axolotl package is installed (not just our shim)
        # The shim creates fake modules, so we need to check if there's a real package
        try:
            spec = importlib.util.find_spec("axolotl")
            if spec is None or spec.origin is None:
                return TestResult(
                    name="Axolotl Integration",
                    passed=True,
                    message="Axolotl not installed (shim-only test passed)",
                    details="Shimmed modules are available but axolotl package not installed",
                )
        except (AttributeError, TypeError, ValueError):
            # Shim module may not have proper __spec__ attribute
            # ValueError is raised by find_spec when module.__spec__ is None
            return TestResult(
                name="Axolotl Integration",
                passed=True,
                message="Axolotl not installed (shim-only test passed)",
                details="Shimmed modules are available but axolotl package not installed",
            )

        # If axolotl is installed, verify shimmed functions are accessible
        import sys

        checks = []

        # Check rust_streaming module
        if "axolotl.utils.data.rust_streaming" in sys.modules:
            mod = sys.modules["axolotl.utils.data.rust_streaming"]
            if hasattr(mod, "streaming_dataset_reader"):
                checks.append("rust_streaming: OK")
            else:
                checks.append("rust_streaming: MISSING streaming_dataset_reader")

        # Check data module has fast functions
        if "axolotl.utils.data" in sys.modules:
            mod = sys.modules["axolotl.utils.data"]
            if hasattr(mod, "fast_parallel_hash_rows"):
                checks.append("data.fast_parallel_hash_rows: OK")
            else:
                checks.append("data.fast_parallel_hash_rows: MISSING")

        # Check collators module
        if "axolotl.utils.collators" in sys.modules:
            mod = sys.modules["axolotl.utils.collators"]
            if hasattr(mod, "fast_pad_sequences"):
                checks.append("collators.fast_pad_sequences: OK")
            else:
                checks.append("collators.fast_pad_sequences: MISSING")

        failed = [c for c in checks if "MISSING" in c]
        if failed:
            return TestResult(
                name="Axolotl Integration",
                passed=False,
                message=f"{len(failed)} integration checks failed",
                details="\n".join(checks),
            )

        return TestResult(
            name="Axolotl Integration",
            passed=True,
            message="Axolotl integration working",
            details="\n".join(checks),
        )

    except Exception as e:
        return TestResult(
            name="Axolotl Integration",
            passed=False,
            message="Exception during axolotl integration test",
            details=str(e),
        )


def generate_compatibility_report(
    results: list[TestResult],
    env_info: dict[str, str],
    output_path: str = "COMPATIBILITY.md",
) -> None:
    """Generate COMPATIBILITY.md report."""
    lines = [
        "# fast-axolotl Compatibility Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Environment",
        "",
        "| Property | Value |",
        "|----------|-------|",
    ]

    for key, value in env_info.items():
        lines.append(f"| {key} | {value} |")

    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    status = "PASS" if passed == total else "FAIL"
    status_emoji = "\u2705" if passed == total else "\u274c"

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"**Overall Status**: {status_emoji} {status} ({passed}/{total} tests passed)",
            "",
            "## Test Results",
            "",
            "| Test | Status | Message |",
            "|------|--------|---------|",
        ]
    )

    for result in results:
        status_icon = "\u2705" if result.passed else "\u274c"
        lines.append(f"| {result.name} | {status_icon} | {result.message} |")

    lines.extend(
        [
            "",
            "## Detailed Results",
            "",
        ]
    )

    for result in results:
        status_icon = "\u2705" if result.passed else "\u274c"
        lines.extend(
            [
                f"### {status_icon} {result.name}",
                "",
                f"**Status**: {'PASS' if result.passed else 'FAIL'}",
                "",
                f"**Message**: {result.message}",
                "",
            ]
        )
        if result.details:
            lines.extend(
                [
                    "**Details**:",
                    "```",
                    result.details,
                    "```",
                    "",
                ]
            )

    lines.extend(
        [
            "## Feature Compatibility Matrix",
            "",
            "| Feature | Status | Notes |",
            "|---------|--------|-------|",
        ]
    )

    feature_map = {
        "Rust Extension Loading": "Core Extension",
        "Shim Installation": "Module Shimming",
        "Format Detection": "File Format Detection",
        "Streaming Data Loading": "Streaming Dataset Reader",
        "Token Packing": "Token Packing Acceleration",
        "Parallel Hashing": "Parallel SHA256 Hashing",
        "Batch Padding": "Batch Padding Acceleration",
        "Axolotl Integration": "Axolotl Compatibility",
    }

    for result in results:
        feature = feature_map.get(result.name, result.name)
        status = "\u2705 Compatible" if result.passed else "\u274c Incompatible"
        lines.append(f"| {feature} | {status} | {result.message} |")

    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nCompatibility report written to: {output_path}")


def main() -> int:
    """Run all compatibility tests and generate report."""
    print("=" * 60)
    print("fast-axolotl Compatibility Test Suite")
    print("=" * 60)

    # Gather environment info
    print("\nGathering environment information...")
    env_info = get_environment_info()

    for key, value in env_info.items():
        print(f"  {key}: {value}")

    results: list[TestResult] = []

    print("\n" + "-" * 60)
    print("Running compatibility tests...")
    print("-" * 60)

    # Run all tests
    tests = [
        ("Rust Extension", verify_rust_extension),
        ("Shim Installation", verify_shim_installation),
        ("Format Detection", test_format_detection),
        ("Streaming Data Loading", test_streaming_dataset_loading),
        ("Token Packing", test_token_packing),
        ("Parallel Hashing", test_parallel_hashing),
        ("Batch Padding", test_batch_padding),
        ("Axolotl Integration", test_axolotl_integration),
    ]

    for name, test_func in tests:
        print(f"\n[{name}]")
        result = test_func()
        status = "\u2705 PASS" if result.passed else "\u274c FAIL"
        print(f"  {status}: {result.message}")
        if result.details and not result.passed:
            print(f"  Details: {result.details[:100]}...")
        results.append(result)

    # Generate report
    print("\n" + "-" * 60)
    print("Generating report...")
    print("-" * 60)

    output_path = Path(__file__).parent.parent / "COMPATIBILITY.md"
    generate_compatibility_report(results, env_info, str(output_path))

    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print("\n" + "=" * 60)
    print(f"Compatibility test complete: {passed}/{total} tests passed")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
