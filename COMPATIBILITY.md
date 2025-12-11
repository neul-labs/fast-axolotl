# fast-axolotl Compatibility Report

Generated: 2025-12-11 13:37:52

## Environment

| Property | Value |
|----------|-------|
| Platform | Linux |
| Platform Version | 6.17.7-x64v3-xanmod1 |
| Architecture | x86_64 |
| Python Version | 3.11.13 |
| fast-axolotl Version | 0.2.0 (rust: 0.2.0) |
| Rust Extension | Available |
| axolotl Version | Unknown |

## Summary

**Overall Status**: ✅ PASS (8/8 tests passed)

## Test Results

| Test | Status | Message |
|------|--------|---------|
| Rust Extension Loading | ✅ | Rust extension loaded successfully |
| Shim Installation | ✅ | All expected modules shimmed correctly |
| Format Detection | ✅ | All format detection tests passed |
| Streaming Data Loading | ✅ | All streaming formats working |
| Token Packing | ✅ | Token packing working correctly |
| Parallel Hashing | ✅ | Parallel hashing working correctly |
| Batch Padding | ✅ | Batch padding working correctly |
| Axolotl Integration | ✅ | Axolotl not installed (shim-only test passed) |

## Detailed Results

### ✅ Rust Extension Loading

**Status**: PASS

**Message**: Rust extension loaded successfully

**Details**:
```
Version: 0.2.0 (rust: 0.2.0)
```

### ✅ Shim Installation

**Status**: PASS

**Message**: All expected modules shimmed correctly

**Details**:
```
Shimmed 8 modules
```

### ✅ Format Detection

**Status**: PASS

**Message**: All format detection tests passed

**Details**:
```
Tested 5 formats, 20 formats supported
```

### ✅ Streaming Data Loading

**Status**: PASS

**Message**: All streaming formats working

**Details**:
```
JSON: OK (2 batches)
JSONL: OK (2 batches)
CSV: OK (2 batches)
Parquet: OK (2 batches)
```

### ✅ Token Packing

**Status**: PASS

**Message**: Token packing working correctly

**Details**:
```
Packed 4 sequences into 3 chunks
```

### ✅ Parallel Hashing

**Status**: PASS

**Message**: Parallel hashing working correctly

**Details**:
```
Hashed 1000 rows, all match Python hashlib
```

### ✅ Batch Padding

**Status**: PASS

**Message**: Batch padding working correctly

**Details**:
```
Both left and right padding verified
```

### ✅ Axolotl Integration

**Status**: PASS

**Message**: Axolotl not installed (shim-only test passed)

**Details**:
```
Shimmed modules are available but axolotl package not installed
```

## Feature Compatibility Matrix

| Feature | Status | Notes |
|---------|--------|-------|
| Core Extension | ✅ Compatible | Rust extension loaded successfully |
| Module Shimming | ✅ Compatible | All expected modules shimmed correctly |
| File Format Detection | ✅ Compatible | All format detection tests passed |
| Streaming Dataset Reader | ✅ Compatible | All streaming formats working |
| Token Packing Acceleration | ✅ Compatible | Token packing working correctly |
| Parallel SHA256 Hashing | ✅ Compatible | Parallel hashing working correctly |
| Batch Padding Acceleration | ✅ Compatible | Batch padding working correctly |
| Axolotl Compatibility | ✅ Compatible | Axolotl not installed (shim-only test passed) |
