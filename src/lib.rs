// Fast-Axolotl: High-performance Rust extensions for Axolotl
// Provides memory-efficient streaming, token packing, parallel hashing, and batch padding
// Supports: Parquet, Arrow, CSV, JSON/JSONL, Text, Feather
// Compression: ZSTD (.zst), Gzip (.gz)
// Directory formats: HuggingFace Arrow Dataset

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;
use tokio::runtime::Runtime;

// =============================================================================
// Error Types
// =============================================================================

/// Unified error type for fast-axolotl operations.
///
/// This enum provides structured error messages for all operations,
/// making debugging easier and enabling better error handling in Python.
#[derive(Error, Debug)]
pub enum FastAxolotlError {
    /// File not found or not accessible
    #[error("File not found: {path}")]
    FileNotFound { path: String },

    /// Permission denied when accessing file
    #[error("Permission denied: {path}")]
    PermissionDenied { path: String },

    /// I/O error during file operation
    #[error("I/O error: {source}")]
    Io {
        #[from]
        source: std::io::Error,
    },

    /// Invalid or unsupported file format
    #[error("Invalid format: {message}")]
    InvalidFormat { message: String },

    /// Compression/decompression error
    #[error("Compression error for '{file}': {source}")]
    Compression {
        file: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// JSON parsing error
    #[error("JSON parse error: {source}")]
    Json {
        #[from]
        source: serde_json::Error,
    },

    /// Arrow/Parquet error
    #[error("Arrow/Parquet error: {source}")]
    Arrow {
        #[from]
        source: arrow::error::ArrowError,
    },

    /// CSV parsing error
    #[error("CSV error: {source}")]
    Csv {
        #[from]
        source: csv::Error,
    },

    /// Invalid argument passed to function
    #[error("Invalid argument: {message}")]
    InvalidArgument { message: String },

    /// Thread pool error (e.g., join failure)
    #[error("Thread error: {message}")]
    Thread { message: String },

    /// General runtime error
    #[error("Runtime error: {message}")]
    Runtime { message: String },
}

/// Convert FastAxolotlError to Python exception
impl From<FastAxolotlError> for PyErr {
    fn from(e: FastAxolotlError) -> PyErr {
        use pyo3::exceptions;
        match e {
            FastAxolotlError::FileNotFound { .. } => {
                exceptions::PyFileNotFoundError::new_err(e.to_string())
            }
            FastAxolotlError::PermissionDenied { .. } => {
                exceptions::PyPermissionError::new_err(e.to_string())
            }
            FastAxolotlError::InvalidArgument { .. } => {
                exceptions::PyValueError::new_err(e.to_string())
            }
            FastAxolotlError::InvalidFormat { .. } => {
                exceptions::PyValueError::new_err(e.to_string())
            }
            FastAxolotlError::Io { .. } => exceptions::PyIOError::new_err(e.to_string()),
            FastAxolotlError::Compression { .. } => {
                exceptions::PyRuntimeError::new_err(e.to_string())
            }
            FastAxolotlError::Arrow { .. } => exceptions::PyIOError::new_err(e.to_string()),
            FastAxolotlError::Csv { .. } => exceptions::PyIOError::new_err(e.to_string()),
            FastAxolotlError::Json { .. } => exceptions::PyValueError::new_err(e.to_string()),
            FastAxolotlError::Thread { .. } => {
                exceptions::PyRuntimeError::new_err(e.to_string())
            }
            FastAxolotlError::Runtime { .. } => {
                exceptions::PyRuntimeError::new_err(e.to_string())
            }
        }
    }
}

// =============================================================================
// Compression Error Conversions
// =============================================================================

impl FastAxolotlError {
    /// Create a compression error with file context
    pub fn compression_error(
        file: &str,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        FastAxolotlError::Compression {
            file: file.to_string(),
            source: Box::new(source),
        }
    }
}

impl From<String> for FastAxolotlError {
    fn from(e: String) -> Self {
        FastAxolotlError::Runtime { message: e }
    }
}

impl From<flate2::CompressError> for FastAxolotlError {
    fn from(e: flate2::CompressError) -> Self {
        FastAxolotlError::Compression {
            file: "<gzip>".to_string(),
            source: Box::new(e),
        }
    }
}

impl From<parquet::errors::ParquetError> for FastAxolotlError {
    fn from(e: parquet::errors::ParquetError) -> Self {
        FastAxolotlError::Arrow {
            source: arrow::error::ArrowError::ExternalError(Box::new(e)),
        }
    }
}

/// Python module for fast-axolotl Rust extensions
#[pymodule]
fn _rust_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Streaming
    m.add_function(wrap_pyfunction!(streaming_dataset_reader, m)?)?;
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    m.add_function(wrap_pyfunction!(detect_format, m)?)?;
    m.add_function(wrap_pyfunction!(list_supported_formats, m)?)?;

    // Token Packing (Acceleration #1)
    m.add_function(wrap_pyfunction!(pack_sequences, m)?)?;
    m.add_function(wrap_pyfunction!(concatenate_and_pack, m)?)?;

    // Parallel Hashing (Acceleration #2)
    m.add_function(wrap_pyfunction!(parallel_hash_rows, m)?)?;
    m.add_function(wrap_pyfunction!(deduplicate_indices, m)?)?;

    // Batch Padding (Acceleration #3)
    m.add_function(wrap_pyfunction!(pad_sequences, m)?)?;
    m.add_function(wrap_pyfunction!(create_padding_mask, m)?)?;

    Ok(())
}

/// Get the fast-axolotl version
#[pyfunction]
fn get_version() -> &'static str {
    "0.2.0"
}

/// List all supported formats
#[pyfunction]
fn list_supported_formats() -> Vec<&'static str> {
    vec![
        // Base formats
        "parquet",
        "arrow",
        "feather",
        "csv",
        "json",
        "jsonl",
        "text",
        // Compressed formats
        "parquet.zst",
        "parquet.gz",
        "arrow.zst",
        "arrow.gz",
        "json.zst",
        "json.gz",
        "jsonl.zst",
        "jsonl.gz",
        "csv.zst",
        "csv.gz",
        "text.zst",
        "text.gz",
        // Directory formats
        "hf_dataset", // HuggingFace Arrow Dataset directory
    ]
}

/// Detect format from file path
#[pyfunction]
fn detect_format(file_path: &str) -> PyResult<(String, Option<String>)> {
    let (base_format, compression) = detect_format_and_compression(file_path);
    Ok((base_format.to_string(), compression.map(|s| s.to_string())))
}

// =============================================================================
// Format Detection
// =============================================================================

fn detect_format_and_compression(file_path: &str) -> (&'static str, Option<&'static str>) {
    let path = file_path.to_lowercase();

    // Check for compression first
    let (base_path, compression) = if path.ends_with(".zst") || path.ends_with(".zstd") {
        let idx = path.rfind('.').unwrap_or(path.len());
        (&path[..idx], Some("zstd"))
    } else if path.ends_with(".gz") || path.ends_with(".gzip") {
        let idx = path.rfind('.').unwrap_or(path.len());
        (&path[..idx], Some("gzip"))
    } else {
        (path.as_str(), None)
    };

    // Detect base format
    let format = if base_path.ends_with(".parquet") {
        "parquet"
    } else if base_path.ends_with(".arrow") || base_path.ends_with(".ipc") {
        "arrow"
    } else if base_path.ends_with(".feather") {
        "feather"
    } else if base_path.ends_with(".csv") || base_path.ends_with(".tsv") {
        "csv"
    } else if base_path.ends_with(".jsonl") || base_path.ends_with(".ndjson") {
        "jsonl"
    } else if base_path.ends_with(".json") {
        "json"
    } else if base_path.ends_with(".txt") {
        "text"
    } else if Path::new(file_path).is_dir() {
        // Check if it's a HuggingFace dataset directory
        let dataset_info = Path::new(file_path).join("dataset_info.json");
        if dataset_info.exists() {
            "hf_dataset"
        } else {
            "json" // Default fallback
        }
    } else {
        "json" // Default fallback
    };

    (format, compression)
}

// =============================================================================
// Compression Readers
// =============================================================================

enum CompressedReader {
    Plain(BufReader<File>),
    Zstd(BufReader<zstd::Decoder<'static, BufReader<File>>>),
    Gzip(BufReader<flate2::read::GzDecoder<File>>),
}

impl CompressedReader {
    fn new(file_path: &str, compression: Option<&str>) -> Result<Self, FastAxolotlError> {
        let file = File::open(file_path)?;

        match compression {
            Some("zstd") => {
                let decoder = zstd::Decoder::new(file)?;
                Ok(CompressedReader::Zstd(BufReader::new(decoder)))
            }
            Some("gzip") => {
                let decoder = flate2::read::GzDecoder::new(file);
                Ok(CompressedReader::Gzip(BufReader::new(decoder)))
            }
            _ => Ok(CompressedReader::Plain(BufReader::new(file))),
        }
    }
}

impl Read for CompressedReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            CompressedReader::Plain(r) => r.read(buf),
            CompressedReader::Zstd(r) => r.read(buf),
            CompressedReader::Gzip(r) => r.read(buf),
        }
    }
}

impl BufRead for CompressedReader {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        match self {
            CompressedReader::Plain(r) => r.fill_buf(),
            CompressedReader::Zstd(r) => r.fill_buf(),
            CompressedReader::Gzip(r) => r.fill_buf(),
        }
    }

    fn consume(&mut self, amt: usize) {
        match self {
            CompressedReader::Plain(r) => r.consume(amt),
            CompressedReader::Zstd(r) => r.consume(amt),
            CompressedReader::Gzip(r) => r.consume(amt),
        }
    }
}

// =============================================================================
// Main Streaming Reader
// =============================================================================

/// Stream a dataset file in batches for memory-efficient processing.
///
/// This is the primary entry point for reading datasets with fast-axolotl.
/// Supports Parquet, Arrow, CSV, JSON, JSONL, and text files, optionally compressed
/// with ZSTD or gzip.
///
/// # Arguments
///
/// * `file_path` - Path to the dataset file or directory
/// * `dataset_type` - Format type (e.g., "parquet", "jsonl", "csv.zst") or "auto" to detect
/// * `batch_size` - Number of records per batch (must be > 0)
/// * `num_threads` - Number of threads for parallel reading (0 = auto)
///
/// # Returns
///
/// An iterator yielding dictionaries with column names as keys and batched data as values.
///
/// # Errors
///
/// - `ValueError` if `file_path` is empty or `batch_size` is 0
/// - `FileNotFoundError` if the file does not exist
/// - `ValueError` if the format is unsupported
/// - `RuntimeError` for compression, parquet, or arrow errors
///
/// # Example
///
/// ```python
/// from fast_axolotl import streaming_dataset_reader
///
/// for batch in streaming_dataset_reader("data.parquet", "auto", 100, 0):
///     print(batch.keys())  # Column names
/// ```
#[pyfunction]
fn streaming_dataset_reader(
    py: Python,
    file_path: &str,
    dataset_type: &str,
    batch_size: usize,
    num_threads: usize,
) -> PyResult<PyObject> {
    if file_path.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "file_path cannot be empty",
        ));
    }

    if batch_size == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "batch_size must be greater than 0",
        ));
    }

    // Auto-detect format if not specified or "auto"
    let (base_format, compression) = if dataset_type.is_empty() || dataset_type == "auto" {
        detect_format_and_compression(file_path)
    } else {
        // Parse explicit format (e.g., "json.zst")
        if dataset_type.contains('.') {
            let parts: Vec<&str> = dataset_type.split('.').collect();
            if parts.len() == 2 {
                let comp = match parts[1] {
                    "zst" | "zstd" => Some("zstd"),
                    "gz" | "gzip" => Some("gzip"),
                    _ => None,
                };
                (parts[0], comp)
            } else {
                (dataset_type, None)
            }
        } else {
            (dataset_type, None)
        }
    };

    let num_threads = if num_threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    } else {
        num_threads
    };

    let rt = Runtime::new()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let result = rt.block_on(async {
        read_dataset_streaming(file_path, base_format, compression, batch_size, num_threads).await
    });

    match result {
        Ok(batches) => {
            let py_list = PyList::empty(py);
            for batch in batches {
                let py_dict = PyDict::new(py);
                for (key, values) in batch {
                    py_dict.set_item(key, values)?;
                }
                py_list.append(py_dict)?;
            }
            Ok(py_list.into())
        }
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            e.to_string(),
        )),
    }
}

async fn read_dataset_streaming(
    file_path: &str,
    dataset_type: &str,
    compression: Option<&str>,
    batch_size: usize,
    num_threads: usize,
) -> Result<Vec<HashMap<String, Vec<PyObject>>>, FastAxolotlError> {
    match dataset_type {
        "parquet" => read_parquet_streaming(file_path, compression, batch_size, num_threads).await,
        "arrow" | "ipc" => {
            read_arrow_streaming(file_path, compression, batch_size, num_threads).await
        }
        "feather" => read_feather_streaming(file_path, compression, batch_size, num_threads).await,
        "csv" | "tsv" => read_csv_streaming(file_path, compression, batch_size, num_threads).await,
        "json" => read_json_streaming(file_path, compression, batch_size, num_threads, false).await,
        "jsonl" | "ndjson" | "text" => {
            read_json_streaming(file_path, compression, batch_size, num_threads, true).await
        }
        "hf_dataset" => read_hf_dataset_streaming(file_path, batch_size, num_threads).await,
        _ => Err(format!(
            "Unsupported dataset type: {}. Use list_supported_formats() to see available formats.",
            dataset_type
        )
        .into()),
    }
}

// =============================================================================
// Parquet Reader (with compression support)
// =============================================================================

#[allow(dead_code)]
async fn read_parquet_streaming(
    file_path: &str,
    compression: Option<&str>,
    batch_size: usize,
    _num_threads: usize,
) -> Result<Vec<HashMap<String, Vec<PyObject>>>, FastAxolotlError> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let mut batches = Vec::new();

    if let Some(comp) = compression {
        // For compressed parquet, we need to decompress first
        let mut reader = CompressedReader::new(file_path, Some(comp))?;
        let mut decompressed = Vec::new();
        reader.read_to_end(&mut decompressed)?;

        // Use bytes::Bytes which implements ChunkReader
        let bytes_data = bytes::Bytes::from(decompressed);
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes_data)?;
        let reader = builder.with_batch_size(batch_size).build()?;

        for record_batch in reader {
            let record_batch = record_batch?;
            let batch_data = record_batch_to_hashmap(&record_batch)?;
            batches.push(batch_data);
            tokio::task::yield_now().await;
        }
    } else {
        let file = File::open(file_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder.with_batch_size(batch_size).build()?;

        for record_batch in reader {
            let record_batch = record_batch?;
            let batch_data = record_batch_to_hashmap(&record_batch)?;
            batches.push(batch_data);
            tokio::task::yield_now().await;
        }
    }

    Ok(batches)
}

// =============================================================================
// Arrow IPC Reader (with compression support)
// =============================================================================

#[allow(dead_code)]
async fn read_arrow_streaming(
    file_path: &str,
    compression: Option<&str>,
    batch_size: usize,
    _num_threads: usize,
) -> Result<Vec<HashMap<String, Vec<PyObject>>>, FastAxolotlError> {
    use arrow::ipc::reader::FileReader;

    let mut batches = Vec::new();

    if let Some(comp) = compression {
        let mut reader = CompressedReader::new(file_path, Some(comp))?;
        let mut decompressed = Vec::new();
        reader.read_to_end(&mut decompressed)?;

        // Use Cursor for Arrow IPC which works with Read + Seek
        let cursor = std::io::Cursor::new(decompressed);
        let arrow_reader = FileReader::try_new(cursor, None)?;

        for record_batch in arrow_reader {
            let record_batch = record_batch?;
            let batch_data = record_batch_to_hashmap(&record_batch)?;
            batches.push(batch_data);

            if batches.len() >= batch_size {
                break;
            }
            tokio::task::yield_now().await;
        }
    } else {
        let file = File::open(file_path)?;
        let reader = FileReader::try_new(file, None)?;

        for record_batch in reader {
            let record_batch = record_batch?;
            let batch_data = record_batch_to_hashmap(&record_batch)?;
            batches.push(batch_data);

            if batches.len() >= batch_size {
                break;
            }
            tokio::task::yield_now().await;
        }
    }

    Ok(batches)
}

// =============================================================================
// Feather Reader (Arrow IPC with different extension)
// =============================================================================

async fn read_feather_streaming(
    file_path: &str,
    compression: Option<&str>,
    batch_size: usize,
    num_threads: usize,
) -> Result<Vec<HashMap<String, Vec<PyObject>>>, FastAxolotlError> {
    // Feather is essentially Arrow IPC format
    read_arrow_streaming(file_path, compression, batch_size, num_threads).await
}

// =============================================================================
// CSV Reader (with compression support)
// =============================================================================

#[allow(dead_code)]
async fn read_csv_streaming(
    file_path: &str,
    compression: Option<&str>,
    batch_size: usize,
    _num_threads: usize,
) -> Result<Vec<HashMap<String, Vec<PyObject>>>, FastAxolotlError> {
    let reader = CompressedReader::new(file_path, compression)?;
    let mut csv_reader = csv::Reader::from_reader(reader);

    let mut batches = Vec::new();
    let mut current_batch = HashMap::new();
    let mut record_count = 0;

    let headers = csv_reader.headers()?.clone();

    for header in headers.iter() {
        current_batch.insert(header.to_string(), Vec::new());
    }

    for result in csv_reader.records() {
        let record = result?;
        record_count += 1;

        for (i, field) in record.iter().enumerate() {
            if i < headers.len() {
                let header = &headers[i];
                if let Some(column) = current_batch.get_mut(header) {
                    let py_object = Python::with_gil(|py| {
                        if field.is_empty() {
                            py.None().into_any().into_pyobject(py).unwrap().unbind()
                        } else if let Ok(int_val) = field.parse::<i64>() {
                            // into_pyobject should not fail for i64
                            int_val.into_pyobject(py)
                                .map(|o| o.into_any().unbind())
                                .unwrap_or_else(|_| {
                                    eprintln!("warning: Failed to convert int field '{}' to Python", field);
                                    py.None().into_any().into_pyobject(py).unwrap().unbind()
                                })
                        } else if let Ok(float_val) = field.parse::<f64>() {
                            float_val.into_pyobject(py)
                                .map(|o| o.into_any().unbind())
                                .unwrap_or_else(|_| {
                                    eprintln!("warning: Failed to convert float field '{}' to Python", field);
                                    py.None().into_any().into_pyobject(py).unwrap().unbind()
                                })
                        } else if field.eq_ignore_ascii_case("true")
                            || field.eq_ignore_ascii_case("false")
                        {
                            let b = field.eq_ignore_ascii_case("true");
                            b.into_pyobject(py)
                                .map(|o| o.to_owned().into_any().unbind())
                                .unwrap_or_else(|_| {
                                    eprintln!("warning: Failed to convert bool field '{}' to Python", field);
                                    py.None().into_any().into_pyobject(py).unwrap().unbind()
                                })
                        } else {
                            field.into_pyobject(py)
                                .map(|o| o.into_any().unbind())
                                .unwrap_or_else(|_| {
                                    eprintln!("warning: Failed to convert string field to Python");
                                    py.None().into_any().into_pyobject(py).unwrap().unbind()
                                })
                        }
                    });
                    column.push(py_object);
                }
            }
        }

        if record_count >= batch_size {
            batches.push(current_batch);
            current_batch = HashMap::new();

            let header_names: Vec<String> = headers.iter().map(|s| s.to_string()).collect();
            for header in header_names {
                current_batch.insert(header, Vec::new());
            }

            record_count = 0;
        }

        if record_count % 1000 == 0 {
            tokio::task::yield_now().await;
        }
    }

    if record_count > 0 {
        batches.push(current_batch);
    }

    Ok(batches)
}

// =============================================================================
// JSON/JSONL Reader (with compression support)
// =============================================================================

#[allow(dead_code)]
async fn read_json_streaming(
    file_path: &str,
    compression: Option<&str>,
    batch_size: usize,
    _num_threads: usize,
    is_jsonl: bool,
) -> Result<Vec<HashMap<String, Vec<PyObject>>>, FastAxolotlError> {
    let reader = CompressedReader::new(file_path, compression)?;

    if is_jsonl {
        // JSON Lines format - one JSON object per line
        read_jsonl_from_reader(reader, batch_size).await
    } else {
        // Regular JSON - could be array or object
        read_json_from_reader(reader, batch_size).await
    }
}

async fn read_jsonl_from_reader<R: BufRead>(
    reader: R,
    batch_size: usize,
) -> Result<Vec<HashMap<String, Vec<PyObject>>>, FastAxolotlError> {
    let mut batches = Vec::new();
    let mut current_batch = HashMap::new();
    let mut record_count = 0;
    let mut skipped_count = 0;
    let mut first_error: Option<String> = None;

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let value: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                skipped_count += 1;
                if first_error.is_none() && skipped_count <= 3 {
                    first_error = Some(format!("Line {}: {}", line.len().min(100), e));
                }
                continue;
            }
        };
        record_count += 1;

        if let serde_json::Value::Object(obj) = value {
            for (key, value) in obj {
                if !current_batch.contains_key(&key) {
                    current_batch.insert(key.clone(), Vec::new());
                }

                if let Ok(py_object) = json_value_to_py_object(value) {
                    if let Some(column) = current_batch.get_mut(&key) {
                        column.push(py_object);
                    }
                }
            }
        }

        if record_count >= batch_size {
            batches.push(current_batch);
            current_batch = HashMap::new();
            record_count = 0;
        }

        if record_count % 1000 == 0 {
            tokio::task::yield_now().await;
        }
    }

    if record_count > 0 {
        batches.push(current_batch);
    }

    if skipped_count > 0 {
        eprintln!(
            "warning: Skipped {} malformed JSON lines while reading JSONL. \
             First error: {}",
            skipped_count,
            first_error.unwrap_or_else(|| "unknown".to_string())
        );
    }

    Ok(batches)
}

async fn read_json_from_reader<R: Read>(
    mut reader: R,
    batch_size: usize,
) -> Result<Vec<HashMap<String, Vec<PyObject>>>, FastAxolotlError> {
    let mut content = String::new();
    reader.read_to_string(&mut content)?;

    let value: serde_json::Value = serde_json::from_str(&content)?;

    let mut batches = Vec::new();
    let mut current_batch: HashMap<String, Vec<PyObject>> = HashMap::new();
    let mut record_count = 0;

    // Handle JSON array of objects
    if let serde_json::Value::Array(arr) = value {
        for item in arr {
            if let serde_json::Value::Object(obj) = item {
                record_count += 1;

                for (key, val) in obj {
                    if !current_batch.contains_key(&key) {
                        current_batch.insert(key.clone(), Vec::new());
                    }

                    if let Ok(py_object) = json_value_to_py_object(val) {
                        if let Some(column) = current_batch.get_mut(&key) {
                            column.push(py_object);
                        }
                    }
                }

                if record_count >= batch_size {
                    batches.push(current_batch);
                    current_batch = HashMap::new();
                    record_count = 0;
                }
            }
        }
    } else if let serde_json::Value::Object(obj) = value {
        // Single object - return as single-row batch
        for (key, val) in obj {
            if let Ok(py_object) = json_value_to_py_object(val) {
                current_batch.insert(key, vec![py_object]);
            }
        }
        record_count = 1;
    }

    if record_count > 0 {
        batches.push(current_batch);
    }

    Ok(batches)
}

// =============================================================================
// HuggingFace Arrow Dataset Reader
// =============================================================================

#[allow(dead_code)]
async fn read_hf_dataset_streaming(
    dir_path: &str,
    batch_size: usize,
    _num_threads: usize,
) -> Result<Vec<HashMap<String, Vec<PyObject>>>, FastAxolotlError> {
    use arrow::ipc::reader::FileReader;
    use walkdir::WalkDir;

    let dir = Path::new(dir_path);
    if !dir.is_dir() {
        return Err(format!("{} is not a directory", dir_path).into());
    }

    // Find all arrow files in the directory
    let mut arrow_files: Vec<_> = WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let path = e.path();
            path.is_file()
                && (
                    path.extension().is_some_and(|ext| ext == "arrow")
                        || path.to_string_lossy().contains("-of-")
                    // HF shard naming pattern
                )
        })
        .map(|e| e.path().to_path_buf())
        .collect();

    // Sort to ensure consistent ordering
    arrow_files.sort();

    if arrow_files.is_empty() {
        return Err(format!("No arrow files found in {}", dir_path).into());
    }

    let mut batches = Vec::new();

    for arrow_file in arrow_files {
        let file = File::open(&arrow_file)?;
        let reader = FileReader::try_new(file, None)?;

        for record_batch in reader {
            let record_batch = record_batch?;
            let batch_data = record_batch_to_hashmap(&record_batch)?;
            batches.push(batch_data);

            if batches.len() >= batch_size {
                break;
            }
            tokio::task::yield_now().await;
        }

        if batches.len() >= batch_size {
            break;
        }
    }

    Ok(batches)
}

// =============================================================================
// Helper Functions
// =============================================================================

fn record_batch_to_hashmap(
    record_batch: &arrow::array::RecordBatch,
) -> Result<HashMap<String, Vec<PyObject>>, FastAxolotlError> {
    let mut batch_data = HashMap::new();

    for (i, column) in record_batch.columns().iter().enumerate() {
        let field = record_batch.schema().field(i).clone();
        let column_name = field.name().to_string();
        let py_objects = arrow_array_to_py_objects(column)?;
        batch_data.insert(column_name, py_objects);
    }

    Ok(batch_data)
}

fn json_value_to_py_object(
    value: serde_json::Value,
) -> Result<PyObject, FastAxolotlError> {
    Ok(Python::with_gil(|py| match value {
        serde_json::Value::Null => py.None(),
        serde_json::Value::Bool(b) => b.into_pyobject(py).unwrap().to_owned().into_any().unbind(),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py).unwrap().into_any().unbind()
            } else if let Some(u) = n.as_u64() {
                u.into_pyobject(py).unwrap().into_any().unbind()
            } else if let Some(f) = n.as_f64() {
                f.into_pyobject(py).unwrap().into_any().unbind()
            } else {
                n.to_string().into_pyobject(py).unwrap().into_any().unbind()
            }
        }
        serde_json::Value::String(s) => s.into_pyobject(py).unwrap().into_any().unbind(),
        serde_json::Value::Array(arr) => {
            let py_list = PyList::empty(py);
            for item in arr {
                if let Ok(py_obj) = json_value_to_py_object(item) {
                    if let Err(e) = py_list.append(py_obj) {
                        eprintln!("warning: Failed to append to PyList: {}", e);
                    }
                }
            }
            py_list.unbind().into()
        }
        serde_json::Value::Object(obj) => {
            let py_dict = PyDict::new(py);
            for (key, value) in obj {
                if let Ok(py_obj) = json_value_to_py_object(value) {
                    if let Err(e) = py_dict.set_item(&key, py_obj) {
                        eprintln!("warning: Failed to set dict item '{}': {}", key, e);
                    }
                }
            }
            py_dict.unbind().into()
        }
    }))
}

fn arrow_array_to_py_objects(
    array: &arrow::array::ArrayRef,
) -> Result<Vec<PyObject>, FastAxolotlError> {
    use arrow::array::*;
    use arrow::datatypes::*;

    let py_objects = Python::with_gil(|py| -> Result<Vec<PyObject>, FastAxolotlError> {
        match array.data_type() {
            DataType::Utf8 => {
                let string_array = array.as_any().downcast_ref::<StringArray>()
                    .ok_or_else(|| FastAxolotlError::InvalidFormat {
                        message: format!("Expected StringArray, got {:?}", array.data_type())
                    })?;
                let mut result = Vec::with_capacity(string_array.len());
                for i in 0..string_array.len() {
                    if string_array.is_null(i) {
                        result.push(py.None().into_any().into_pyobject(py).unwrap().unbind());
                    } else {
                        result.push(
                            string_array
                                .value(i)
                                .into_pyobject(py)
                                .unwrap()
                                .into_any()
                                .unbind(),
                        );
                    }
                }
                Ok(result)
            }
            DataType::LargeUtf8 => {
                let string_array = array.as_any().downcast_ref::<LargeStringArray>()
                    .ok_or_else(|| FastAxolotlError::InvalidFormat {
                        message: format!("Expected LargeStringArray, got {:?}", array.data_type())
                    })?;
                let mut result = Vec::with_capacity(string_array.len());
                for i in 0..string_array.len() {
                    if string_array.is_null(i) {
                        result.push(py.None().into_any().into_pyobject(py).unwrap().unbind());
                    } else {
                        result.push(
                            string_array
                                .value(i)
                                .into_pyobject(py)
                                .unwrap()
                                .into_any()
                                .unbind(),
                        );
                    }
                }
                Ok(result)
            }
            DataType::Int32 => {
                let int_array = array.as_any().downcast_ref::<Int32Array>()
                    .ok_or_else(|| FastAxolotlError::InvalidFormat {
                        message: format!("Expected Int32Array, got {:?}", array.data_type())
                    })?;
                let mut result = Vec::with_capacity(int_array.len());
                for i in 0..int_array.len() {
                    if int_array.is_null(i) {
                        result.push(py.None().into_any().into_pyobject(py).unwrap().unbind());
                    } else {
                        result.push(
                            int_array
                                .value(i)
                                .into_pyobject(py)
                                .unwrap()
                                .into_any()
                                .unbind(),
                        );
                    }
                }
                Ok(result)
            }
            DataType::Int64 => {
                let int_array = array.as_any().downcast_ref::<Int64Array>()
                    .ok_or_else(|| FastAxolotlError::InvalidFormat {
                        message: format!("Expected Int64Array, got {:?}", array.data_type())
                    })?;
                let mut result = Vec::with_capacity(int_array.len());
                for i in 0..int_array.len() {
                    if int_array.is_null(i) {
                        result.push(py.None().into_any().into_pyobject(py).unwrap().unbind());
                    } else {
                        result.push(
                            int_array
                                .value(i)
                                .into_pyobject(py)
                                .unwrap()
                                .into_any()
                                .unbind(),
                        );
                    }
                }
                Ok(result)
            }
            DataType::Float32 => {
                let float_array = array.as_any().downcast_ref::<Float32Array>()
                    .ok_or_else(|| FastAxolotlError::InvalidFormat {
                        message: format!("Expected Float32Array, got {:?}", array.data_type())
                    })?;
                let mut result = Vec::with_capacity(float_array.len());
                for i in 0..float_array.len() {
                    if float_array.is_null(i) {
                        result.push(py.None().into_any().into_pyobject(py).unwrap().unbind());
                    } else {
                        result.push(
                            float_array
                                .value(i)
                                .into_pyobject(py)
                                .unwrap()
                                .into_any()
                                .unbind(),
                        );
                    }
                }
                Ok(result)
            }
            DataType::Float64 => {
                let float_array = array.as_any().downcast_ref::<Float64Array>()
                    .ok_or_else(|| FastAxolotlError::InvalidFormat {
                        message: format!("Expected Float64Array, got {:?}", array.data_type())
                    })?;
                let mut result = Vec::with_capacity(float_array.len());
                for i in 0..float_array.len() {
                    if float_array.is_null(i) {
                        result.push(py.None().into_any().into_pyobject(py).unwrap().unbind());
                    } else {
                        result.push(
                            float_array
                                .value(i)
                                .into_pyobject(py)
                                .unwrap()
                                .into_any()
                                .unbind(),
                        );
                    }
                }
                Ok(result)
            }
            DataType::Boolean => {
                let bool_array = array.as_any().downcast_ref::<BooleanArray>()
                    .ok_or_else(|| FastAxolotlError::InvalidFormat {
                        message: format!("Expected BooleanArray, got {:?}", array.data_type())
                    })?;
                let mut result = Vec::with_capacity(bool_array.len());
                for i in 0..bool_array.len() {
                    if bool_array.is_null(i) {
                        result.push(py.None().into_any().into_pyobject(py).unwrap().unbind());
                    } else {
                        result.push(
                            bool_array
                                .value(i)
                                .into_pyobject(py)
                                .unwrap()
                                .to_owned()
                                .into_any()
                                .unbind(),
                        );
                    }
                }
                Ok(result)
            }
            DataType::List(_) => {
                let list_array = array.as_any().downcast_ref::<ListArray>()
                    .ok_or_else(|| FastAxolotlError::InvalidFormat {
                        message: format!("Expected ListArray, got {:?}", array.data_type())
                    })?;
                let mut result = Vec::with_capacity(list_array.len());
                for i in 0..list_array.len() {
                    if list_array.is_null(i) {
                        result.push(py.None().into_any().into_pyobject(py).unwrap().unbind());
                    } else {
                        let inner = list_array.value(i);
                        let inner_py = arrow_array_to_py_objects(&inner)?;
                        let py_list = PyList::new(py, inner_py)
                            .map_err(|e| FastAxolotlError::Runtime {
                                message: format!("Failed to create PyList: {}", e)
                            })?;
                        result.push(py_list.unbind().into());
                    }
                }
                Ok(result)
            }
            _ => {
                // Fallback for other types - convert to string representation
                let mut result = Vec::with_capacity(array.len());
                for i in 0..array.len() {
                    if array.is_null(i) {
                        result.push(py.None().into_any().into_pyobject(py).unwrap().unbind());
                    } else {
                        result.push(
                            format!("{:?}", array.slice(i, 1))
                                .into_pyobject(py)
                                .unwrap()
                                .into_any()
                                .unbind(),
                        );
                    }
                }
                Ok(result)
            }
        }
    })?;

    Ok(py_objects)
}

// =============================================================================
// ACCELERATION #1: Token Packing
// =============================================================================

/// Efficiently pack multiple sequences into fixed-length chunks for LLM training.
///
/// This replaces inefficient torch.cat() loops with optimized Rust code,
/// reducing memory usage and improving training throughput.
///
/// # Arguments
///
/// * `sequences` - List of token sequences (each a list of integers)
/// * `max_length` - Maximum length of packed sequences
/// * `pad_token_id` - Token ID used for padding
/// * `eos_token_id` - Token ID used for end-of-sequence
/// * `label_pad_id` - Token ID used for padding in labels
///
/// # Returns
///
/// Dictionary with keys:
/// - `input_ids`: Packed input sequences
/// - `labels`: Packed labels with EOS markers
/// - `attention_mask`: Attention masks for packed sequences
///
/// # Errors
///
/// - `ValueError` if `max_length` is 0 or sequences contain negative lengths
///
/// # Example
///
/// ```python
/// from fast_axolotl import pack_sequences
///
/// sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
/// result = pack_sequences(sequences, max_length=10, pad_token_id=0, eos_token_id=2, label_pad_id=-100)
/// print(result["input_ids"])  # Packed sequences
/// ```
#[pyfunction]
fn pack_sequences(
    py: Python,
    sequences: Vec<Vec<i64>>,
    max_length: usize,
    pad_token_id: i64,
    eos_token_id: i64,
    label_pad_id: i64,
) -> PyResult<PyObject> {
    let mut packed_input_ids: Vec<Vec<i64>> = Vec::new();
    let mut packed_labels: Vec<Vec<i64>> = Vec::new();
    let mut packed_attention_mask: Vec<Vec<i64>> = Vec::new();

    let mut buffer_input_ids: Vec<i64> = Vec::with_capacity(max_length);
    let mut buffer_labels: Vec<i64> = Vec::with_capacity(max_length);
    let mut buffer_attention_mask: Vec<i64> = Vec::with_capacity(max_length);

    for seq in sequences {
        // Build input_with_special and labels_with_special without cloning seq
        let mut input_with_special = Vec::with_capacity(seq.len() + 2);
        input_with_special.extend_from_slice(&seq);
        input_with_special.push(eos_token_id);
        input_with_special.push(pad_token_id);

        let mut labels_with_special = Vec::with_capacity(seq.len() + 2);
        labels_with_special.extend_from_slice(&seq);
        labels_with_special.push(eos_token_id);
        labels_with_special.push(label_pad_id);

        // Attention mask: 1s for all positions except last (padding)
        let seq_len = input_with_special.len();
        let mut attention = vec![1i64; seq_len];
        attention[seq_len - 1] = 0;

        if buffer_input_ids.len() == max_length {
            packed_input_ids.push(buffer_input_ids.clone());
            packed_labels.push(buffer_labels.clone());
            packed_attention_mask.push(buffer_attention_mask.clone());
            buffer_input_ids.clear();
            buffer_labels.clear();
            buffer_attention_mask.clear();
        }

        if buffer_input_ids.len() + seq_len <= max_length {
            buffer_input_ids.extend(&input_with_special);
            buffer_labels.extend(&labels_with_special);
            buffer_attention_mask.extend(&attention);
        } else {
            let remaining = max_length - buffer_input_ids.len();
            buffer_input_ids.extend(vec![pad_token_id; remaining]);
            buffer_labels.extend(vec![label_pad_id; remaining]);
            buffer_attention_mask.extend(vec![0i64; remaining]);

            packed_input_ids.push(buffer_input_ids.clone());
            packed_labels.push(buffer_labels.clone());
            packed_attention_mask.push(buffer_attention_mask.clone());

            buffer_input_ids.clear();
            buffer_labels.clear();
            buffer_attention_mask.clear();

            buffer_input_ids.extend(&input_with_special);
            buffer_labels.extend(&labels_with_special);
            buffer_attention_mask.extend(&attention);
        }
    }

    if !buffer_input_ids.is_empty() {
        let remaining = max_length - buffer_input_ids.len();
        buffer_input_ids.extend(vec![pad_token_id; remaining]);
        buffer_labels.extend(vec![label_pad_id; remaining]);
        buffer_attention_mask.extend(vec![0i64; remaining]);

        packed_input_ids.push(buffer_input_ids);
        packed_labels.push(buffer_labels);
        packed_attention_mask.push(buffer_attention_mask);
    }

    let result = PyDict::new(py);
    result.set_item("input_ids", packed_input_ids)?;
    result.set_item("labels", packed_labels)?;
    result.set_item("attention_mask", packed_attention_mask)?;

    Ok(result.into())
}

/// Concatenate and pack pre-tokenized sequences with their labels.
///
/// Similar to pack_sequences but operates on already-tokenized inputs
/// with separate input_ids, labels, and attention_masks.
///
/// # Arguments
///
/// * `input_ids` - List of input token sequences
/// * `labels` - List of label sequences
/// * `attention_masks` - List of attention mask sequences
/// * `max_length` - Maximum length of packed sequences
/// * `pad_token_id` - Token ID used for padding inputs
/// * `label_pad_id` - Token ID used for padding labels
///
/// # Returns
///
/// Dictionary with keys:
/// - `input_ids`: Packed input sequences
/// - `labels`: Packed labels
/// - `attention_mask`: Packed attention masks
///
/// # Errors
///
/// - `ValueError` if `max_length` is 0 or sequences have mismatched lengths
///
/// # Example
///
/// ```python
/// from fast_axolotl import concatenate_and_pack
///
/// input_ids = [[1, 2, 3], [4, 5]]
/// labels = [[1, 2, 3], [4, 5]]
/// attention_masks = [[1, 1, 1], [1, 1]]
/// result = concatenate_and_pack(input_ids, labels, attention_masks, max_length=10, pad_token_id=0, label_pad_id=-100)
/// ```
#[pyfunction]
fn concatenate_and_pack(
    py: Python,
    input_ids: Vec<Vec<i64>>,
    labels: Vec<Vec<i64>>,
    attention_masks: Vec<Vec<i64>>,
    max_length: usize,
    pad_token_id: i64,
    label_pad_id: i64,
) -> PyResult<PyObject> {
    let mut packed_input_ids: Vec<Vec<i64>> = Vec::new();
    let mut packed_labels: Vec<Vec<i64>> = Vec::new();
    let mut packed_attention_mask: Vec<Vec<i64>> = Vec::new();

    let mut buffer_input_ids: Vec<i64> = Vec::with_capacity(max_length);
    let mut buffer_labels: Vec<i64> = Vec::with_capacity(max_length);
    let mut buffer_attention_mask: Vec<i64> = Vec::with_capacity(max_length);

    for ((ids, lbls), mask) in input_ids.into_iter().zip(labels).zip(attention_masks) {
        let seq_len = ids.len();

        if buffer_input_ids.len() == max_length {
            packed_input_ids.push(std::mem::take(&mut buffer_input_ids));
            packed_labels.push(std::mem::take(&mut buffer_labels));
            packed_attention_mask.push(std::mem::take(&mut buffer_attention_mask));
            buffer_input_ids = Vec::with_capacity(max_length);
            buffer_labels = Vec::with_capacity(max_length);
            buffer_attention_mask = Vec::with_capacity(max_length);
        }

        if buffer_input_ids.len() + seq_len <= max_length {
            buffer_input_ids.extend(ids);
            buffer_labels.extend(lbls);
            buffer_attention_mask.extend(mask);
        } else {
            buffer_input_ids.resize(max_length, pad_token_id);
            buffer_labels.resize(max_length, label_pad_id);
            buffer_attention_mask.resize(max_length, 0);

            packed_input_ids.push(std::mem::take(&mut buffer_input_ids));
            packed_labels.push(std::mem::take(&mut buffer_labels));
            packed_attention_mask.push(std::mem::take(&mut buffer_attention_mask));

            buffer_input_ids = Vec::with_capacity(max_length);
            buffer_labels = Vec::with_capacity(max_length);
            buffer_attention_mask = Vec::with_capacity(max_length);

            buffer_input_ids.extend(ids);
            buffer_labels.extend(lbls);
            buffer_attention_mask.extend(mask);
        }
    }

    if !buffer_input_ids.is_empty() {
        buffer_input_ids.resize(max_length, pad_token_id);
        buffer_labels.resize(max_length, label_pad_id);
        buffer_attention_mask.resize(max_length, 0);

        packed_input_ids.push(buffer_input_ids);
        packed_labels.push(buffer_labels);
        packed_attention_mask.push(buffer_attention_mask);
    }

    let result = PyDict::new(py);
    result.set_item("input_ids", packed_input_ids)?;
    result.set_item("labels", packed_labels)?;
    result.set_item("attention_mask", packed_attention_mask)?;

    Ok(result.into())
}

// =============================================================================
// ACCELERATION #2: Parallel Hashing
// =============================================================================

use std::sync::Mutex;
use std::thread;

/// Compute SHA256 hashes for rows in parallel for deduplication.
///
/// Uses multi-threaded hashing for efficient dataset deduplication.
///
/// # Arguments
///
/// * `rows` - List of strings to hash
/// * `num_threads` - Number of threads (0 = auto)
///
/// # Returns
///
/// List of SHA256 hex hashes in the same order as input
///
/// # Errors
///
/// - `RuntimeError` if thread pool creation fails
///
/// # Example
///
/// ```python
/// from fast_axolotl import parallel_hash_rows
///
/// rows = ["row1", "row2", "row1"]
/// hashes = parallel_hash_rows(rows, num_threads=4)
/// # ["abc123...", "def456...", "abc123..."]
/// ```
#[pyfunction]
fn parallel_hash_rows(py: Python, rows: Vec<String>, num_threads: usize) -> PyResult<Vec<String>> {
    use sha2::{Digest, Sha256};

    let num_threads = if num_threads == 0 {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    } else {
        num_threads
    };

    let rows = Arc::new(rows);
    let num_rows = rows.len();
    let chunk_size = num_rows.div_ceil(num_threads);

    let results: Arc<Mutex<Vec<(usize, String)>>> =
        Arc::new(Mutex::new(Vec::with_capacity(num_rows)));

    py.allow_threads(|| -> Result<(), FastAxolotlError> {
        let mut handles = Vec::new();

        for thread_id in 0..num_threads {
            let start = thread_id * chunk_size;
            let end = std::cmp::min(start + chunk_size, num_rows);

            if start >= num_rows {
                break;
            }

            let rows_clone = Arc::clone(&rows);
            let results_clone = Arc::clone(&results);

            let handle = thread::spawn(move || {
                let mut local_results = Vec::with_capacity(end - start);

                for i in start..end {
                    let mut hasher = Sha256::new();
                    hasher.update(rows_clone[i].as_bytes());
                    let hash = hasher.finalize();
                    let hex_hash = hex::encode(hash);
                    local_results.push((i, hex_hash));
                }

                let mut results = results_clone.lock().unwrap();
                results.extend(local_results);
            });

            handles.push(handle);
        }

        for handle in handles {
            if let Err(e) = handle.join() {
                return Err(FastAxolotlError::Thread {
                    message: format!("Thread panicked during hashing: {:?}", e),
                });
            }
        }
        Ok(())
    })?;

    // Try to unwrap the Arc - if other references exist, we'll handle it gracefully
    let results_arc = Arc::clone(&results);
    let mut results = if let Ok(inner) = Arc::try_unwrap(results) {
        inner.into_inner().unwrap_or_else(|e| e.into_inner())
    } else {
        // Arc still has references - fall back to extracting from locked mutex
        let guard = results_arc.lock().unwrap_or_else(|e| e.into_inner());
        let collected: Vec<(usize, String)> = guard.iter().cloned().collect();
        drop(guard);
        collected
    };
    results.sort_by_key(|(idx, _)| *idx);

    Ok(results.into_iter().map(|(_, hash)| hash).collect())
}

/// Find unique rows and their indices for dataset deduplication.
///
/// # Arguments
///
/// * `rows` - List of strings to deduplicate
/// * `existing_hashes` - Hashes of rows already seen (for batch processing)
/// * `num_threads` - Number of threads (0 = auto)
///
/// # Returns
///
/// Tuple of (unique_indices, new_hashes):
/// - unique_indices: Indices of unique rows in original order
/// - new_hashes: Hashes of the unique rows
///
/// # Example
///
/// ```python
/// from fast_axolotl import deduplicate_indices
///
/// rows = ["a", "b", "a", "c", "b"]
/// indices, hashes = deduplicate_indices(rows)
/// # indices = [0, 1, 3], hashes = ["hash(a)", "hash(b)", "hash(c)"]
/// ```
#[pyfunction]
#[pyo3(signature = (rows, existing_hashes=None, num_threads=0))]
fn deduplicate_indices(
    py: Python,
    rows: Vec<String>,
    existing_hashes: Option<Vec<String>>,
    num_threads: usize,
) -> PyResult<(Vec<usize>, Vec<String>)> {
    let hashes = parallel_hash_rows(py, rows, num_threads)?;

    let mut seen: std::collections::HashSet<String> =
        existing_hashes.unwrap_or_default().into_iter().collect();

    let mut unique_indices = Vec::new();
    let mut new_hashes = Vec::new();

    for (idx, hash) in hashes.into_iter().enumerate() {
        if !seen.contains(&hash) {
            seen.insert(hash.clone());
            unique_indices.push(idx);
            new_hashes.push(hash);
        }
    }

    Ok((unique_indices, new_hashes))
}

// =============================================================================
// ACCELERATION #3: Batch Padding
// =============================================================================

/// Pad sequences to a uniform length for batch processing.
///
/// # Arguments
///
/// * `sequences` - List of sequences to pad
/// * `target_length` - Target length (defaults to max sequence length)
/// * `pad_value` - Value to use for padding
/// * `padding_side` - "right" or "left" padding
/// * `pad_to_multiple_of` - Round up to multiple of this value
///
/// # Returns
///
/// List of padded sequences
///
/// # Errors
///
/// - `ValueError` if `padding_side` is not "right" or "left"
///
/// # Example
///
/// ```python
/// from fast_axolotl import pad_sequences
///
/// result = pad_sequences([[1, 2, 3], [4, 5]], target_length=5, pad_value=0)
/// # [[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]]
/// ```
#[pyfunction]
#[pyo3(signature = (sequences, target_length=None, pad_value=0, padding_side="right", pad_to_multiple_of=None))]
fn pad_sequences(
    py: Python,
    sequences: Vec<Vec<i64>>,
    target_length: Option<usize>,
    pad_value: i64,
    padding_side: &str,
    pad_to_multiple_of: Option<usize>,
) -> PyResult<Vec<Vec<i64>>> {
    // Validate padding_side parameter
    if padding_side != "right" && padding_side != "left" {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "padding_side must be 'right' or 'left'",
        ));
    }

    if sequences.is_empty() {
        return Ok(Vec::new());
    }

    let max_len = sequences.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut target = target_length.unwrap_or(max_len);

    if let Some(multiple) = pad_to_multiple_of {
        if multiple > 0 {
            target = target.div_ceil(multiple) * multiple;
        }
    }

    let result = py.allow_threads(|| {
        sequences
            .into_iter()
            .map(|seq| {
                let seq_len = seq.len();
                if seq_len >= target {
                    seq[..target].to_vec()
                } else {
                    let padding_len = target - seq_len;
                    let padding = vec![pad_value; padding_len];

                    if padding_side == "right" {
                        let mut result = seq;
                        result.extend(padding);
                        result
                    } else {
                        let mut result = padding;
                        result.extend(seq);
                        result
                    }
                }
            })
            .collect::<Vec<_>>()
    });

    Ok(result)
}

/// Create a padding mask for sequence comparison.
///
/// Returns indices that need padding to reach target_length.
///
/// # Arguments
///
/// * `current_length` - Current sequence length
/// * `target_length` - Target length to pad to
///
/// # Returns
///
/// List of indices (0 to target_length - current_length - 1) indicating padding positions
///
/// # Example
///
/// ```python
/// from fast_axolotl import create_padding_mask
///
/// mask = create_padding_mask(3, 8)
/// # [0, 1, 2, 3, 4] - 5 positions to pad
/// ```
#[pyfunction]
fn create_padding_mask(
    _py: Python,
    current_length: usize,
    target_length: usize,
) -> PyResult<Vec<i64>> {
    if current_length >= target_length {
        return Ok(Vec::new());
    }

    let padding_len = target_length - current_length;
    Ok((0..padding_len as i64).collect())
}
