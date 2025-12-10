// Fast-Axolotl: High-performance Rust extensions for Axolotl
// This module provides memory-efficient streaming for large datasets

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tokio::runtime::Runtime;

/// Python module for fast-axolotl Rust extensions
#[pymodule]
fn _rust_ext(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(streaming_dataset_reader, m)?)?;
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    Ok(())
}

/// Get the fast-axolotl version
#[pyfunction]
fn get_version() -> &'static str {
    "0.1.0"
}

/// Stream dataset files with memory efficiency
#[pyfunction]
fn streaming_dataset_reader(
    py: Python,
    file_path: &str,
    dataset_type: &str,
    batch_size: usize,
    num_threads: usize,
) -> PyResult<PyObject> {
    if file_path.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("file_path cannot be empty"));
    }

    if dataset_type.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("dataset_type cannot be empty"));
    }

    if batch_size == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("batch_size must be greater than 0"));
    }

    if num_threads == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("num_threads must be greater than 0"));
    }

    let rt = Runtime::new().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let result = rt.block_on(async {
        read_dataset_streaming(file_path, dataset_type, batch_size, num_threads).await
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
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
}

async fn read_dataset_streaming(
    file_path: &str,
    dataset_type: &str,
    batch_size: usize,
    num_threads: usize,
) -> Result<Vec<std::collections::HashMap<String, Vec<PyObject>>>, Box<dyn std::error::Error>> {
    match dataset_type {
        "parquet" => read_parquet_streaming(file_path, batch_size, num_threads).await,
        "arrow" => read_arrow_streaming(file_path, batch_size, num_threads).await,
        "csv" => read_csv_streaming(file_path, batch_size, num_threads).await,
        "json" | "text" => read_json_streaming(file_path, batch_size, num_threads).await,
        _ => Err(format!("Unsupported dataset type: {}. Supported types are: parquet, arrow, csv, json, text", dataset_type).into()),
    }
}

async fn read_parquet_streaming(
    file_path: &str,
    batch_size: usize,
    _num_threads: usize,
) -> Result<Vec<std::collections::HashMap<String, Vec<PyObject>>>, Box<dyn std::error::Error>> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::fs::File;

    let file = File::open(file_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.with_batch_size(batch_size).build()?;

    let mut batches = Vec::new();

    for record_batch in reader {
        let record_batch = record_batch?;
        let mut batch_data = std::collections::HashMap::new();

        for (i, column) in record_batch.columns().iter().enumerate() {
            let field = record_batch.schema().field(i).clone();
            let column_name = field.name().to_string();
            let py_objects = arrow_array_to_py_objects(column)?;
            batch_data.insert(column_name, py_objects);
        }

        batches.push(batch_data);
        tokio::task::yield_now().await;
    }

    Ok(batches)
}

async fn read_arrow_streaming(
    file_path: &str,
    batch_size: usize,
    _num_threads: usize,
) -> Result<Vec<std::collections::HashMap<String, Vec<PyObject>>>, Box<dyn std::error::Error>> {
    use arrow::ipc::reader::FileReader;
    use std::fs::File;

    let file = File::open(file_path)?;
    let mut reader = FileReader::try_new(file, None)?;

    let mut batches = Vec::new();

    while let Some(record_batch) = reader.next() {
        let record_batch = record_batch?;
        let mut batch_data = std::collections::HashMap::new();

        for (i, column) in record_batch.columns().iter().enumerate() {
            let field = record_batch.schema().field(i).clone();
            let column_name = field.name().to_string();
            let py_objects = arrow_array_to_py_objects(column)?;
            batch_data.insert(column_name, py_objects);
        }

        batches.push(batch_data);

        if batches.len() >= batch_size {
            break;
        }

        tokio::task::yield_now().await;
    }

    Ok(batches)
}

async fn read_csv_streaming(
    file_path: &str,
    batch_size: usize,
    _num_threads: usize,
) -> Result<Vec<std::collections::HashMap<String, Vec<PyObject>>>, Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut csv_reader = csv::Reader::from_reader(reader);

    let mut batches = Vec::new();
    let mut current_batch = std::collections::HashMap::new();
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
                            py.None()
                        } else if let Ok(int_val) = field.parse::<i64>() {
                            int_val.to_object(py).into()
                        } else if let Ok(float_val) = field.parse::<f64>() {
                            float_val.to_object(py).into()
                        } else if field.eq_ignore_ascii_case("true") || field.eq_ignore_ascii_case("false") {
                            field.parse::<bool>().unwrap_or(false).to_object(py).into()
                        } else {
                            field.to_object(py).into()
                        }
                    });
                    column.push(py_object);
                }
            }
        }

        if record_count >= batch_size {
            batches.push(current_batch);
            current_batch = std::collections::HashMap::new();

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

async fn read_json_streaming(
    file_path: &str,
    batch_size: usize,
    _num_threads: usize,
) -> Result<Vec<std::collections::HashMap<String, Vec<PyObject>>>, Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::{BufReader, BufRead};

    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let mut batches = Vec::new();
    let mut current_batch = std::collections::HashMap::new();
    let mut record_count = 0;

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let value: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Error parsing JSON line: {}", e);
                continue;
            }
        };
        record_count += 1;

        if let serde_json::Value::Object(obj) = value {
            for (key, value) in obj {
                if !current_batch.contains_key(&key) {
                    current_batch.insert(key.clone(), Vec::new());
                }

                match json_value_to_py_object(value) {
                    Ok(py_object) => {
                        if let Some(column) = current_batch.get_mut(&key) {
                            column.push(py_object);
                        }
                    }
                    Err(e) => {
                        eprintln!("Error converting JSON value to Python object: {}", e);
                        if let Some(column) = current_batch.get_mut(&key) {
                            Python::with_gil(|py| {
                                column.push(py.None().into());
                            });
                        }
                    }
                }
            }
        }

        if record_count >= batch_size {
            batches.push(current_batch);
            current_batch = std::collections::HashMap::new();

            let keys: Vec<String> = current_batch.keys().cloned().collect();
            for key in keys {
                current_batch.insert(key, Vec::new());
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

fn json_value_to_py_object(value: serde_json::Value) -> Result<PyObject, Box<dyn std::error::Error>> {
    Ok(Python::with_gil(|py| {
        match value {
            serde_json::Value::Null => py.None().into(),
            serde_json::Value::Bool(b) => b.to_object(py).into(),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    i.to_object(py).into()
                } else if let Some(u) = n.as_u64() {
                    u.to_object(py).into()
                } else if let Some(f) = n.as_f64() {
                    f.to_object(py).into()
                } else {
                    n.to_string().to_object(py).into()
                }
            },
            serde_json::Value::String(s) => s.to_object(py).into(),
            serde_json::Value::Array(arr) => {
                let py_list = PyList::empty(py);
                for item in arr {
                    if let Ok(py_obj) = json_value_to_py_object(item) {
                        py_list.append(py_obj).unwrap_or(());
                    }
                }
                py_list.into()
            },
            serde_json::Value::Object(obj) => {
                let py_dict = PyDict::new(py);
                for (key, value) in obj {
                    if let Ok(py_obj) = json_value_to_py_object(value) {
                        py_dict.set_item(key, py_obj).unwrap_or(());
                    }
                }
                py_dict.into()
            }
        }
    }))
}

fn arrow_array_to_py_objects(array: &arrow::array::ArrayRef) -> Result<Vec<PyObject>, Box<dyn std::error::Error>> {
    use arrow::array::*;
    use arrow::datatypes::*;

    let py_objects = Python::with_gil(|py| -> Result<Vec<PyObject>, Box<dyn std::error::Error>> {
        match array.data_type() {
            DataType::Utf8 => {
                let string_array = array.as_any().downcast_ref::<StringArray>().unwrap();
                let mut result = Vec::new();
                for i in 0..string_array.len() {
                    if string_array.is_null(i) {
                        result.push(py.None().into());
                    } else {
                        result.push(string_array.value(i).to_object(py).into());
                    }
                }
                Ok(result)
            },
            DataType::LargeUtf8 => {
                let string_array = array.as_any().downcast_ref::<LargeStringArray>().unwrap();
                let mut result = Vec::new();
                for i in 0..string_array.len() {
                    if string_array.is_null(i) {
                        result.push(py.None().into());
                    } else {
                        result.push(string_array.value(i).to_object(py).into());
                    }
                }
                Ok(result)
            },
            DataType::Int32 => {
                let int_array = array.as_any().downcast_ref::<Int32Array>().unwrap();
                let mut result = Vec::new();
                for i in 0..int_array.len() {
                    if int_array.is_null(i) {
                        result.push(py.None().into());
                    } else {
                        result.push(int_array.value(i).to_object(py).into());
                    }
                }
                Ok(result)
            },
            DataType::Int64 => {
                let int_array = array.as_any().downcast_ref::<Int64Array>().unwrap();
                let mut result = Vec::new();
                for i in 0..int_array.len() {
                    if int_array.is_null(i) {
                        result.push(py.None().into());
                    } else {
                        result.push(int_array.value(i).to_object(py).into());
                    }
                }
                Ok(result)
            },
            DataType::Float32 => {
                let float_array = array.as_any().downcast_ref::<Float32Array>().unwrap();
                let mut result = Vec::new();
                for i in 0..float_array.len() {
                    if float_array.is_null(i) {
                        result.push(py.None().into());
                    } else {
                        result.push(float_array.value(i).to_object(py).into());
                    }
                }
                Ok(result)
            },
            DataType::Float64 => {
                let float_array = array.as_any().downcast_ref::<Float64Array>().unwrap();
                let mut result = Vec::new();
                for i in 0..float_array.len() {
                    if float_array.is_null(i) {
                        result.push(py.None().into());
                    } else {
                        result.push(float_array.value(i).to_object(py).into());
                    }
                }
                Ok(result)
            },
            DataType::Boolean => {
                let bool_array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
                let mut result = Vec::new();
                for i in 0..bool_array.len() {
                    if bool_array.is_null(i) {
                        result.push(py.None().into());
                    } else {
                        result.push(bool_array.value(i).to_object(py).into());
                    }
                }
                Ok(result)
            },
            _ => {
                let mut result = Vec::new();
                for i in 0..array.len() {
                    if array.is_null(i) {
                        result.push(py.None().into());
                    } else {
                        result.push(format!("{:?}", array.slice(i, 1)).to_object(py).into());
                    }
                }
                Ok(result)
            }
        }
    })?;

    Ok(py_objects)
}
