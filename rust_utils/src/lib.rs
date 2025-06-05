use pyo3::prelude::*;
use pyo3::types::PyList; // Ensure PyList is imported

// Helper function to calculate the median of a slice of f64.
fn median_f64_slice(slice: &[f64]) -> f64 {
    debug_assert!(!slice.is_empty(), "median_f64_slice called with empty slice");
    let mut sorted_window = slice.to_vec();
    sorted_window.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted_window[sorted_window.len() / 2]
}

/// Applies the Hampel filter to a time series to detect and replace outliers.
/// This function is exposed to Python.
#[pyfunction]
fn hampel_filter(py: Python<'_>, data: Vec<f64>, half_window: usize, n_sigma: f64) -> PyResult<Py<PyList>> {
    let mut filtered_data_rust_vec = data.clone();
    let n = data.len();
    
    let window_size = half_window.saturating_mul(2).saturating_add(1);

    if n < window_size || window_size <= 1 {
        // Create PyList and then convert the Bound reference to an owned Py<PyList>
        let py_list_bound = PyList::new(py, &filtered_data_rust_vec);
        return Ok(py_list_bound?.to_owned().into()); // Use .to_owned()
    }

    for i in half_window..(n - half_window) {
        let window_start_idx = i - half_window;
        let window_end_idx = i + half_window;
        let current_window_slice = &data[window_start_idx..=window_end_idx];
        
        let window_median = median_f64_slice(current_window_slice);

        let abs_deviations: Vec<f64> = current_window_slice
            .iter()
            .map(|&val| (val - window_median).abs())
            .collect();
        
        let mad_value = median_f64_slice(&abs_deviations);

        const K_MAD_SCALING_FACTOR: f64 = 1.4826;
        let threshold = n_sigma * K_MAD_SCALING_FACTOR * mad_value;
        
        let current_original_value = data[i];
        
        if (current_original_value - window_median).abs() > threshold {
            filtered_data_rust_vec[i] = window_median;
        }
    }
    
    // Create PyList and then convert the Bound reference to an owned Py<PyList>
    let py_list_result_bound = PyList::new(py, &filtered_data_rust_vec);
    Ok(py_list_result_bound?.to_owned().into()) // Use .to_owned()
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_utils(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hampel_filter, m)?)?;
    Ok(())
}