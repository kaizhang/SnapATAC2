use anndata::ArrayData;
use pyanndata::data::PyArrayData;
use pyo3::{prelude::*, PyResult};
use numpy::{PyReadonlyArray, Ix2};
use snapatac2_core::utils::knn;

#[pyfunction]
pub(crate) fn nearest_neighbour_graph(
    data: PyReadonlyArray<'_, f64, Ix2>,
    k: usize,
) -> PyResult<PyArrayData>
{
    let data = data.as_array();
    let knn = knn::nearest_neighbour_graph(data, k);
    Ok(ArrayData::from(knn).into())
}

// Search for and save nearest neighbors using ANN
#[pyfunction]
pub(crate) fn approximate_nearest_neighbour_graph(
    data: PyReadonlyArray<'_, f32, Ix2>,
    k: usize,
) -> PyResult<PyArrayData>
{
    let data = data.as_array();
    let knn = knn::approximate_nearest_neighbour_graph(data, k);
    Ok(ArrayData::from(knn).into())
}