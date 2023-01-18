use nalgebra_sparse::CsrMatrix;

/// The IDF transformation.
pub fn idf_l2(csr: &mut CsrMatrix<f64>) {
    let n = csr.ncols();
    let mut idf = vec![0.0; n];
    csr.row_iter().for_each(|row|
        row.col_indices().into_iter().zip(row.values().into_iter()).for_each(|(i, v)|
            idf[*i] += v
        )
    );
    idf.iter_mut().for_each(|c| *c = (n as f64 / (1.0 + *c)).ln());
    csr.row_iter_mut().for_each(|mut row| {
        let (cols, values) = row.cols_and_values_mut();
        let s = cols.into_iter().zip(values.into_iter())
            .map(|(i, v)| {
                *v *= idf[*i];
                *v * *v
            }).sum::<f64>().sqrt();
        values.iter_mut().for_each(|v| *v /= s);
    });
}