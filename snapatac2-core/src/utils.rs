pub mod similarity;

use bed_utils::bed::{BEDLike, NarrowPeak, merge_bed_with};
use nalgebra_sparse::CsrMatrix;

pub fn from_csr_rows<T>(rows: Vec<Vec<(usize, T)>>, num_cols: usize) -> CsrMatrix<T> {
    let num_rows = rows.len();
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = Vec::with_capacity(num_rows + 1);
    let mut nnz = 0;
    for row in rows {
        indptr.push(nnz);
        for (col, val) in row {
            data.push(val);
            indices.push(col);
            nnz += 1;
        }
    }
    indptr.push(nnz);
    CsrMatrix::try_from_csr_data(num_rows, num_cols, indptr, indices, data).unwrap()
}

pub fn merge_peaks<I>(peaks: I, half_window_size: u64) -> impl Iterator<Item = Vec<NarrowPeak>>
where
    I: Iterator<Item = NarrowPeak>,
{
    fn iterative_merge(mut peaks: Vec<NarrowPeak>) -> Vec<NarrowPeak> {
        let mut result = Vec::new();
        while !peaks.is_empty() {
            let best_peak = peaks.iter()
                .max_by(|a, b| a.p_value.partial_cmp(&b.p_value).unwrap()).unwrap()
                .clone();
            peaks = peaks.into_iter().filter(|x| x.n_overlap(&best_peak) == 0).collect();
            result.push(best_peak);
        }
        result
    }

    merge_bed_with(
        peaks.map(move |mut x| {
            let summit = x.start() + x.peak;
            x.start = summit.saturating_sub(half_window_size);
            x.end = summit + half_window_size + 1;
            x.peak = summit - x.start;
            x
        }),
        iterative_merge,
        None::<&str>,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use bed_utils::bed::io::Reader;

    #[test]
    fn test_merge_peaks() {
        let input = "chr1\t9977\t16487\ta\t1000\t.\t74.611\t290.442\t293.049\t189
chr1\t9977\t16487\tb\t1000\t.\t74.1871\t295.33\t290.939\t425
chr1\t9977\t16487\tb\t1000\t.\t74.1871\t295\t290.939\t425
chr1\t9977\t16487\tb\t1000\t.\t74.1871\t295\t290.939\t625
chr1\t9977\t16487\tb\t1000\t.\t74.1871\t290\t290.939\t925
chr2\t9977\t16487\tb\t1000\t.\t74.1871\t295\t290.939\t625
chr2\t9977\t16487\tb\t1000\t.\t74.1871\t290\t290.939\t325
chr2\t9977\t16487\tb\t1000\t.\t74.1871\t290\t290.939\t525
chr2\t9977\t16487\tb\t1000\t.\t74.1871\t290\t290.939\t725
chr3\t0\t1164\tb\t1000\t.\t74.1871\t290\t290.939\t100
";
        let output = "chr1\t10202\t10603\tb\t1000\t.\t74.1871\t295.33\t290.939\t200
chr1\t10702\t11103\tb\t1000\t.\t74.1871\t290\t290.939\t200
chr2\t10402\t10803\tb\t1000\t.\t74.1871\t295\t290.939\t200
chr3\t0\t301\tb\t1000\t.\t74.1871\t290\t290.939\t100
";

        let expected: Vec<NarrowPeak> = Reader::new(output.as_bytes(), None)
            .into_records().map(|x| x.unwrap()).collect();
        let result: Vec<NarrowPeak> = merge_peaks(
            Reader::new(input.as_bytes(), None).into_records().map(|x| x.unwrap()),
            200
        ).flatten().collect();

        assert_eq!(expected, result);
    }
}