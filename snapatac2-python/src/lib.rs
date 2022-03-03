use pyo3::prelude::*;
use pyo3::types::PyIterator;
use numpy::{PyReadonlyArrayDyn, PyReadonlyArray, Ix1, Ix2, PyArray, IntoPyArray};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

use bed_utils::{bed, bed::GenomicRange, bed::BED};
use std::fs::File;
use std::collections::BTreeMap;
use flate2::read::MultiGzDecoder;
use hdf5;
use linreg::lin_reg_imprecise;
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use linfa::traits::{Fit, Predict};
use rand_core::SeedableRng;
use rand_isaac::Isaac64Rng;
use hora::core::ann_index::ANNIndex;

use snapatac2_core::{
    tile_matrix::{create_tile_matrix},
    gene_score::create_gene_matrix,
    qc,
    utils::{anndata::to_csr_matrix, gene::read_transcripts},
};

#[pyfunction]
fn mk_gene_matrix(input_file: &str,
                  gff_file: &str,
                  output_file: &str,
                  num_cpu: usize,
) -> PyResult<()>
{
    let file = hdf5::File::create(output_file).unwrap();
    let transcripts = if is_gzipped(gff_file) {
        let reader = File::open(gff_file).map(MultiGzDecoder::new)
            .map(std::io::BufReader::new).unwrap();
        read_transcripts(reader).into_values().collect()
    } else {
        let reader = File::open(gff_file).map(std::io::BufReader::new).unwrap();
        read_transcripts(reader).into_values().collect()
    };

    let group = hdf5::File::open(input_file).unwrap().group("obsm").unwrap()
        .group("base_count").unwrap();
    let insertions = qc::read_insertions(group).unwrap();

    rayon::ThreadPoolBuilder::new().num_threads(num_cpu)
        .build().unwrap().install(||
            create_gene_matrix(&file, insertions.iter(), transcripts).unwrap()
        );
    file.close().unwrap();

    Ok(())
}

#[pyfunction]
fn mk_tile_matrix(input_file: &str,
                  bin_size: u64,
                  num_cpu: usize,
                  ) -> PyResult<()>
{
    let file = hdf5::File::open_rw(input_file).unwrap();
    rayon::ThreadPoolBuilder::new().num_threads(num_cpu)
        .build().unwrap().install(|| create_tile_matrix(&file, bin_size).unwrap());
    file.close().unwrap();
    Ok(())
} 

#[pyfunction]
fn import_fragments(
    output_file: &str,
    fragment_file: &str,
    gtf_file: &str,
    chrom_size: BTreeMap<&str, u64>,
    min_num_fragment: u64,
    min_tsse: f64,
    fragment_is_sorted_by_name: bool,
    num_cpu: usize,
    ) -> PyResult<()>
{
    let file = hdf5::File::create(output_file).unwrap();
    let gtf_file_reader = File::open(gtf_file).unwrap();
    let promoters = if is_gzipped(gtf_file) {
        qc::make_promoter_map(qc::read_tss(MultiGzDecoder::new(gtf_file_reader)))
    } else {
        qc::make_promoter_map(qc::read_tss(gtf_file_reader))
    };

    let white_list = if fragment_is_sorted_by_name {
        None
    } else if min_num_fragment > 0 {
        let fragment_file_reader = File::open(fragment_file).unwrap();
        let mut barcode_count = if is_gzipped(fragment_file) {
            qc::get_barcode_count(bed::io::Reader::new(
                MultiGzDecoder::new(fragment_file_reader), Some("#".to_string())
                ).into_records().map(Result::unwrap)
            )
        } else {
            let fragment_file_reader = File::open(fragment_file).unwrap();
            qc::get_barcode_count(bed::io::Reader::new(
                fragment_file_reader, Some("#".to_string()))
                .into_records().map(Result::unwrap)
            )
        };
        Some(barcode_count.drain().filter_map(|(k, v)|
            if v >= min_num_fragment { Some(k) } else { None }).collect()
        )
    } else {
        None
    };

    let result = rayon::ThreadPoolBuilder::new().num_threads(num_cpu)
        .build().unwrap().install(|| {
        let fragment_file_reader = File::open(fragment_file).unwrap();
        if is_gzipped(fragment_file) {
            qc::import_fragments(
            &file,
            bed::io::Reader::new(
                MultiGzDecoder::new(fragment_file_reader), Some("#".to_string())
            ).into_records().map(Result::unwrap),
            &promoters,
            &chrom_size.into_iter().map(|(chr, s)| GenomicRange::new(chr, 0, s)).collect(),
            white_list.as_ref(),
            min_num_fragment,
            min_tsse,
            fragment_is_sorted_by_name,
            ).unwrap()
        } else {
            qc::import_fragments(
            &file,
            bed::io::Reader::new(fragment_file_reader, Some("#".to_string()))
                .into_records().map(Result::unwrap),
            &promoters,
            &chrom_size.into_iter().map(|(chr, s)| GenomicRange::new(chr, 0, s)).collect(),
            white_list.as_ref(),
            min_num_fragment,
            min_tsse,
            fragment_is_sorted_by_name,
            ).unwrap()
        }
    });
    file.close().unwrap();
    Ok(result)
} 



/// Simple linear regression
#[pyfunction]
fn simple_lin_reg(py_iter: &PyIterator) -> PyResult<(f64, f64)> {
    Ok(lin_reg_imprecise(py_iter.map(|x| x.unwrap().extract().unwrap())).unwrap())
}


/// Perform regression
#[pyfunction]
fn jm_regress(
    jm_: PyReadonlyArrayDyn<'_, f64>,
    count_: PyReadonlyArrayDyn<'_, f64>,
) -> PyResult<(f64, f64)> {
    let jm = &jm_.as_array();
    let n_row = jm.shape()[0];
    let count = &count_.as_array();
    let iter = (0..n_row).flat_map(|i| (i+1..n_row)
        .map(move |j| (1.0 / (1.0 / count[[i, 0]] + 1.0 / count[[j, 0]] - 1.0), jm[[i, j]]))
    );
    Ok(lin_reg_imprecise(iter).unwrap())
}

#[pyfunction]
fn intersect_bed(regions: Vec<&str>, bed_file: &str) -> PyResult<Vec<bool>> {
    let bed_tree: bed::tree::BedTree<()> = if is_gzipped(bed_file) {
        bed::io::Reader::new(MultiGzDecoder::new(File::open(bed_file).unwrap()), None)
            .into_records().map(|x: Result<BED<3>, _>| (x.unwrap(), ())).collect()
    } else {
        bed::io::Reader::new(File::open(bed_file).unwrap(), None)
            .into_records().map(|x: Result<BED<3>, _>| (x.unwrap(), ())).collect()
    };
    Ok(regions.into_iter()
        .map(|x| bed_tree.is_overlapped(&str_to_genomic_region(x).unwrap()))
        .collect()
    )
}

#[pyfunction]
fn kmeans<'py>(
    py: Python<'py>,
    n_clusters: usize,
    observations_: PyReadonlyArray<'_, f64, Ix2>,
) -> PyResult<&'py PyArray<usize, Ix1>> {
    let seed = 42;
    let rng: Isaac64Rng = SeedableRng::seed_from_u64(seed);
    let observations = DatasetBase::from(observations_.as_array());
    let model = KMeans::params_with_rng(n_clusters, rng)
        .fit(&observations)
        .expect("KMeans fitted");
    Ok(model.predict(observations).targets.into_pyarray(py))
}

// Search for and save nearest neighbors using ANN
#[pyfunction]
fn approximate_nearest_neighbors(
    data_: PyReadonlyArray<'_, f32, Ix2>,
    k: usize,
) -> PyResult<(Vec<f32>, Vec<i32>, Vec<i32>)>
{
    let data = data_.as_array();
    let dimension = data.shape()[1];
    let mut index = hora::index::hnsw_idx::HNSWIndex::<f32, usize>::new(
        dimension,
        &hora::index::hnsw_params::HNSWParams::<f32>::default(),
    );
    for (i, sample) in data.axis_iter(ndarray::Axis(0)).enumerate() {
        index.add(sample.to_vec().as_slice(), i).unwrap();
    }
    index.build(hora::core::metrics::Metric::Euclidean).unwrap();
    let row_iter = data.axis_iter(ndarray::Axis(0)).map(move |row| {
        index.search_nodes(row.to_vec().as_slice(), k).into_iter()
            .map(|(n, d)| (n.idx().unwrap(), d)).collect::<Vec<_>>()
    });
    Ok(to_csr_matrix(row_iter, data.shape()[0]))
}

// Convert string such as "chr1:134-2222" to `GenomicRange`.
fn str_to_genomic_region(txt: &str) -> Option<GenomicRange> {
    let mut iter1 = txt.splitn(2, ":");
    let chr = iter1.next()?;
    let mut iter2 = iter1.next().map(|x| x.splitn(2, "-"))?;
    let start: u64 = iter2.next().map_or(None, |x| x.parse().ok())?;
    let end: u64 = iter2.next().map_or(None, |x| x.parse().ok())?;
    Some(GenomicRange::new(chr, start, end))
}

/// Determine if a file is gzipped.
fn is_gzipped(file: &str) -> bool {
    MultiGzDecoder::new(File::open(file).unwrap()).header().is_some()
}

#[pymodule]
fn _snapatac2(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(import_fragments, m)?)?;
    m.add_function(wrap_pyfunction!(mk_tile_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(mk_gene_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(simple_lin_reg, m)?)?;
    m.add_function(wrap_pyfunction!(jm_regress, m)?)?;
    m.add_function(wrap_pyfunction!(intersect_bed, m)?)?;
    m.add_function(wrap_pyfunction!(kmeans, m)?)?;
    m.add_function(wrap_pyfunction!(approximate_nearest_neighbors, m)?)?;

    Ok(())
}