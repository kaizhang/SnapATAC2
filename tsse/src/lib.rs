pub mod qc;
pub mod utils;
use utils::hdf5::*;
use ndarray::{Array, arr1};
use std::path::Path;
use qc::{CellBarcode, read_fragments};
use flate2::read::GzDecoder;

use hdf5::{File, Error, Selection, H5Type, Result, Extent, Group};
use bed_utils::bed::{BED, BEDLike, GenomicRange, tree::GenomeRegions};
use itertools::{Itertools, GroupBy};

///
pub fn create_count_matrix<B, I>(
    file: File,
    fragments: GroupBy<CellBarcode, I, impl FnMut(&BED<5>) -> CellBarcode>,
    regions: &GenomeRegions<B>,
    bin_size: Option<u64>) -> Result<()>
where
    B: BEDLike,
    I: Iterator<Item = BED<5>>,
{
    let group = file.create_group("X")?;
    let mut list_of_barcodes = Vec::new();
    save_sparse_matrix(
        &group,
        fragments.into_iter().map(|(barcode, iter)| {
            list_of_barcodes.push(barcode);
            get_insertion_counts(regions, bin_size, iter)
        }),
        regions.len()
    )?;
    Ok(())
}

/// Compressed Row Storage (CRS) stores the sparse matrix in 3 vectors:
/// one for floating-point numbers (data), and the other two for integers (indices, indptr).
/// The `data` vector stores the values of the nonzero elements of the matrix,
/// as they are traversed in a row-wise fashion.
/// The `indices` vector stores the column indexes of the elements in the `data` vector.
/// The `indptr` vector stores the locations in the `data` vector that start a row,
/// The last element is NNZ.
fn save_sparse_matrix<I>(group: &Group, iter: I, num_col: usize) -> Result<()>
where
    I: Iterator<Item = Vec<(usize, u64)>>,
{
    create_str_attr(group, "encoding-type", "csr_matrix")?;
    create_str_attr(group, "encoding-version", "0.1.0")?;

    let data: ResizableVectorData<u32> = ResizableVectorData::new(group, "data", 100000)?;
    let indices: ResizableVectorData<u32> = ResizableVectorData::new(group, "indices", 100000)?;
    let mut indptr: Vec<u32> = vec![0];
    let iter = iter.scan(0, |state, x| {
        *state = *state + x.len();
        Some((*state, x))
    });
    for chunk in &iter.chunks(1000) {
        let (a, b): (Vec<u32>, Vec<u32>) = chunk.map(|(x, vec)| {
            indptr.push(x.try_into().unwrap());
            vec
        }).flatten().map(|(x, y)| -> (u32, u32) {
            (x.try_into().unwrap(), y.try_into().unwrap()) }).unzip();
        indices.extend(a.into_iter())?;
        data.extend(b.into_iter())?;
    }
    group.new_attr_builder()
        .with_data(&arr1(&[indptr.len() - 1, num_col]))
        .create("shape")?;

    group.new_dataset_builder().deflate(9)
        .with_data(&Array::from_vec(indptr)).create("indptr")?;
    Ok(())
}

fn get_insertion_counts<B, I>(regions: &GenomeRegions<B>,
                             bin_size: Option<u64>,
                             fragments: I) -> Vec<(usize, u64)>
where
    B: BEDLike,
    I: Iterator<Item = BED<5>>,
{
    match bin_size {
        None => regions.get_coverage(to_insertions(fragments)).0.into_iter().enumerate()
            .filter(|(_, x)| *x != 0).collect(),
        Some(k) => regions.get_binned_coverage(k, to_insertions(fragments)).0
            .into_iter().flatten().enumerate().filter(|(_, x)| *x != 0).collect(),
    }
}

fn to_insertions<I>(fragments: I) -> impl Iterator<Item = GenomicRange>
where
    I: Iterator<Item = BED<5>>,
{
    fragments.flat_map(|x| {
        [ GenomicRange::new(x.chrom().to_string(), x.start(), x.start() + 1)
        , GenomicRange::new(x.chrom().to_string(), x.end() - 1, x.end()) ]
    })
}