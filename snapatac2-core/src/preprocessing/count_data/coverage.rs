use crate::preprocessing::count_data::genome::{
    FeatureCounter, GenomeBaseIndex, ChromSizes, ChromValues,
};

use anndata::data::{utils::to_csr_data, CsrNonCanonical};
use bed_utils::bed::{BedGraph, BEDLike};
use nalgebra_sparse::{CsrMatrix, pattern::SparsityPattern};
use num::traits::{FromPrimitive, Zero};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{collections::{BTreeMap, HashSet}, ops::AddAssign};

/// `GenomeCoverage` represents a genome's base-resolution coverage.
/// It stores the coverage as an iterator of tuples, each containing a
/// compressed sparse row matrix, a start index, and an end index.
/// It also keeps track of the resolution and excluded chromosomes.
pub struct GenomeCoverage<I> {
    index: GenomeBaseIndex,
    coverage: I,
    resolution: usize,
    exclude_chroms: HashSet<String>,
}

impl<I> GenomeCoverage<I>
where
    I: ExactSizeIterator<Item = (CsrMatrix<u8>, usize, usize)>,
{
    pub fn new(chrom_sizes: ChromSizes, coverage: I) -> Self {
        Self {
            index: GenomeBaseIndex::new(&chrom_sizes),
            coverage,
            resolution: 1,
            exclude_chroms: HashSet::new(),
        }
    }

    pub fn get_gindex(&self) -> GenomeBaseIndex {
        if !self.exclude_chroms.is_empty() {
            let chr_sizes: ChromSizes = self.index
                .chrom_sizes()
                .filter_map(|(chr, size)| {
                    if self.exclude_chroms.contains(chr) {
                        None
                    } else {
                        Some((chr.clone(), size))
                    }
                })
                .collect();
            GenomeBaseIndex::new(&chr_sizes).with_step(self.resolution)
        } else {
            self.index.with_step(self.resolution)
        }
    }

    /// Set the resolution of the coverage matrix.
    pub fn with_resolution(mut self, s: usize) -> Self {
        self.resolution = s;
        self
    }

    pub fn exclude(mut self, chroms: &[&str]) -> Self {
        self.exclude_chroms = chroms
            .iter()
            .filter(|x| self.index.chroms.contains(**x))
            .map(|x| x.to_string())
            .collect();
        self
    }

    /// Convert the coverage matrix into a vector of `BedGraph` objects.
    pub fn into_chrom_values<T: Zero + FromPrimitive + AddAssign + Send>(
        self,
    ) -> impl ExactSizeIterator<Item = (Vec<ChromValues<T>>, usize, usize)> {
        if !self.exclude_chroms.is_empty() {
            todo!("Implement exclude_chroms")
        }
        let index = self.get_gindex();
        self.coverage.map(move |(mat, i, j)| {
            let n = j - i;
            let values = (0..n)
                .into_par_iter()
                .map(|k| {
                    let row = mat.get_row(k).unwrap();
                    let row_entry_iter = row.col_indices().into_iter().zip(row.values());
                    if index.step <= 1 {
                        row_entry_iter
                            .map(|(idx, val)| {
                                let region = index.get_locus(*idx);
                                BedGraph::from_bed(&region, T::from_u8(*val).unwrap())
                            })
                            .collect::<Vec<_>>()
                    } else {
                        let mut count: BTreeMap<usize, T> = BTreeMap::new();
                        row_entry_iter.for_each(|(idx, val)| {
                            let i = index.get_coarsed_position(*idx);
                            let val = T::from_u8(*val).unwrap();
                            *count.entry(i).or_insert(Zero::zero()) += val;
                        });
                        count
                            .into_iter()
                            .map(|(i, val)| {
                                let region = index.get_locus(i);
                                BedGraph::from_bed(&region, val)
                            })
                            .collect::<Vec<_>>()
                    }
                })
                .collect();
            (values, i, j)
        })
    }

    /// Output the raw coverage matrix.
    pub fn into_values<T: Zero + FromPrimitive + AddAssign + Send>(
        self,
    ) -> impl ExactSizeIterator<Item = (CsrMatrix<T>, usize, usize)> {
        let index = self.get_gindex();
        let ori_index = self.index;
        self.coverage.map(move |(mat, i, j)| {
            let new_mat = if self.resolution <= 1 && self.exclude_chroms.is_empty() {
                let (pattern, data) = mat.into_pattern_and_values();
                let new_data = data
                    .into_iter()
                    .map(|x| T::from_u8(x).unwrap())
                    .collect::<Vec<_>>();
                CsrMatrix::try_from_pattern_and_values(pattern, new_data).unwrap()
            } else {
                let n = j - i;
                let vec = (0..n)
                    .into_par_iter()
                    .map(|k| {
                        let row = mat.get_row(k).unwrap();
                        let mut count: BTreeMap<usize, T> = BTreeMap::new();
                        row.col_indices()
                            .into_iter()
                            .zip(row.values())
                            .for_each(|(idx, val)| {
                                let locus = ori_index.get_locus(*idx);
                                if self.exclude_chroms.is_empty()
                                    || !self.exclude_chroms.contains(locus.chrom())
                                {
                                    let i = index.get_position(locus.chrom(), locus.start());
                                    let val = T::from_u8(*val).unwrap();
                                    *count.entry(i).or_insert(Zero::zero()) += val;
                                }
                            });
                        count.into_iter().collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let (r, c, offset, ind, data) = to_csr_data(vec, index.len());
                CsrMatrix::try_from_csr_data(r,c,offset,ind, data).unwrap()
            };
            (new_mat, i, j)
        })
    }

    /// Aggregate the coverage by a feature counter.
    pub fn aggregate_by<C, T>(
        self,
        mut counter: C,
    ) -> impl ExactSizeIterator<Item = (CsrMatrix<T>, usize, usize)>
    where
        C: FeatureCounter<Value = T> + Clone + Sync,
        T: Send,
    {
        if !self.exclude_chroms.is_empty() {
            todo!("Implement exclude_chroms")
        }
        let n_col = counter.num_features();
        counter.reset();
        let index = self.index;
        self.coverage.map(move |(mat, i, j)| {
            let n = j - i;
            let vec = (0..n)
                .into_par_iter()
                .map(|k| {
                    let row = mat.get_row(k).unwrap();
                    let mut coverage = counter.clone();
                    row.col_indices()
                        .into_iter()
                        .zip(row.values())
                        .for_each(|(idx, val)| {
                            coverage.insert(&index.get_locus(*idx), *val);
                        });
                    coverage.get_counts()
                })
                .collect::<Vec<_>>();
            let (r, c, offset, ind, data) = to_csr_data(vec, n_col);
            (CsrMatrix::try_from_csr_data(r,c,offset,ind, data).unwrap(), i, j)
        })
    }
}

pub fn fragments_to_insertions(
    fragments: CsrNonCanonical<i32>,
) -> CsrMatrix<u8> {
    let nrows = fragments.nrows();
    let ncols = fragments.ncols();
    let mut new_values = Vec::new();
    let mut new_col_indices = Vec::new();
    let mut new_row_offsets = Vec::new();
    new_row_offsets.push(0);

    let row_offsets = fragments.row_offsets();
    let col_indices = fragments.col_indices();

    for i in 0..(row_offsets.len() - 1) {
        let row_start = row_offsets[i];
        let row_end = row_offsets[i + 1];

        let mut accum = 0;
        let mut count: u8 = 0;
        for j in row_start..row_end {
            let col = col_indices[j];
            if col == accum {
                count = count.saturating_add(1);
            } else {
                if count > 0 {
                    new_values.push(count);
                    new_col_indices.push(accum);
                }
                accum = col;
                count = 1;
            }
        }
        if count > 0 {
            new_values.push(count);
            new_col_indices.push(accum);
        }
        new_row_offsets.push(new_values.len());
    }
    let pattern = unsafe {
        SparsityPattern::from_offset_and_indices_unchecked(nrows, ncols, new_row_offsets, new_col_indices)
    };
    CsrMatrix::try_from_pattern_and_values(pattern, new_values).unwrap()
}

pub struct ContactMap<I> {
    index: GenomeBaseIndex,
    coverage: I,
    resolution: usize,
}

impl<I> ContactMap<I>
where
    I: ExactSizeIterator<Item = (CsrMatrix<u8>, usize, usize)>,
{
    pub fn new(chrom_sizes: ChromSizes, coverage: I) -> Self {
        Self {
            index: GenomeBaseIndex::new(&chrom_sizes),
            coverage,
            resolution: 1,
        }
    }

    pub fn get_gindex(&self) -> GenomeBaseIndex {
        self.index.with_step(self.resolution)
    }

    /// Set the resolution of the coverage matrix.
    pub fn with_resolution(mut self, s: usize) -> Self {
        self.resolution = s;
        self
    }

    /// Output the raw coverage matrix.
    pub fn into_values<T: Zero + FromPrimitive + AddAssign + Send>(
        self,
    ) -> impl ExactSizeIterator<Item = (CsrMatrix<T>, usize, usize)> {
        let index = self.get_gindex();
        let ori_index = self.index;
        let genome_size = ori_index.len();
        let new_size = index.len();
        self.coverage.map(move |(mat, i, j)| {
            let new_mat = if self.resolution <= 1 {
                let (pattern, data) = mat.into_pattern_and_values();
                let new_data = data
                    .into_iter()
                    .map(|x| T::from_u8(x).unwrap())
                    .collect::<Vec<_>>();
                CsrMatrix::try_from_pattern_and_values(pattern, new_data).unwrap()
            } else {
                let n = j - i;
                let vec = (0..n)
                    .into_par_iter()
                    .map(|k| {
                        let row = mat.get_row(k).unwrap();
                        let mut count: BTreeMap<usize, T> = BTreeMap::new();
                        row.col_indices()
                            .into_iter()
                            .zip(row.values())
                            .for_each(|(idx, val)| {
                                let ridx = idx / genome_size;
                                let cidx = idx % genome_size;
                                let locus1 = ori_index.get_locus(ridx);
                                let locus2 = ori_index.get_locus(cidx);
                                let i1 = index.get_position(locus1.chrom(), locus1.start());
                                let i2 = index.get_position(locus2.chrom(), locus2.start());
                                let i = i1 * new_size + i2;
                                let val = T::from_u8(*val).unwrap();
                                *count.entry(i).or_insert(Zero::zero()) += val;
                            });
                        count.into_iter().collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let (r, c, offset, ind, data) = to_csr_data(vec, new_size * new_size);
                CsrMatrix::try_from_csr_data(r,c,offset,ind, data).unwrap()
            };
            (new_mat, i, j)
        })
    }
}
