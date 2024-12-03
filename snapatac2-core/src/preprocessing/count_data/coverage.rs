use crate::preprocessing::{count_data::genome::{FeatureCounter, GenomeBaseIndex, ChromSizes}, Fragment};

use std::collections::HashMap;
use anndata::data::{utils::to_csr_data, CsrNonCanonical};
use bed_utils::bed::{BEDLike, Strand};
use nalgebra_sparse::{CsrMatrix, pattern::SparsityPattern};
use num::traits::{FromPrimitive, One, Zero, SaturatingAdd};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{collections::{BTreeMap, HashSet}, ops::AddAssign};

pub enum ValueType {
    FragmentSingle(CsrNonCanonical<i32>),
    FragmentPaired(CsrNonCanonical<u32>),
}

/// The `CountingStrategy` enum represents different counting strategies.
/// It is used to count the number of fragments that overlap for a given list of genomic features.
/// Three counting strategies are supported: Insertion, Fragment, and Paired-Insertion Counting (PIC).
#[derive(Clone, Copy, Debug)]
pub enum CountingStrategy {
    Insertion, // Insertion based counting
    Fragment,  // Fragment based counting
    PIC,       // Paired-Insertion Counting (PIC)
}

/// The `GenomeCount` struct is used to count the number of reads that overlap
/// for a given list of genomic features (such as genes, exons, ChIP-Seq peaks, or the like).
/// It stores the counts as an iterator of tuples, each containing a
/// compressed sparse row matrix, a start index, and an end index.
/// The output count matrix can be configured to have a fixed bin size and
/// to exclude certain chromosomes.
pub struct GenomeCount<I> {
    index: GenomeBaseIndex,
    fragments: I,
    resolution: usize,
    exclude_chroms: HashSet<String>,
    min_fragment_size: Option<u64>,
    max_fragment_size: Option<u64>,
    counting_strategy: CountingStrategy,
}

impl<I> GenomeCount<I>
where
    I: ExactSizeIterator<Item = (ValueType, usize, usize)>,
{
    pub fn new(chrom_sizes: ChromSizes, fragments: I) -> Self {
        Self {
            index: GenomeBaseIndex::new(&chrom_sizes),
            fragments,
            resolution: 1,
            exclude_chroms: HashSet::new(),
            min_fragment_size: None,
            max_fragment_size: None,
            counting_strategy: CountingStrategy::Insertion,
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

    /// Set the bin size of the output coverage.
    pub fn with_resolution(mut self, s: usize) -> Self {
        self.resolution = s;
        self
    }

    /// Exclude certain chromosomes from the output coverage.
    pub fn exclude(mut self, chroms: &[&str]) -> Self {
        self.exclude_chroms = chroms
            .iter()
            .filter(|x| self.index.chroms.contains(**x))
            .map(|x| x.to_string())
            .collect();
        self
    }

    /// Set the minimum fragment size.
    pub fn min_fragment_size(mut self, size: u64) -> Self {
        self.min_fragment_size = Some(size);
        self
    }

    /// Set the maximum fragment size.
    pub fn max_fragment_size(mut self, size: u64) -> Self {
        self.max_fragment_size = Some(size);
        self
    }

    pub fn set_counting_strategy(mut self, counting_strategy: CountingStrategy) -> Self {
        self.counting_strategy = counting_strategy;
        self
    }

    /// Return an iterator of raw fragments.
    pub fn into_fragments(self) -> impl ExactSizeIterator<Item = (Vec<Vec<Fragment>>, usize, usize)> {
        let index = self.index;
        self.fragments.map(move |(raw_mat, a, b)| {
            let beds = match raw_mat {
                ValueType::FragmentSingle(mat) => {
                    let row_offsets = mat.row_offsets();
                    let col_indices = mat.col_indices();
                    let values = mat.values();
                    (0..(row_offsets.len() - 1)).into_par_iter().map(|i| {
                        let row_start = row_offsets[i];
                        let row_end = row_offsets[i + 1];
                        (row_start..row_end).flat_map(|j| {
                            let (chrom, start) = index.get_position(col_indices[j]);
                            if self.exclude_chroms.contains(chrom) {
                                None
                            } else {
                                let size = values[j];
                                let barcode = None;
                                let count = 1;
                                let end;
                                let strand;
                                if size > 0 {
                                    end = start + size as u64;
                                    strand = Some(Strand::Forward);
                                } else {
                                    end = start + 1;
                                    strand = Some(Strand::Reverse);
                                }
                                Some(Fragment { chrom: chrom.to_string(), start, end, barcode, count, strand })
                            }
                        }).collect()
                    }).collect()
                },
                ValueType::FragmentPaired(mat) => {
                    let row_offsets = mat.row_offsets();
                    let col_indices = mat.col_indices();
                    let values = mat.values();
                    (0..(row_offsets.len() - 1)).into_par_iter().map(|i| {
                        let row_start = row_offsets[i];
                        let row_end = row_offsets[i + 1];
                        (row_start..row_end).flat_map(|j| {
                            let size = values[j] as u64;
                            let (chrom, start) = index.get_position(col_indices[j]);
                            if self.exclude_chroms.contains(chrom)
                                || self.min_fragment_size.map_or(false, |x| size < x)
                                || self.max_fragment_size.map_or(false, |x| size > x)
                            {
                                None
                            } else {
                                Some(Fragment {
                                    chrom: chrom.to_string(),
                                    start,
                                    end: start + size,
                                    barcode: None,
                                    count: 1,
                                    strand: None,
                                })
                            }
                        }).collect()
                    }).collect()
                },
            };
            (beds, a, b)
        })
    }

    /// Return an iterator of raw fragments grouped by a key function.
    /// The key function takes the index of current cell as the input and
    /// returns a key for grouping.
    pub fn into_fragment_groups<F, K>(self, key: F) -> impl ExactSizeIterator<Item = HashMap<K, Vec<Fragment>>>
    where
        F: Fn(usize) -> K,
        K: Eq + PartialEq + std::hash::Hash,
    {
        self.into_fragments().map(move |(vals, start, _)| {
            let mut ordered = HashMap::new();
            vals.into_iter().enumerate().for_each(|(i, xs)| {
                let k = key(start + i);
                ordered
                    .entry(k)
                    .or_insert_with(Vec::new)
                    .extend(xs.into_iter());
            });
            ordered
        })
    }

    /// Output the raw coverage matrix.
    pub fn into_counts<T>(self) -> impl ExactSizeIterator<Item = (CsrMatrix<T>, usize, usize)>
    where
        T: Zero + One + FromPrimitive + SaturatingAdd + Send + Sync,
    {
        let index = self.get_gindex();
        let ori_index = self.index;
        self.fragments.map(move |(raw_mat, i, j)| {
            let new_mat = match raw_mat {
                ValueType::FragmentSingle(mat) => {
                    let row_offsets = mat.row_offsets();
                    let col_indices = mat.col_indices();
                    let n = j - i;
                    let vec = (0..n)
                        .into_par_iter()
                        .map(|row| {
                            let mut count: BTreeMap<usize, T> = BTreeMap::new();
                            let row_start = row_offsets[row];
                            let row_end = row_offsets[row + 1];

                            for k in row_start..row_end {
                                let (chrom, pos) = ori_index.get_position(col_indices[k]);
                                if self.exclude_chroms.is_empty()
                                    || !self.exclude_chroms.contains(chrom)
                                {
                                    let i = index.get_position_rev(chrom, pos);
                                    let entry = count.entry(i).or_insert(Zero::zero());
                                    *entry = entry.saturating_add(&One::one());
                                }
                            }
                            count.into_iter().collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    let (r, c, offset, ind, data) = to_csr_data(vec, index.len());
                    CsrMatrix::try_from_csr_data(r,c,offset,ind, data).unwrap()
                },
                ValueType::FragmentPaired(mat) => {
                    let row_offsets = mat.row_offsets();
                    let col_indices = mat.col_indices();
                    let values = mat.values();
                    let n = j - i;
                    let vec = (0..n)
                        .into_par_iter()
                        .map(|row| {
                            let mut count: BTreeMap<usize, T> = BTreeMap::new();
                            let row_start = row_offsets[row];
                            let row_end = row_offsets[row + 1];
                            for k in row_start..row_end {
                                let (chrom, start) = ori_index.get_position(col_indices[k]);
                                let frag_size = values[k] as u64;
                                let end = start + frag_size - 1;
                                if !self.exclude_chroms.contains(chrom)
                                    && self.min_fragment_size.map_or(true, |x| frag_size >= x)
                                    && self.max_fragment_size.map_or(true, |x| frag_size <= x)
                                {
                                    let start_ = index.get_position_rev(chrom, start);
                                    let end_ = index.get_position_rev(chrom, end);
                                    match self.counting_strategy {
                                        CountingStrategy::Insertion => {
                                            [start_, end_].into_iter().for_each(|i| {
                                                count.entry(i)
                                                    .and_modify(|x| { *x = x.saturating_add(&One::one()) })
                                                    .or_insert(One::one());
                                            });
                                        },
                                        CountingStrategy::Fragment => {
                                            (start_..=end_).into_iter().for_each(|i| {
                                                count.entry(i)
                                                    .and_modify(|x| { *x = x.saturating_add(&One::one()) })
                                                    .or_insert(One::one());
                                            });
                                        },
                                        CountingStrategy::PIC => {
                                            count.entry(start_)
                                                .and_modify(|x| { *x = x.saturating_add(&One::one()) })
                                                .or_insert(One::one());
                                            if start_ != end_ {
                                                count.entry(end_)
                                                    .and_modify(|x| { *x = x.saturating_add(&One::one()) })
                                                    .or_insert(One::one());
                                            }
                                        }
                                    }
                                }
                            }
                            count.into_iter().collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    let (r, c, offset, ind, data) = to_csr_data(vec, index.len());
                    CsrMatrix::try_from_csr_data(r,c,offset,ind, data).unwrap()
                },
            };
            (new_mat, i, j)
        })
    }

    /// Aggregate the coverage by a feature counter.
    pub fn aggregate_counts_by<C, T>(
        self,
        mut counter: C,
    ) -> impl ExactSizeIterator<Item = (CsrMatrix<T>, usize, usize)>
    where
        C: FeatureCounter<Value = T> + Clone + Sync,
        T: Send,
    {
        let n_col = counter.num_features();
        counter.reset();
        let strategy = self.counting_strategy;
        self.into_fragments().map(move |(data, i, j)| {
            let vec = data
                .into_par_iter()
                .map(|beds| {
                    let mut coverage = counter.clone();
                    beds.into_iter().for_each(|fragment| {
                        coverage.insert_fragment(&fragment, &strategy);
                    });
                    coverage.get_counts()
                })
                .collect::<Vec<_>>();
            let (r, c, offset, ind, data) = to_csr_data(vec, n_col);
            (CsrMatrix::try_from_csr_data(r,c,offset,ind, data).unwrap(), i, j)
        })
    }
}

pub fn _fragments_to_insertions(
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
    I: Iterator<Item = CsrMatrix<u32>>,
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
    ) -> impl Iterator<Item = CsrMatrix<T>> {
        let index = self.get_gindex();
        let ori_index = self.index;
        let genome_size = ori_index.len();
        let new_size = index.len();
        self.coverage.map(move |mat| {
            let new_mat = if self.resolution <= 1 {
                let (pattern, data) = mat.into_pattern_and_values();
                let new_data = data
                    .into_iter()
                    .map(|x| T::from_u32(x).unwrap())
                    .collect::<Vec<_>>();
                CsrMatrix::try_from_pattern_and_values(pattern, new_data).unwrap()
            } else {
                let n = mat.nrows();
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
                                let locus1 = ori_index.get_region(ridx);
                                let locus2 = ori_index.get_region(cidx);
                                let i1 = index.get_position_rev(locus1.chrom(), locus1.start());
                                let i2 = index.get_position_rev(locus2.chrom(), locus2.start());
                                let i = i1 * new_size + i2;
                                let val = T::from_u32(*val).unwrap();
                                *count.entry(i).or_insert(Zero::zero()) += val;
                            });
                        count.into_iter().collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let (r, c, offset, ind, data) = to_csr_data(vec, new_size * new_size);
                CsrMatrix::try_from_csr_data(r,c,offset,ind, data).unwrap()
            };
            new_mat
        })
    }
}
