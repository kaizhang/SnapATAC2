use crate::genome::{ChromSizes, GenomeBaseIndex};
use crate::preprocessing::Fragment;
use crate::feature_count::{CountingStrategy, FeatureCounter};

use anndata::{data::{utils::to_csr_data, CsrNonCanonical}, ArrayData};
use bed_utils::bed::{BEDLike, BedGraph, GenomicRange, Strand};
use nalgebra_sparse::CsrMatrix;
use num::traits::{FromPrimitive, One, Zero};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::collections::HashMap;
use std::{
    collections::{BTreeMap, HashSet},
    ops::AddAssign,
};

pub enum FragmentDataIter {
    FragmentSingle(Box<dyn ExactSizeIterator<Item = (CsrNonCanonical<i32>, usize, usize)>>),
    FragmentPaired(Box<dyn ExactSizeIterator<Item = (CsrNonCanonical<u32>, usize, usize)>>),
}

fn single_to_fragments(
    index: GenomeBaseIndex,
    exclude_chroms: HashSet<String>,
    data_iter: impl ExactSizeIterator<Item = (CsrNonCanonical<i32>, usize, usize)>
) -> impl ExactSizeIterator<Item = (Vec<Vec<Fragment>>, usize, usize)>
{
    data_iter.map(move |(mat, a, b)| {
        let row_offsets = mat.row_offsets();
        let col_indices = mat.col_indices();
        let values = mat.values();
        let beds = (0..(row_offsets.len() - 1))
            .into_par_iter()
            .map(|i| {
                let row_start = row_offsets[i];
                let row_end = row_offsets[i + 1];
                (row_start..row_end)
                    .flat_map(|j| {
                        let (chrom, start) = index.get_position(col_indices[j]);
                        if exclude_chroms.contains(chrom) {
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
                            Some(Fragment {
                                chrom: chrom.to_string(),
                                start,
                                end,
                                barcode,
                                count,
                                strand,
                            })
                        }
                    })
                    .collect()
            })
            .collect();
        (beds, a, b)
    })
}

fn pair_to_fragments(
    index: GenomeBaseIndex,
    exclude_chroms: HashSet<String>,
    min_fragment_size: Option<u64>,
    max_fragment_size: Option<u64>,
    data_iter: impl ExactSizeIterator<Item = (CsrNonCanonical<u32>, usize, usize)>
) -> impl ExactSizeIterator<Item = (Vec<Vec<Fragment>>, usize, usize)>
{
    data_iter.map(move |(mat, a, b)| {
        let row_offsets = mat.row_offsets();
        let col_indices = mat.col_indices();
        let values = mat.values();
        let beds = (0..(row_offsets.len() - 1))
            .into_par_iter()
            .map(|i| {
                let row_start = row_offsets[i];
                let row_end = row_offsets[i + 1];
                (row_start..row_end)
                    .flat_map(|j| {
                        let size = values[j] as u64;
                        let (chrom, start) = index.get_position(col_indices[j]);
                        if exclude_chroms.contains(chrom)
                            || min_fragment_size.map_or(false, |x| size < x)
                            || max_fragment_size.map_or(false, |x| size > x)
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
                    })
                    .collect()
            })
            .collect();
        (beds, a, b)
    })
}

/// The `FragmentData` struct is used to count the number of reads that overlap
/// for a given list of genomic features (such as genes, exons, ChIP-Seq peaks, or the like).
/// It stores the counts as an iterator of tuples, each containing a
/// compressed sparse row matrix, a start index, and an end index.
/// The output count matrix can be configured to have a fixed bin size and
/// to exclude certain chromosomes.
pub struct FragmentData {
    index: GenomeBaseIndex,
    data_iter: FragmentDataIter,
    resolution: usize,
    exclude_chroms: HashSet<String>,
    min_fragment_size: Option<u64>,
    max_fragment_size: Option<u64>,
    counting_strategy: CountingStrategy,
}

impl FragmentData
{
    pub fn new(chrom_sizes: ChromSizes, data_iter: FragmentDataIter) -> Self {
        Self {
            index: GenomeBaseIndex::new(&chrom_sizes),
            data_iter,
            resolution: 1,
            exclude_chroms: HashSet::new(),
            min_fragment_size: None,
            max_fragment_size: None,
            counting_strategy: CountingStrategy::Insertion,
        }
    }

    pub fn into_inner(self) -> FragmentDataIter {
        self.data_iter
    }

    pub fn is_paired(&self) -> bool {
        matches!(self.data_iter, FragmentDataIter::FragmentPaired(_))
    }

    pub fn get_gindex(&self) -> GenomeBaseIndex {
        if !self.exclude_chroms.is_empty() {
            let chr_sizes: ChromSizes = self
                .index
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
    pub fn into_fragments(
        self,
    ) -> Box<dyn ExactSizeIterator<Item = (Vec<Vec<Fragment>>, usize, usize)>> {
        match self.data_iter {
            FragmentDataIter::FragmentSingle(iter) => Box::new(single_to_fragments(self.index, self.exclude_chroms, iter)),
            FragmentDataIter::FragmentPaired(iter) => Box::new(pair_to_fragments(self.index, self.exclude_chroms, self.min_fragment_size, self.max_fragment_size, iter)),
        }
    }

    /// Return an iterator of raw fragments grouped by a key function.
    /// The key function takes the index of current cell as the input and
    /// returns a key for grouping.
    pub fn into_fragment_groups<F, K>(
        self,
        key: F,
    ) -> impl ExactSizeIterator<Item = HashMap<K, Vec<Fragment>>>
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
    pub fn into_array_iter(self) -> Box<dyn ExactSizeIterator<Item = (CsrMatrix<u32>, usize, usize)>>
    {
        let index = self.get_gindex();
        let ori_index = self.index;
        match self.data_iter {
            FragmentDataIter::FragmentPaired(mat_iter) => {
                Box::new(mat_iter.map(move |(mat, i, j)| {
                    let new_mat = gen_mat_pair::<u32>(
                        &ori_index,
                        &index,
                        &self.exclude_chroms,
                        self.min_fragment_size,
                        self.max_fragment_size,
                        self.counting_strategy,
                        mat,
                    );
                    (new_mat, i, j)
                }))
            }
            FragmentDataIter::FragmentSingle(mat_iter) => {
                Box::new(mat_iter.map(move |(mat, i, j)| {
                    let new_mat = gen_mat_single::<u32>(&ori_index, &index, &self.exclude_chroms, mat);
                    (new_mat, i, j)
                }))
            }
        }
    }

    /// Aggregate the coverage by a feature counter.
    pub fn into_aggregated_array_iter<C>(
        self,
        counter: C,
    ) -> impl ExactSizeIterator<Item = (CsrMatrix<u32>, usize, usize)>
    where
        C: FeatureCounter<Value = u32> + Clone + Sync,
    {
        let n_col = counter.num_features();
        let strategy = self.counting_strategy;
        self.into_fragments().map(move |(data, i, j)| {
            let vec = data
                .into_par_iter()
                .map(|beds| {
                    let mut coverage = counter.clone();
                    beds.into_iter().for_each(|fragment| {
                        coverage.insert_fragment(&fragment, &strategy);
                    });
                    coverage.get_values()
                })
                .collect::<Vec<_>>();
            let (r, c, offset, ind, data) = to_csr_data(vec, n_col);
            (
                CsrMatrix::try_from_csr_data(r, c, offset, ind, data).unwrap(),
                i,
                j,
            )
        })
    }
}

#[inline]
fn gen_mat_single<T>(
    ori_index: &GenomeBaseIndex,
    new_index: &GenomeBaseIndex,
    exclude_chroms: &HashSet<String>,
    mat: CsrNonCanonical<i32>,
) -> CsrMatrix<T>
where
    T: Zero + One + FromPrimitive + AddAssign + Copy + Send + Sync,
{
    let row_offsets = mat.row_offsets();
    let col_indices = mat.col_indices();
    let vec = (0..mat.nrows())
        .into_par_iter()
        .map(|row| {
            let mut count: BTreeMap<usize, T> = BTreeMap::new();
            let row_start = row_offsets[row];
            let row_end = row_offsets[row + 1];

            for k in row_start..row_end {
                let (chrom, pos) = ori_index.get_position(col_indices[k]);
                if exclude_chroms.is_empty() || !exclude_chroms.contains(chrom) {
                    let i = new_index.get_position_rev(chrom, pos);
                    let entry = count.entry(i).or_insert(Zero::zero());
                    *entry += One::one();
                }
            }
            count.into_iter().collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let (r, c, offset, ind, data) = to_csr_data(vec, new_index.len());
    CsrMatrix::try_from_csr_data(r, c, offset, ind, data).unwrap()
}

#[inline]
fn gen_mat_pair<T>(
    ori_index: &GenomeBaseIndex,
    new_index: &GenomeBaseIndex,
    exclude_chroms: &HashSet<String>,
    min_fragment_size: Option<u64>,
    max_fragment_size: Option<u64>,
    counting_strategy: CountingStrategy,
    mat: CsrNonCanonical<u32>,
) -> CsrMatrix<T>
where
    T: Zero + One + FromPrimitive + AddAssign + Copy + Send + Sync,
{
    let row_offsets = mat.row_offsets();
    let col_indices = mat.col_indices();
    let values = mat.values();
    let vec = (0..mat.nrows())
        .into_par_iter()
        .map(|row| {
            let mut count: BTreeMap<usize, T> = BTreeMap::new();
            let row_start = row_offsets[row];
            let row_end = row_offsets[row + 1];
            for k in row_start..row_end {
                let (chrom, start) = ori_index.get_position(col_indices[k]);
                let frag_size = values[k] as u64;
                let end = start + frag_size - 1;
                if !exclude_chroms.contains(chrom)
                    && min_fragment_size.map_or(true, |x| frag_size >= x)
                    && max_fragment_size.map_or(true, |x| frag_size <= x)
                {
                    let start_ = new_index.get_position_rev(chrom, start);
                    let end_ = new_index.get_position_rev(chrom, end);
                    match counting_strategy {
                        CountingStrategy::Insertion => {
                            [start_, end_].into_iter().for_each(|i| {
                                count
                                    .entry(i)
                                    .and_modify(|x| *x += One::one())
                                    .or_insert(One::one());
                            });
                        }
                        CountingStrategy::Fragment => {
                            (start_..=end_).into_iter().for_each(|i| {
                                count
                                    .entry(i)
                                    .and_modify(|x| *x += One::one())
                                    .or_insert(One::one());
                            });
                        }
                        CountingStrategy::PIC => {
                            count
                                .entry(start_)
                                .and_modify(|x| *x += One::one())
                                .or_insert(One::one());
                            if start_ != end_ {
                                count
                                    .entry(end_)
                                    .and_modify(|x| *x += One::one())
                                    .or_insert(One::one());
                            }
                        }
                    }
                }
            }
            count.into_iter().collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let (r, c, offset, ind, data) = to_csr_data(vec, new_index.len());
    CsrMatrix::try_from_csr_data(r, c, offset, ind, data).unwrap()
}

pub struct ContactData<I> {
    index: GenomeBaseIndex,
    coverage: I,
    resolution: usize,
}

impl<I> ContactData<I>
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
                CsrMatrix::try_from_csr_data(r, c, offset, ind, data).unwrap()
            };
            new_mat
        })
    }
}

pub struct BaseData<I> {
    index: GenomeBaseIndex,
    data_iter: I,
    resolution: usize,
    exclude_chroms: HashSet<String>,
}

impl<I> BaseData<I>
where
    I: ExactSizeIterator<Item = (CsrMatrix<f32>, usize, usize)>,
{
    pub fn new(chrom_sizes: ChromSizes, data_iter: I) -> Self {
        Self {
            index: GenomeBaseIndex::new(&chrom_sizes),
            data_iter,
            resolution: 1,
            exclude_chroms: HashSet::new(),
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

    /// Exclude certain chromosomes from the output coverage.
    pub fn exclude(mut self, chroms: &[&str]) -> Self {
        self.exclude_chroms = chroms
            .iter()
            .filter(|x| self.index.chroms.contains(**x))
            .map(|x| x.to_string())
            .collect();
        self
    }

    /// Output the raw coverage matrix. Note the values belong to the same interval
    /// will be aggregated by the mean value.
    pub fn into_array_iter(self) -> impl ExactSizeIterator<Item = (ArrayData, usize, usize)>
    {
        let index = self.get_gindex();
        let ori_index = self.index;

        self.data_iter.map(move |(mat, i, j)| {
            let row_offsets = mat.row_offsets();
            let col_indices = mat.col_indices();
            let values = mat.values();
            let vec = (0..mat.nrows())
                .into_par_iter()
                .map(|row| {
                    let mut count: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
                    let row_start = row_offsets[row];
                    let row_end = row_offsets[row + 1];

                    for k in row_start..row_end {
                        let (chrom, pos) = ori_index.get_position(col_indices[k]);
                        if self.exclude_chroms.is_empty() || !self.exclude_chroms.contains(chrom) {
                            let i = index.get_position_rev(chrom, pos);
                            let entry = count.entry(i).or_insert(Vec::new());
                            entry.push(values[k]);
                        }
                    }
                    count.into_iter().map(|(k, v)| {
                        let len = v.len();
                        let sum: f32 = v.into_iter().sum();
                        (k, sum / len as f32)
                    }).collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let (r, c, offset, ind, data) = to_csr_data(vec, index.len());
            let new_mat = CsrMatrix::try_from_csr_data(r, c, offset, ind, data).unwrap();
            (new_mat.into(), i, j)
        })
    }

    /// Aggregate the coverage by a feature counter. Values belong to the same interval
    /// will be aggregated by the mean value.
    pub fn into_aggregated_array_iter<C>(self, counter: C) -> impl ExactSizeIterator<Item = (ArrayData, usize, usize)>
    where
        C: FeatureCounter<Value=f32> + Clone + Sync,
    {
        let n_col = counter.num_features();
        self.data_iter.map(move |(data, i, j)| {
            let vec = (0..data.nrows())
                .into_par_iter()
                .map(|i| {
                    let mut coverage = counter.clone();
                    let row = data.get_row(i).unwrap();
                    row.col_indices().into_iter().zip(row.values()).for_each(|(idx, val)| {
                        let (chrom, pos) = self.index.get_position(*idx);
                        if self.exclude_chroms.is_empty() || !self.exclude_chroms.contains(chrom) {
                            coverage.insert(&GenomicRange::new(chrom, pos, pos+1), *val);
                        }
                    });
                    coverage.get_values_and_counts().map(|(idx, (val, count))| {
                        (idx, val / count as f32)
                    }).collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let (r, c, offset, ind, data) = to_csr_data(vec, n_col);
            (
                CsrMatrix::try_from_csr_data(r, c, offset, ind, data).unwrap().into(),
                i,
                j,
            )
        })
    }
}

/// `ChromValues` is a type alias for a vector of `BedGraph<N>` objects.
/// Each `BedGraph` instance represents a genomic region along with a
/// numerical value (like coverage or score).
pub type ChromValues<N> = Vec<BedGraph<N>>;

/// `ChromValueIter` represents an iterator over the chromosome values.
/// Each item in the iterator is a tuple of a vector of `ChromValues<N>` objects,
/// a start index, and an end index.
pub struct ChromValueIter<I> {
    pub(crate) iter: I,
    pub(crate) regions: Vec<GenomicRange>,
    pub(crate) length: usize,
}

impl<'a, I, T> ChromValueIter<I>
where
    I: ExactSizeIterator<Item = (CsrMatrix<T>, usize, usize)> + 'a,
    T: Copy,
{
    /// Aggregate the values in the iterator by the given `FeatureCounter`.
    pub fn aggregate_by<C>(
        self,
        mut counter: C,
    ) -> impl ExactSizeIterator<Item = (CsrMatrix<T>, usize, usize)>
    where
        C: FeatureCounter<Value = T> + Clone + Sync,
        T: Sync + Send + num::ToPrimitive,
    {
        let n_col = counter.num_features();
        counter.reset();
        self.iter.map(move |(mat, i, j)| {
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
                            coverage.insert(&self.regions[*idx], *val);
                        });
                    coverage.get_values()
                })
                .collect::<Vec<_>>();
            let (r, c, offset, ind, data) = to_csr_data(vec, n_col);
            (CsrMatrix::try_from_csr_data(r,c,offset,ind, data).unwrap(), i, j)
        })
    }
}

impl<I, T> Iterator for ChromValueIter<I>
where
    I: Iterator<Item = (CsrMatrix<T>, usize, usize)>,
    T: Copy,
{
    type Item = (Vec<ChromValues<T>>, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(x, start, end)| {
            let values = x
                .row_iter()
                .map(|row| {
                    row.col_indices()
                        .iter()
                        .zip(row.values())
                        .map(|(i, v)| BedGraph::from_bed(&self.regions[*i], *v))
                        .collect()
                })
                .collect();
            (values, start, end)
        })
    }
}

impl<I, T> ExactSizeIterator for ChromValueIter<I>
where
    I: Iterator<Item = (CsrMatrix<T>, usize, usize)>,
    T: Copy,
{
    fn len(&self) -> usize {
        self.length
    }
}

