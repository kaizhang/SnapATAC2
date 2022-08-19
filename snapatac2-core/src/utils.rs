pub mod gene;
pub mod similarity;

use bed_utils::bed::{
    ParseError, OptionalFields, NarrowPeak, BEDLike, GenomicRange, BED,
    merge_bed_with,
    tree::{SparseCoverage,  SparseBinnedCoverage}, Strand,
};
use anyhow::{Result, anyhow};
use anndata_rs::{
    anndata::{AnnData, AnnDataSet},
    element::ElemCollection,
};
use polars::frame::DataFrame;
use std::fmt::Debug;
use nalgebra_sparse::CsrMatrix;
use itertools::Itertools;

pub type CellBarcode = String;

/// Fragments from single-cell ATAC-seq experiment. Each fragment is represented
/// by a genomic coordinate, cell barcode and a integer value.
pub struct Fragment {
    pub chrom: String,
    pub start: u64,
    pub end: u64,
    pub barcode: CellBarcode,
    pub count: u32,
    pub strand: Option<Strand>,
}

impl BEDLike for Fragment {
    fn chrom(&self) -> &str { &self.chrom }
    fn set_chrom(&mut self, chrom: &str) -> &mut Self {
        self.chrom = chrom.to_string();
        self
    }
    fn start(&self) -> u64 { self.start }
    fn set_start(&mut self, start: u64) -> &mut Self {
        self.start = start;
        self
    }
    fn end(&self) -> u64 { self.end }
    fn set_end(&mut self, end: u64) -> &mut Self {
        self.end = end;
        self
    }
    fn name(&self) -> Option<&str> { None }
    fn score(&self) -> Option<bed_utils::bed::Score> { None }
    fn strand(&self) -> Option<Strand> { None }
}

impl std::str::FromStr for Fragment {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut fields = s.split('\t');
        let chrom = fields.next().ok_or(ParseError::MissingReferenceSequenceName)?.to_string();
        let start = fields.next().ok_or(ParseError::MissingStartPosition)
            .and_then(|s| lexical::parse(s).map_err(ParseError::InvalidStartPosition))?;
        let end = fields.next().ok_or(ParseError::MissingEndPosition)
            .and_then(|s| lexical::parse(s).map_err(ParseError::InvalidEndPosition))?;
        let barcode = fields.next().ok_or(ParseError::MissingName)
            .map(|s| s.into())?;
        let count = fields.next().map_or(Ok(1), |s| if s == "." {
            Ok(1)
        } else {
            lexical::parse(s).map_err(ParseError::InvalidStartPosition)
        })?;
        let strand = fields.next().map_or(Ok(None), |s| if s == "." {
            Ok(None)
        } else {
            s.parse().map(Some).map_err(ParseError::InvalidStrand)
        })?;
        Ok(Fragment { chrom, start, end, barcode, count, strand })
    }
}


/// Genomic interval associating with integer values
pub struct ChromValues(pub Vec<(GenomicRange, u32)>);

impl ChromValues {
    pub fn to_bed<'a>(&'a self, id: &'a str) -> impl Iterator<Item = BED<4>> + 'a {
        self.0.iter().map(|(x, v)| {
            let bed = BED::new(
                x.chrom(), x.start(), x.end(), Some(id.to_string()), None, None,
                OptionalFields::default(),
            );
            vec![bed; *v as usize]
        }).flatten()
    }
}

/// Storing feature counts.
pub trait FeatureCounter {
    type Value;

    /// Reset the counter.
    fn reset(&mut self);

    /// Update counter according to the region and the assocated count.
    fn insert<B: BEDLike>(&mut self, tag: &B, count: u32);

    fn inserts<B: Into<ChromValues>>(&mut self, data: B) {
        let ChromValues(ins) = data.into();
        ins.into_iter().for_each(|(i, v)| self.insert(&i, v));
    }

    /// Retrieve feature ids.
    fn get_feature_ids(&self) -> Vec<String>;

    /// Retrieve feature names.
    fn get_feature_names(&self) -> Option<Vec<String>> { None }

    /// Retrieve stored counts.
    fn get_counts(&self) -> Vec<(usize, Self::Value)>;
}

impl<D: BEDLike + Clone> FeatureCounter for SparseBinnedCoverage<'_, D, u32> {
    type Value = u32;

    fn reset(&mut self) { self.reset(); }

    fn insert<B: BEDLike>(&mut self, tag: &B, count: u32) { self.insert(tag, count); }

    fn get_feature_ids(&self) -> Vec<String> {
        self.get_regions().flatten().map(|x| x.to_string()).collect()
    }

    fn get_counts(&self) -> Vec<(usize, Self::Value)> {
        self.get_coverage().iter().map(|(k, v)| (*k, *v)).collect()
    }
}

impl<D: BEDLike> FeatureCounter for SparseCoverage<'_, D, u32> {
    type Value = u32;

    fn reset(&mut self) { self.reset(); }

    fn insert<B: BEDLike>(&mut self, tag: &B, count: u32) { self.insert(tag, count); }

    fn get_feature_ids(&self) -> Vec<String> {
        self.get_regions().map(|x| x.to_string()).collect()
    }

    fn get_counts(&self) -> Vec<(usize, Self::Value)> {
        self.get_coverage().iter().map(|(k, v)| (*k, *v)).collect()
    }
}


// Convert string such as "chr1:134-2222" to `GenomicRange`.
pub fn str_to_genomic_region(txt: &str) -> Option<GenomicRange> {
    let mut iter1 = txt.splitn(2, ":");
    let chr = iter1.next()?;
    let mut iter2 = iter1.next().map(|x| x.splitn(2, "-"))?;
    let start: u64 = iter2.next().map_or(None, |x| x.parse().ok())?;
    let end: u64 = iter2.next().map_or(None, |x| x.parse().ok())?;
    Some(GenomicRange::new(chr, start, end))
}

/// GenomeIndex stores genomic loci in a compact way. It maps
/// integers to genomic intervals.
trait GenomeIndex {
    fn lookup_region(&self, i: usize) -> GenomicRange;
}

/// Compact representation of consecutive genomic loci.
pub struct GenomeBaseIndex(Vec<(String, u64)>);

impl GenomeBaseIndex {
    pub fn read_from_anndata(elems: &mut ElemCollection) -> Result<Self> {
        let (chrs, chr_sizes): (Vec<_>, Vec<_>) = get_reference_seq_info_(elems)?.into_iter().unzip();
        let chrom_index = chrs.into_iter().zip(
            std::iter::once(0).chain(chr_sizes.into_iter().scan(0, |state, x| {
                *state = *state + x;
                Some(*state)
            }))
        ).collect();
        Ok(Self(chrom_index))
    }
}

impl GenomeIndex for GenomeBaseIndex {
    fn lookup_region(&self, i: usize) -> GenomicRange {
        let i_ = self.0.binary_search_by_key(
            &i, |s| s.1.try_into().unwrap()
        );
        match i_  {
            Ok(j) => GenomicRange::new(self.0[j].0.clone(), 0, 1),
            Err(j) => {
                let (chr, p) = self.0[j - 1].clone();
                GenomicRange::new(chr, i as u64 - p, i as u64 - p + 1)
            },
        }
    }
}

/// A set of genomic loci.
pub struct GenomeRegionIndex(pub Vec<GenomicRange>);

impl GenomeIndex for GenomeRegionIndex {
    fn lookup_region(&self, i: usize) -> GenomicRange { self.0[i].clone() }
}

pub struct ChromValueIter<I, G> {
    pub iter: I,
    pub genome_index: G,
}

impl<I, G, N> Iterator for ChromValueIter<I, G>
where
    I: Iterator<Item = Vec<Vec<(usize, N)>>>,
    G: GenomeIndex,
    N: std::convert::TryInto<u32>,
    <N as TryInto<u32>>::Error: Debug,
{
    type Item = Vec<ChromValues>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|items| items.into_iter().map(|item| { 
            let ins = item.into_iter().map(|(i, x)|
                (self.genome_index.lookup_region(i), x.try_into().unwrap())
            ).collect();
            ChromValues(ins)
        }).collect())
    }
}

pub type TN5InsertionIter = ChromValueIter<Box<dyn Iterator<Item = Vec<Vec<(usize, u8)>>>>, GenomeBaseIndex>;
pub type ChromValueIterator = ChromValueIter<Box<dyn Iterator<Item = Vec<Vec<(usize, u32)>>>>, GenomeRegionIndex>;

/// Read genomic region and its associated account
pub trait ChromValuesReader {
    fn read_insertions(&self, chunk_size: usize) -> Result<TN5InsertionIter>;

    fn read_chrom_values(&self) -> Result<ChromValueIterator>;

    fn get_reference_seq_info(&self) -> Result<Vec<(String, u64)>>;
}


impl ChromValuesReader for AnnData {
    fn read_insertions(&self, chunk_size: usize) -> Result<TN5InsertionIter> {
       Ok(ChromValueIter {
            iter: Box::new(self.get_obsm().inner().get("insertion")
                .expect("cannot find 'insertion' in .obsm")
                .chunked(chunk_size).map(|x| {
                    let csr = *x.into_any().downcast::<CsrMatrix<u8>>().unwrap();
                    csr.row_iter().map(|row|
                        row.col_indices().iter().zip(row.values())
                            .map(|(i, v)| (*i, *v)).collect::<Vec<(usize, u8)>>()
                    ).collect::<Vec<_>>()
                })
            ),
            genome_index: GenomeBaseIndex::read_from_anndata(&mut self.get_uns().inner())?,
        })
    }

    fn read_chrom_values(&self) -> Result<ChromValueIterator>
    {
        Ok(ChromValueIter {
            genome_index: GenomeRegionIndex(
                self.var_names()?.into_iter()
                    .map(|x| str_to_genomic_region(x.as_str()).unwrap()).collect()
            ),
            iter: Box::new(
                self.get_x().chunked(500).map(|x| {
                    let csr = *x.into_any().downcast::<CsrMatrix<u32>>().unwrap();
                    csr.row_iter().map(|row|
                        row.col_indices().iter().zip(row.values())
                            .map(|(i, v)| (*i, *v)).collect::<Vec<_>>()
                    ).collect::<Vec<_>>()
                })
            ),
        })
    }

    fn get_reference_seq_info(&self) -> Result<Vec<(String, u64)>> {
        get_reference_seq_info_(&mut self.get_uns().inner())
    }
}

impl ChromValuesReader for AnnDataSet {
    fn read_insertions(&self, chunk_size: usize) -> Result<TN5InsertionIter> {
        let inner = self.anndatas.inner();
        let ref_seq_same = inner.iter().map(|(_, adata)|
            get_reference_seq_info_(&mut adata.get_uns().inner()).unwrap()
        ).all_equal();
        if !ref_seq_same {
            return Err(anyhow!("reference genome information mismatch"));
        }
        let genome_index = GenomeBaseIndex::read_from_anndata(
            &mut inner.iter().next().unwrap().1.get_uns().inner()
        )?;

        Ok(ChromValueIter {
            iter: Box::new(inner.obsm.data.get("insertion").unwrap()
                .chunked(chunk_size).map(|x| {
                    let csr = *x.into_any().downcast::<CsrMatrix<u8>>().unwrap();
                    csr.row_iter().map(|row|
                        row.col_indices().iter().zip(row.values())
                            .map(|(i, v)| (*i, *v)).collect()
                    ).collect()
                })),
            genome_index,
        })
    }

    fn read_chrom_values(&self) -> Result<ChromValueIterator>
    {
        Ok(ChromValueIter {
            genome_index: GenomeRegionIndex(
                self.var_names()?.into_iter()
                    .map(|x| str_to_genomic_region(x.as_str()).unwrap()).collect()
            ),
            iter: Box::new(
                self.anndatas.inner().x.chunked(500).map(|x| {
                    let csr = *x.into_any().downcast::<CsrMatrix<u32>>().unwrap();
                    csr.row_iter().map(|row|
                        row.col_indices().iter().zip(row.values())
                            .map(|(i, v)| (*i, *v)).collect::<Vec<_>>()
                    ).collect::<Vec<_>>()
                })
            ),
        })
    }

    fn get_reference_seq_info(&self) -> Result<Vec<(String, u64)>> {
        get_reference_seq_info_(&mut self.anndatas.inner().iter().next().unwrap()
            .1.get_uns().inner())
    }
}

pub fn merge_peaks<I>(peaks: I, half_window_size: u64) -> impl Iterator<Item = Vec<NarrowPeak>>
where
    I: Iterator<Item = NarrowPeak>,
{
    merge_bed_with(
        peaks.map(|mut x| {
            let summit = x.start() + x.peak;
            x.start = (summit - half_window_size).max(0);
            x.end = summit + half_window_size + 1;
            x.peak = half_window_size;
            x
        }),
        iterative_merge,
    )
}

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

fn get_reference_seq_info_(elems: &mut ElemCollection) -> Result<Vec<(String, u64)>> {
    let df: Box<DataFrame> = elems.get_mut("reference_sequences").unwrap()
        .read()?.into_any().downcast().unwrap();
    let chrs = df.column("reference_seq_name").unwrap().utf8()?;
    let chr_sizes = df.column("reference_seq_length").unwrap().u64()?;
    Ok(chrs.into_iter().flatten().map(|x| x.to_string()).zip(
        chr_sizes.into_iter().flatten()
    ).collect())
}