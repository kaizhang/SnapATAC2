pub mod gene;
pub mod similarity;

use bed_utils::bed::{
    OptionalFields,
    BEDLike, GenomicRange, BED,
    tree::{SparseCoverage, SparseBinnedCoverage},
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

pub trait FeatureCounter {
    type Value;

    /// Reset the counter
    fn reset(&mut self);

    fn insert<B: BEDLike>(&mut self, tag: &B, count: u32);

    fn inserts<B: Into<ChromValues>>(&mut self, data: B) {
        let ChromValues(ins) = data.into();
        ins.into_iter().for_each(|(i, v)| self.insert(&i, v));
    }

    fn get_feature_ids(&self) -> Vec<String>;

    fn get_feature_names(&self) -> Option<Vec<String>> { None }

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

/// Integer to genomic position lookup table.
trait GenomeIndex {
    fn lookup_region(&self, i: usize) -> GenomicRange;
}

pub struct GenomeBaseIndex(Vec<(String, u64)>);

impl GenomeBaseIndex {
    pub fn read_from_anndata(elems: &mut ElemCollection) -> Result<Self> {
        let (chrs, chr_sizes): (Vec<_>, Vec<_>) = get_reference_seq_info(elems)?.into_iter().unzip();
        let chrom_index = chrs.into_iter().zip(
            std::iter::once(0).chain(chr_sizes.into_iter().scan(0, |state, x| {
                *state = *state + x;
                Some(*state)
            }))
        ).collect();
        Ok(Self(chrom_index))
    }
}

pub fn get_reference_seq_info(elems: &mut ElemCollection) -> Result<Vec<(String, u64)>> {
    let df: Box<DataFrame> = elems.get_mut("reference_sequences").unwrap()
        .read()?.into_any().downcast().unwrap();
    let chrs = df.column("reference_seq_name").unwrap().utf8()?;
    let chr_sizes = df.column("reference_seq_length").unwrap().u64()?;
    Ok(chrs.into_iter().flatten().map(|x| x.to_string()).zip(
        chr_sizes.into_iter().flatten()
    ).collect())
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

pub struct GenomeRegions(pub Vec<GenomicRange>);

impl GenomeIndex for GenomeRegions {
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
pub type ChromValueIterator = ChromValueIter<Box<dyn Iterator<Item = Vec<Vec<(usize, u32)>>>>, GenomeRegions>;

/// Read genomic region and its associated account
pub trait ChromValuesReader {
    fn read_insertions(&self, chunk_size: usize) -> Result<TN5InsertionIter>;

    fn read_chrom_values(&self) -> Result<ChromValueIterator>;
}


impl ChromValuesReader for AnnData {
    fn read_insertions(&self, chunk_size: usize) -> Result<TN5InsertionIter> {
       Ok(ChromValueIter {
            iter: Box::new(
                self.get_obsm().inner().get("insertion").unwrap().chunked(chunk_size).map(|x| {
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
            genome_index: GenomeRegions(
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
}

impl ChromValuesReader for AnnDataSet {
    fn read_insertions(&self, chunk_size: usize) -> Result<TN5InsertionIter> {
        let inner = self.anndatas.inner();
        let ref_seq_same = inner.iter().map(|(_, adata)|
            get_reference_seq_info(&mut adata.get_uns().inner()).unwrap()
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
            genome_index: GenomeRegions(
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
}