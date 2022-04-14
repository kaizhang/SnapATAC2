pub mod gene;

use bed_utils::bed::{
    BEDLike, GenomicRange, BED,
    tree::{SparseCoverage, SparseBinnedCoverage},
};
use anyhow::Result;
use anndata_rs::{
    anndata::AnnData,
};
use polars::frame::DataFrame;
use std::fmt::Debug;

pub trait Barcoded {
    fn get_barcode(&self) -> &str;
}

pub struct Fragment(BED<5>);

impl<'a> Barcoded for &'a Fragment {
    fn get_barcode(&self) -> &'a str {
        self.0.name().unwrap()
    }
}

impl Barcoded for Fragment {
    fn get_barcode(&self) -> &str {
        self.0.name().unwrap()
    }
}

pub struct Insertions(pub Vec<(GenomicRange, u32)>);

impl From<&Fragment> for Insertions {
    fn from(rec: &Fragment) -> Self {
        Insertions(vec![
            (GenomicRange::new(rec.0.chrom().to_string(), rec.0.start(), rec.0.start() + 1), 1),
            (GenomicRange::new(rec.0.chrom().to_string(), rec.0.end() - 1, rec.0.end()), 1),
        ])
    }
}

impl From<Fragment> for Insertions {
    fn from(rec: Fragment) -> Self {
        Insertions(vec![
            (GenomicRange::new(rec.0.chrom().to_string(), rec.0.start(), rec.0.start() + 1), 1),
            (GenomicRange::new(rec.0.chrom().to_string(), rec.0.end() - 1, rec.0.end()), 1),
        ])
    }
}

pub trait FeatureCounter {
    type Value;

    /// Reset the counter
    fn reset(&mut self);

    fn insert<B: BEDLike>(&mut self, tag: &B, count: u32);

    fn inserts<B: Into<Insertions>>(&mut self, data: B) {
        let Insertions(ins) = data.into();
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

trait GenomeIndex {
    fn lookup_region(&self, i: usize) -> GenomicRange;
}

pub struct GenomeBaseIndex(Vec<(String, u64)>);

impl GenomeBaseIndex {
    pub fn read_from_anndata(anndata: &AnnData) -> Result<Self> {
        let df: Box<DataFrame> = anndata.get_uns().inner()
            .get_mut("reference_sequences").unwrap().read()?.into_any().downcast().unwrap();
        let chrs = df.column("reference_seq_name").unwrap().utf8().unwrap();
        let chr_sizes = df.column("reference_seq_length").unwrap().u64().unwrap();
        let chrom_index = chrs.into_iter().flatten().map(|x| x.to_string()).zip(
            std::iter::once(0).chain(chr_sizes.into_iter().flatten().scan(0, |state, x| {
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

pub struct GenomeRegions(pub Vec<GenomicRange>);

impl GenomeIndex for GenomeRegions {
    fn lookup_region(&self, i: usize) -> GenomicRange { self.0[i].clone() }
}

pub struct InsertionIter<I, G> {
    pub iter: I,
    pub genome_index: G,
}

impl<I, G, N> Iterator for InsertionIter<I, G>
where
    I: Iterator<Item = Vec<Vec<(usize, N)>>>,
    G: GenomeIndex,
    N: std::convert::TryInto<u32>,
    <N as TryInto<u32>>::Error: Debug,
{
    type Item = Vec<Insertions>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|items| items.into_iter().map(|item| { 
            let ins = item.into_iter().map(|(i, x)|
                (self.genome_index.lookup_region(i), x.try_into().unwrap())
            ).collect();
            Insertions(ins)
        }).collect())
    }
}

