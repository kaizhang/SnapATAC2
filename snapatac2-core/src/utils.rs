pub mod hdf5;
pub mod anndata;
pub mod gene;

use crate::utils::anndata::AnnSparseRow;

use bed_utils::bed::{
    BEDLike, GenomicRange, BED,
    tree::{SparseCoverage, SparseBinnedCoverage},
};

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

impl<T> Barcoded for AnnSparseRow<T> {
    fn get_barcode(&self) -> &str { &self.row_name }
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

impl From<AnnSparseRow<u32>> for Insertions {
    fn from(rec: AnnSparseRow<u32>) -> Self {
        Insertions(rec.data.into_iter()
            .map(|(i, v)| (i.replace(&[':', '-'], "\t").parse().unwrap(), v))
            .collect()
        )
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