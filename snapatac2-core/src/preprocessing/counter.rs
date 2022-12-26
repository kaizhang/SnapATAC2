use std::collections::{BTreeMap, HashSet};
use indexmap::map::IndexMap;
use bed_utils::bed::{GenomicRange, BEDLike, tree::SparseCoverage};
use num::traits::{ToPrimitive, NumCast};

use super::Promoters;

/// A structure that stores the feature counts.
pub trait FeatureCounter {
    type Value;

    /// Reset the counter.
    fn reset(&mut self);

    /// Update counter according to the region and the assocated count.
    fn insert<B: BEDLike, N: ToPrimitive + Copy>(&mut self, tag: &B, count: N);

    /*
    fn inserts<B, N>(&mut self, data: B)
    where
        B: Into<ChromValues<N>>,
        N: ToPrimitive + Copy,
    {
        data.into().into_iter().for_each(|x| self.insert(&x, x.value));
    }
    */

    /// Retrieve feature ids.
    fn get_feature_ids(&self) -> Vec<String>;

    /// Retrieve feature names.
    fn get_feature_names(&self) -> Option<Vec<String>> { None }

    /// Retrieve stored counts.
    fn get_counts(&self) -> Vec<(usize, Self::Value)>;
}

impl<D: BEDLike> FeatureCounter for SparseCoverage<'_, D, u32> {
    type Value = u32;

    fn reset(&mut self) { self.reset(); }

    fn insert<B: BEDLike, N: ToPrimitive + Copy>(&mut self, tag: &B, count: N) {
        self.insert(tag, <u32 as NumCast>::from(count).unwrap());
    }

    fn get_feature_ids(&self) -> Vec<String> {
        self.get_regions().map(|x| x.to_genomic_range().pretty_show()).collect()
    }

    fn get_counts(&self) -> Vec<(usize, Self::Value)> {
        self.get_coverage().iter().map(|(k, v)| (*k, *v)).collect()
    }
}

#[derive(Clone)]
pub struct TranscriptCount<'a> {
    counter: SparseCoverage<'a, GenomicRange, u32>,
    promoters: &'a Promoters,
}

impl<'a> TranscriptCount<'a> {
    pub fn new(promoters: &'a Promoters) -> Self {
        Self {
            counter: SparseCoverage::new(&promoters.regions),
            promoters,
        }
    }

    pub fn gene_names(&self) -> Vec<String> {
        self.promoters
            .transcripts
            .iter()
            .map(|x| x.gene_name.clone())
            .collect()
    }
}

#[derive(Clone)]
pub struct GeneCount<'a> {
    counter: TranscriptCount<'a>,
    gene_name_to_idx: IndexMap<&'a str, usize>,
}

impl<'a> GeneCount<'a> {
    pub fn new(counter: TranscriptCount<'a>) -> Self {
        let gene_name_to_idx: IndexMap<_, _> = counter
            .promoters
            .transcripts
            .iter()
            .map(|x| x.gene_name.as_str())
            .collect::<HashSet<_>>()
            .into_iter()
            .enumerate()
            .map(|(a, b)| (b, a))
            .collect();
        Self {
            counter,
            gene_name_to_idx,
        }
    }
}


impl FeatureCounter for TranscriptCount<'_> {
    type Value = u32;

    fn reset(&mut self) { self.counter.reset(); }

    fn insert<B: BEDLike, N: ToPrimitive + Copy>(&mut self, tag: &B, count: N) {
        self.counter.insert(tag, <u32 as NumCast>::from(count).unwrap());
    }

    fn get_feature_ids(&self) -> Vec<String> {
        self.promoters.transcripts.iter().map(|x| x.transcript_id.clone()).collect()
    }

    fn get_counts(&self) -> Vec<(usize, Self::Value)> {
        self.counter.get_counts()
    }
}

impl FeatureCounter for GeneCount<'_> {
    type Value = u32;

    fn reset(&mut self) { self.counter.reset(); }

    fn insert<B: BEDLike, N: ToPrimitive + Copy>(&mut self, tag: &B, count: N) {
        self.counter.insert(tag, <u32 as NumCast>::from(count).unwrap());
    }

    fn get_feature_ids(&self) -> Vec<String> {
        self.gene_name_to_idx.keys().map(|x| x.to_string()).collect()
    }

    fn get_counts(&self) -> Vec<(usize, Self::Value)> {
        let mut counts = BTreeMap::new();
        self.counter.get_counts().into_iter().for_each(|(k, v)| {
            let idx = *self.gene_name_to_idx.get(
                self.counter.promoters.transcripts[k].gene_name.as_str()
            ).unwrap();
            let current_v = counts.entry(idx).or_insert(v);
            if *current_v < v { *current_v = v }
        });
        counts.into_iter().collect()
    }
}