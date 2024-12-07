use anyhow::bail;
use bed_utils::bed::map::GIntervalIndexSet;
use bed_utils::bed::BEDLike;
use indexmap::map::IndexMap;
use itertools::Itertools;
use num::{
    traits::{NumAssignOps, NumCast, ToPrimitive},
    Num,
};
use std::{collections::BTreeMap, fmt::Debug};

use crate::genome::Promoters;
use crate::preprocessing::Fragment;

/// The `CountingStrategy` enum represents different counting strategies.
/// It is used to count the number of fragments that overlap for a given list of genomic features.
/// Three counting strategies are supported: Insertion, Fragment, and Paired-Insertion Counting (PIC).
#[derive(Clone, Copy, Debug)]
pub enum CountingStrategy {
    Insertion, // Insertion based counting
    Fragment,  // Fragment based counting
    PIC,       // Paired-Insertion Counting (PIC)
}

impl TryFrom<&str> for CountingStrategy {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "insertion" => Ok(CountingStrategy::Insertion),
            "fragment" => Ok(CountingStrategy::Fragment),
            "paired-insertion" => Ok(CountingStrategy::PIC),
            _ => bail!("Counting strategy must be one of 'insertion', 'fragment', or 'paired-insertion'"),
        }
    }
}

/// `FeatureCounter` is a trait that provides an interface for counting genomic features.
/// Types implementing `FeatureCounter` can store feature counts and provide several
/// methods for manipulating and retrieving those counts.
pub trait FeatureCounter {
    type Value;

    /// Returns the total number of distinct features counted.
    fn num_features(&self) -> usize {
        self.get_feature_ids().len()
    }

    /// Resets the counter for all features.
    fn reset(&mut self);

    /// Updates the counter according to the given region and count.
    fn insert<B: BEDLike, N: ToPrimitive + Copy>(&mut self, tag: &B, count: N);

    /// Updates the counter according to the given fragment
    fn insert_fragment(&mut self, tag: &Fragment, strategy: &CountingStrategy);

    /// Returns a vector of feature ids.
    fn get_feature_ids(&self) -> Vec<String>;

    /// Returns a vector of feature names if available.
    fn get_feature_names(&self) -> Option<Vec<String>> {
        None
    }

    /// Returns a vector of tuples, each containing a feature's index and its count.
    fn get_values(&self) -> Vec<(usize, Self::Value)>;

    /// Returns a vector of tuples, each containing a feature's index and its average count.
    fn get_values_and_counts(&self) -> impl Iterator<Item = (usize, (Self::Value, usize))>;
}

#[derive(Clone)]
pub struct RegionCounter<'a, V> {
    regions: &'a GIntervalIndexSet,
    values: BTreeMap<usize, (V, usize)>,
}

impl<'a, V> RegionCounter<'a, V> {
    pub fn new(regions: &'a GIntervalIndexSet) -> Self {
        Self {
            regions,
            values: BTreeMap::new(),
        }
    }
}

impl<V: Num + NumCast + NumAssignOps + Copy> FeatureCounter for RegionCounter<'_, V> {
    type Value = V;

    fn reset(&mut self) {
        self.values.clear();
    }

    fn insert<B: BEDLike, N: ToPrimitive + Copy>(&mut self, tag: &B, count: N) {
        let val = <V as NumCast>::from(count).unwrap();
        self.regions.find_index_of(tag).for_each(|idx| {
            self.values
                .entry(idx)
                .and_modify(|(v, c)| {
                    *v += val;
                    *c += 1;
                })
                .or_insert((val, 1));
        });
    }

    fn insert_fragment(&mut self, tag: &Fragment, strategy: &CountingStrategy) {
        if tag.is_single() {
            tag.to_insertions().iter().for_each(|x| {
                self.insert(x, V::one());
            });
        } else {
            match strategy {
                CountingStrategy::Fragment => {
                    self.insert(tag, V::one());
                }
                CountingStrategy::Insertion => {
                    tag.to_insertions().iter().for_each(|x| {
                        self.insert(x, V::one());
                    });
                }
                CountingStrategy::PIC => {
                    tag.to_insertions()
                        .into_iter()
                        .flat_map(|x| self.regions.find_index_of(&x))
                        .unique()
                        .collect::<Vec<_>>()
                        .into_iter()
                        .for_each(|i| {
                            self.values
                                .entry(i)
                                .and_modify(|(v, c)| {
                                    *v += V::one();
                                    *c += 1;
                                })
                                .or_insert((V::one(), 1));
                        });
                }
            }
        }
    }

    fn get_feature_ids(&self) -> Vec<String> {
        self.regions
            .iter()
            .map(|x| x.to_genomic_range().pretty_show())
            .collect()
    }

    fn get_values(&self) -> Vec<(usize, Self::Value)> {
        self.values.iter().map(|(k, v)| (*k, v.0)).collect()
    }

    fn get_values_and_counts(&self) -> impl Iterator<Item = (usize, (Self::Value, usize))> {
        self.values
            .iter().map(|(k, v)| (*k, (v.0, v.1)))
    }
}

/// `TranscriptCount` is a struct that represents the count of genomic features at the transcript level.
/// It holds a `SparseCoverage` counter and a reference to `Promoters`.
#[derive(Clone)]
pub struct TranscriptCount<'a> {
    counter: RegionCounter<'a, u32>,
    promoters: &'a Promoters,
}

impl<'a> TranscriptCount<'a> {
    pub fn new(promoters: &'a Promoters) -> Self {
        Self {
            counter: RegionCounter::new(&promoters.regions),
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

/// `GeneCount` is a struct that represents the count of genomic features at the gene level.
/// It holds a `TranscriptCount` counter and a map from gene names to their indices.
#[derive(Clone)]
pub struct GeneCount<'a> {
    counter: TranscriptCount<'a>,
    gene_name_to_idx: IndexMap<&'a str, usize>,
}

/// Implementation of `GeneCount`
impl<'a> GeneCount<'a> {
    pub fn new(counter: TranscriptCount<'a>) -> Self {
        let gene_name_to_idx: IndexMap<_, _> = counter
            .promoters
            .transcripts
            .iter()
            .map(|x| x.gene_name.as_str())
            .unique()
            .enumerate()
            .map(|(a, b)| (b, a))
            .collect();
        Self {
            counter,
            gene_name_to_idx,
        }
    }
}

/// Implementations of `FeatureCounter` trait for `TranscriptCount` and `GeneCount` structs.
impl FeatureCounter for TranscriptCount<'_> {
    type Value = u32;

    fn reset(&mut self) {
        self.counter.reset();
    }

    fn insert<B: BEDLike, N: ToPrimitive + Copy>(&mut self, tag: &B, count: N) {
        self.counter
            .insert(tag, <u32 as NumCast>::from(count).unwrap());
    }

    fn insert_fragment(&mut self, tag: &Fragment, strategy: &CountingStrategy) {
        self.counter.insert_fragment(tag, strategy);
    }

    fn get_feature_ids(&self) -> Vec<String> {
        self.promoters
            .transcripts
            .iter()
            .map(|x| x.transcript_id.clone())
            .collect()
    }

    fn get_values(&self) -> Vec<(usize, Self::Value)> {
        self.counter.get_values()
    }

    fn get_values_and_counts(&self) -> impl Iterator<Item = (usize, (Self::Value, usize))> {
        self.counter.get_values_and_counts()
    }
}

impl FeatureCounter for GeneCount<'_> {
    type Value = u32;

    fn reset(&mut self) {
        self.counter.reset();
    }

    fn insert<B: BEDLike, N: ToPrimitive + Copy>(&mut self, tag: &B, count: N) {
        self.counter
            .insert(tag, <u32 as NumCast>::from(count).unwrap());
    }

    fn insert_fragment(&mut self, tag: &Fragment, strategy: &CountingStrategy) {
        self.counter.insert_fragment(tag, strategy);
    }

    fn get_feature_ids(&self) -> Vec<String> {
        self.gene_name_to_idx
            .keys()
            .map(|x| x.to_string())
            .collect()
    }

    fn get_values(&self) -> Vec<(usize, Self::Value)> {
        let mut counts = BTreeMap::new();
        self.counter.get_values().into_iter().for_each(|(k, v)| {
            let idx = *self
                .gene_name_to_idx
                .get(self.counter.promoters.transcripts[k].gene_name.as_str())
                .unwrap();
            let current_v = counts.entry(idx).or_insert(v);
            if *current_v < v {
                *current_v = v
            }
        });
        counts.into_iter().collect()
    }

    fn get_values_and_counts(&self) -> impl Iterator<Item = (usize, (Self::Value, usize))> {
        todo!();
        self.counter.get_values_and_counts()
    }
}
