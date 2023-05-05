//! # Genomic Feature Counter Module
//!
//! This module provides the functionality to count genomic features (such as genes or transcripts) 
//! in genomic data. The primary structures in this module are `TranscriptCount` and `GeneCount`, 
//! both of which implement the `FeatureCounter` trait. The `FeatureCounter` trait provides a 
//! common interface for handling feature counts, including methods for resetting counts, 
//! updating counts, and retrieving feature IDs, names, and counts.
//!
//! `SparseCoverage`, from the bed_utils crate, is used for maintaining counts of genomic features, 
//! and this structure also implements the `FeatureCounter` trait in this module.
//!
//! `TranscriptCount` and `GeneCount` structures also hold a reference to `Promoters`, which 
//! provides additional information about the genomic features being counted.
//!
//! To handle the mapping of gene names to indices, an `IndexMap` is used in the `GeneCount` structure. 
//! This allows for efficient look-up of gene indices by name, which is useful when summarizing counts 
//! at the gene level.
//!
//! The module aims to provide a comprehensive, efficient, and flexible way to handle and manipulate 
//! genomic feature counts in Rust.
use std::collections::BTreeMap;
use indexmap::map::IndexMap;
use bed_utils::bed::{GenomicRange, BEDLike, tree::SparseCoverage};
use itertools::Itertools;
use num::traits::{ToPrimitive, NumCast};

use super::Promoters;

/// `FeatureCounter` is a trait that provides an interface for counting genomic features.
/// Types implementing `FeatureCounter` can store feature counts and provide several 
/// methods for manipulating and retrieving those counts.
pub trait FeatureCounter {
    type Value;

    /// Returns the total number of distinct features counted.
    fn num_features(&self) -> usize { self.get_feature_ids().len() }

    /// Resets the counter for all features.
    fn reset(&mut self);

    /// Updates the counter according to the given region and count.
    fn insert<B: BEDLike, N: ToPrimitive + Copy>(&mut self, tag: &B, count: N);

    /// Returns a vector of feature ids.
    fn get_feature_ids(&self) -> Vec<String>;

    /// Returns a vector of feature names if available.
    fn get_feature_names(&self) -> Option<Vec<String>> { None }

    /// Returns a vector of tuples, each containing a feature's index and its count.
    fn get_counts(&self) -> Vec<(usize, Self::Value)>;
}

/// Implementation of `FeatureCounter` trait for `SparseCoverage` struct.
/// `SparseCoverage` represents a sparse coverage map for genomic data.
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

/// `TranscriptCount` is a struct that represents the count of genomic features at the transcript level.
/// It holds a `SparseCoverage` counter and a reference to `Promoters`.
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