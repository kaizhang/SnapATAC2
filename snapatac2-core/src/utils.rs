pub mod hdf5;
pub mod anndata;

use std::ops::Range;
use std::collections::HashMap;
use itertools::Itertools;
use std::hash::Hash;
use num::Integer;

use bed_utils::bed::{split_by_len, GenomicRange, BEDLike, tree::GenomeRegions};

pub struct SparseBinnedCoverage<'a, B> {
    pub len: usize,
    pub bin_size: u64,
    pub consumed_tags: u64,
    genome_regions: &'a GenomeRegions<B>,
    accu_size: Vec<usize>,
    coverage: HashMap<usize, u64>,
}

impl <'a, B: BEDLike> SparseBinnedCoverage<'a, B> {
    pub fn new(genome_regions: &'a GenomeRegions<B>, bin_size: u64) -> Self {
        let mut len = 0;
        let accu_size = genome_regions.regions.iter().map(|x| {
            let n = x.len().div_ceil(&bin_size) as usize;
            let output = len;
            len += n;
            output
        }).collect();
        Self {
            len, bin_size, consumed_tags: 0, genome_regions, accu_size,
            coverage: HashMap::new()
        }
    }

    pub fn reset(&mut self) {
        self.consumed_tags = 0;
        self.coverage = HashMap::new();
    }

    pub fn add<D>(&mut self, tag: &D)
    where
        D: BEDLike,
    {
        self.consumed_tags += 1;
        self.genome_regions.indices.find(tag).for_each(|(region, out_idx)| {
            let i = tag.start().saturating_sub(region.start()).div_floor(&self.bin_size);
            let j = (tag.end() - 1 - region.start())
                .min(region.len() - 1).div_floor(&self.bin_size);
            let n = self.accu_size[*out_idx];
            (i..=j).for_each(|in_idx| {
                let counter = self.coverage.entry(n + in_idx as usize).or_insert(0);
                *counter += 1;
            });
        });
    }

    pub fn get_regions(&'a self) -> impl Iterator<Item = impl Iterator<Item = B>> + 'a
    where
        B: Clone
    {
        self.genome_regions.regions.iter()
            .map(|x| split_by_len(x, self.bin_size))
    }

    pub fn get_coverage(&self) -> &HashMap<usize, u64> { &self.coverage }
}

