use crate::utils::{
    Insertions,
    Barcoded,
    FeatureCounter,
    gene::Transcript,
};
use crate::peak_matrix::create_feat_matrix;

use std::collections::BTreeMap;
use std::collections::HashSet;
use std::collections::HashMap;
use hdf5::{File, Result};
use noodles_gff::record::Strand;
use bed_utils::bed::{
    GenomicRange, BEDLike, tree::GenomeRegions,
    tree::{SparseCoverage},
};

pub struct Promoters {
    regions: GenomeRegions<GenomicRange>,
    transcript_ids: Vec<String>,
    gene_names: Vec<String>,
    unique_gene_index: HashMap<String, usize>,
}

impl Promoters {
    pub fn new(transcripts: Vec<Transcript>, half_size: u64) -> Self {
        let mut transcript_ids = Vec::new();
        let mut gene_names = Vec::new();
        let regions = transcripts.into_iter().map(|transcript| {
            let (start, end) = match transcript.strand {
                Strand::Forward => ((transcript.left as u64).saturating_sub(half_size), transcript.right as u64),
                Strand::Reverse => (transcript.left as u64, (transcript.right as u64) + half_size),
                _ => panic!("Miss strand information for {}", transcript.transcript_id),
            };
            transcript_ids.push(transcript.transcript_id);
            gene_names.push(transcript.gene_name);
            GenomicRange::new(transcript.chrom, start, end)
        }).collect();
        let unique_gene_index = gene_names.clone().into_iter()
            .collect::<HashSet<_>>().into_iter()
            .enumerate().map(|(a,b)| (b,a)).collect();
        Promoters { regions, transcript_ids, gene_names, unique_gene_index }
    }
}

#[derive(Clone)]
struct PromoterCoverage<'a> {
    counter: SparseCoverage<'a, GenomicRange, u32>,
    promoters: &'a Promoters,
}

impl<'a> PromoterCoverage<'a> {
    pub fn new(promoters: &'a Promoters) -> Self {
        Self {
            counter: SparseCoverage::new(&promoters.regions),
            promoters,
        }
    }
}

impl FeatureCounter for PromoterCoverage<'_> {
    type Value = u32;

    fn reset(&mut self) { self.counter.reset(); }

    fn insert<B: BEDLike>(&mut self, tag: &B, count: u32) { self.counter.insert(tag, count); }

    fn get_feature_ids(&self) -> Vec<String> {
        let mut names: Vec<(&String, &usize)> = self.promoters.unique_gene_index.iter().collect();
        names.sort_by(|a, b| a.1.cmp(b.1));
        names.iter().map(|(a, _)| (*a).clone()).collect()
    }

    fn get_counts(&self) -> Vec<(usize, Self::Value)> {
        let mut counts = BTreeMap::new();
        self.counter.get_coverage().iter().for_each(|(k, v)| {
            let name = &self.promoters.gene_names[*k];
            let idx = *self.promoters.unique_gene_index.get(name).unwrap();
            let current_v = counts.entry(idx).or_insert(*v);
            if *current_v < *v { *current_v = *v }
        });
        counts.into_iter().collect()
    }
}

pub fn create_gene_matrix<'a, I, D>(
    file: &File,
    fragments: I,
    transcripts: Vec<Transcript>,
    ) -> Result<()>
where
    I: Iterator<Item = D>,
    D: Into<Insertions> + Send,
{
    let promoters = Promoters::new(transcripts, 2000);
    let feature_counter: PromoterCoverage<'_> = PromoterCoverage::new(&promoters);
    create_feat_matrix(file, fragments, feature_counter)
}