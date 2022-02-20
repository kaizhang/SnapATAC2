use crate::utils::{
    Fragment,
    Insertions,
    Barcoded,
    FeatureCounter,
    gene::Transcript,
    anndata::{AnnData, SparseRowWriter, create_obs, create_var},
};
use crate::peak_matrix::create_feat_matrix;

use std::collections::HashSet;
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
}

impl Promoters {
    pub fn new(transcripts: Vec<Transcript>, half_size: u64) -> Self {
        let mut transcript_ids = Vec::new();
        let mut gene_names = Vec::new();
        let regions = transcripts.into_iter().map(|transcript| {
            let tss = match transcript.strand {
                Strand::Forward => transcript.left as u64,
                Strand::Reverse => transcript.right as u64,
                _ => panic!("Miss strand information for {}", transcript.transcript_id),
            };
            transcript_ids.push(transcript.transcript_id);
            gene_names.push(transcript.gene_name);
            GenomicRange::new(transcript.chrom, tss.saturating_sub(half_size), tss + half_size + 1)
        }).collect();
        Promoters { regions, transcript_ids, gene_names }
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
        self.promoters.transcript_ids.clone()
    }

    fn get_counts(&self) -> Vec<(usize, Self::Value)> {
        self.counter.get_coverage().iter().map(|(k, v)| (*k, *v)).collect()
    }
}

pub fn create_gene_matrix<'a, I, D>(
    file: File,
    fragments: I,
    transcripts: Vec<Transcript>,
    white_list: Option<&HashSet<String>>,
    fragment_is_sorted_by_name: bool,
    ) -> Result<()>
where
    I: Iterator<Item = D>,
    D: Into<Insertions> + Barcoded + Send,
{
    let promoters = Promoters::new(transcripts, 2000);
    let feature_counter: PromoterCoverage<'_> = PromoterCoverage::new(&promoters);
    create_feat_matrix(file, fragments, feature_counter, white_list, fragment_is_sorted_by_name)
}