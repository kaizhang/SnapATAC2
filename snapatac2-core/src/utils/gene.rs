use crate::utils::FeatureCounter;

use noodles_gff::record::Strand;
use noodles_gff as gff;
use std::io::BufRead;
use std::collections::HashMap;
use std::collections::{BTreeMap, HashSet};
use indexmap::map::IndexMap;
use bed_utils::bed::{
    GenomicRange, BEDLike, tree::GenomeRegions,
    tree::{SparseCoverage},
};

/// Position is 0-based.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Transcript {
    pub transcript_name: Option<String>,
    pub transcript_id: String,
    pub gene_name: String,
    pub gene_id: String,
    pub is_coding: Option<bool>,
    pub chrom: String,
    pub left: i32,
    pub right: i32,
    pub strand: Strand,
}

impl Transcript {
    pub fn get_tss(&self) -> Option<i32> {
        match self.strand {
            Strand::Forward => Some(self.left),
            Strand::Reverse => Some(self.right),
            _ => None,
        }
    }
}

pub fn read_transcripts<R>(input: R) -> Vec<Transcript>
where
    R: BufRead, 
{
    gff::Reader::new(input).records().flat_map(|r| {
        let record = r.unwrap();
        if record.ty() == "transcript" {
            let err_msg = |x: &str| -> String {
                format!("failed to find '{}' in record: {}", x, record)
            };
            let left = record.start();
            let right = record.end();
            let attributes: HashMap<&str, &str> = record.attributes().iter()
                .map(|x| (x.key(), x.value())).collect();
            Some(Transcript {
                transcript_name: attributes.get("transcript_name")
                        .map(|x| x.to_string()),
                transcript_id: attributes.get("transcript_id")
                    .expect(&err_msg("transcript_id"))
                    .to_string(),
                gene_name: attributes.get("gene_name")
                    .expect(&err_msg("gene_name"))
                    .to_string(),
                gene_id: attributes.get("gene_id")
                    .expect(&err_msg("gene_id"))
                    .to_string(),
                is_coding: attributes.get("transcript_type")
                    .map(|x| *x == "protein_coding"),
                chrom: record.reference_sequence_name().to_string(),
                left, right, strand: record.strand(),
            })
        } else {
            None
        }
    }).collect()
}

pub struct Promoters {
    pub regions: GenomeRegions<GenomicRange>,
    pub transcripts: Vec<Transcript>,
}

impl Promoters {
    pub fn new(
        transcripts: Vec<Transcript>,
        upstream: u64,
        downstream: u64,
        include_gene_body: bool,
    ) -> Self
    {
        let regions = transcripts.iter().map(|transcript| {
            let (start, end) = match transcript.strand {
                Strand::Forward => (
                    (transcript.left as u64).saturating_sub(upstream),
                    downstream + (if include_gene_body { transcript.right } else { transcript.left } as u64)
                ),
                Strand::Reverse => (
                    (if include_gene_body { transcript.left } else { transcript.right} as u64).saturating_sub(downstream),
                    (transcript.right as u64) + upstream
                ),
                _ => panic!("Miss strand information for {}", transcript.transcript_id),
            };
            GenomicRange::new(transcript.chrom.clone(), start, end)
        }).collect();
        Promoters { regions, transcripts }
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
        self.promoters.transcripts.iter().map(|x| x.gene_name.clone()).collect()
    }
}

impl FeatureCounter for TranscriptCount<'_> {
    type Value = u32;

    fn reset(&mut self) { self.counter.reset(); }

    fn insert<B: BEDLike>(&mut self, tag: &B, count: u32) { self.counter.insert(tag, count); }

    fn get_feature_ids(&self) -> Vec<String> {
        self.promoters.transcripts.iter().map(|x| x.transcript_id.clone()).collect()
    }

    fn get_counts(&self) -> Vec<(usize, Self::Value)> {
        self.counter.get_counts()
    }
}

#[derive(Clone)]
pub struct GeneCount<'a> {
    counter: TranscriptCount<'a>,
    gene_id_to_idx: IndexMap<&'a str, usize>,
    gene_names: Vec<&'a str>,
}

impl<'a> GeneCount<'a> {
    pub fn new(counter: TranscriptCount<'a>) -> Self {
        let gene_id_to_idx: IndexMap<_, _> = counter.promoters.transcripts.iter()
            .map(|x| x.gene_id.as_str()).collect::<HashSet<_>>().into_iter()
            .enumerate().map(|(a,b)| (b,a)).collect();
        let gene_id_to_name: HashMap<_, _> = counter.promoters.transcripts.iter()
            .map(|x| (x.gene_id.as_str(), x.gene_name.as_str())).collect();
        let gene_names = gene_id_to_idx.keys()
            .map(|k| *gene_id_to_name.get(k).unwrap()).collect();
        Self { counter, gene_id_to_idx, gene_names }
    }

    pub fn gene_names(&self) -> Vec<String> {
        self.gene_names.iter().map(|x| x.to_string()).collect()
    }
}

impl FeatureCounter for GeneCount<'_> {
    type Value = u32;

    fn reset(&mut self) { self.counter.reset(); }

    fn insert<B: BEDLike>(&mut self, tag: &B, count: u32) { self.counter.insert(tag, count); }

    fn get_feature_ids(&self) -> Vec<String> {
        self.gene_id_to_idx.keys().map(|x| x.to_string()).collect()
    }

    fn get_counts(&self) -> Vec<(usize, Self::Value)> {
        let mut counts = BTreeMap::new();
        self.counter.get_counts().into_iter().for_each(|(k, v)| {
            let idx = *self.gene_id_to_idx.get(
                self.counter.promoters.transcripts[k].gene_id.as_str()
            ).unwrap();
            let current_v = counts.entry(idx).or_insert(v);
            if *current_v < v { *current_v = v }
        });
        counts.into_iter().collect()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_transcript() {
        let input = "chr1\tHAVANA\tgene\t11869\t14409\t.\t+\t.\tgene_id=ENSG00000223972.5;gene_type=transcribed_unprocessed_pseudogene;gene_name=DDX11L1;level=2;hgnc_id=HGNC:37102;havana_gene=OTTHUMG00000000961.2\n\
                     chr1\tHAVANA\ttranscript\t11869\t14409\t.\t+\t.\tgene_id=ENSG00000223972.5;transcript_id=ENST00000456328.2;gene_type=transcribed_unprocessed_pseudogene;gene_name=DDX11L1;transcript_type=processed_transcript;transcript_name=DDX11L1-202;level=2;transcript_support_level=1\n\
                     chr1\tHAVANA\texon\t11869\t12227\t.\t+\t.\tgene_id=ENSG00000223972.5;transcript_id=ENST00000456328.2;gene_type=transcribed_unprocessed_pseudogene;gene_name=DDX11L1;transcript_type=processed_transcript;transcript_name=DDX11L1-202;exon_number=1\n\
                     chr1\tHAVANA\texon\t12613\t12721\t.\t+\t.\tgene_id=ENSG00000223972.5;transcript_id=ENST00000456328.2;gene_type=transcribed_unprocessed_pseudogene;gene_name=DDX11L1;transcript_type=processed_transcript;transcript_name=DDX11L1-202;exon_number=2\n\
                     chr1\tHAVANA\texon\t13221\t14409\t.\t+\t.\tgene_id=ENSG00000223972.5;transcript_id=ENST00000456328.2;gene_type=transcribed_unprocessed_pseudogene;gene_name=DDX11L1;transcript_type=processed_transcript;transcript_name=DDX11L1-202;exon_number=3";
        let expected = Transcript {
            transcript_name: Some("DDX11L1-202".to_string()),
            transcript_id: "ENST00000456328.2".to_string(),
            gene_name: "DDX11L1".to_string(),
            gene_id: "ENSG00000223972.5".to_string(),
            is_coding: Some(false),
            chrom: "chr1".to_string(),
            left: 11869,
            right: 14409,
            strand: Strand::Forward,
        };
        assert_eq!(read_transcripts(input.as_bytes())[0], expected)
    }

}