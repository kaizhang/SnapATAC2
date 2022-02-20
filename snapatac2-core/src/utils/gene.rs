use noodles_gff::record::Strand;
use noodles_gff as gff;
use std::io::BufRead;
use std::collections::HashMap;
use std::collections::BTreeSet;

/// Position is 0-based.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Transcript {
    pub transcript_name: String,
    pub transcript_id: String,
    pub gene_name: String,
    pub gene_id: String,
    pub is_coding: Option<bool>,
    pub chrom: String,
    pub left: i32,
    pub right: i32,
    pub strand: Strand,
    pub exons: Vec<(i32, i32)>,
    pub utr: Vec<(i32, i32)>,
}

pub fn read_transcripts<R>(input: R) -> HashMap<String, Transcript>
where
    R: BufRead, 
{
    let mut all_transcripts = HashMap::new();
    let mut exons = HashMap::new();
    let mut utrs = HashMap::new();
    gff::Reader::new(input).records().for_each(|r| {
        let record = r.unwrap();
        let attributes: HashMap<&str, &str> = record.attributes().iter()
            .map(|x| (x.key(), x.value())).collect();
        let left = record.start();
        let right = record.end();
        match record.ty() {
            "transcript" => {
                let transcript_id = attributes.get("transcript_id").unwrap().to_string();
                let transcript = Transcript {
                    transcript_name: attributes.get("transcript_name").unwrap().to_string(),
                    transcript_id: transcript_id.clone(),
                    gene_name: attributes.get("gene_name").unwrap().to_string(),
                    gene_id: attributes.get("gene_id").unwrap().to_string(),
                    is_coding: attributes.get("transcript_type").map(|x| *x == "protein_coding"),
                    chrom: record.reference_sequence_name().to_string(),
                    left,
                    right,
                    strand: record.strand(),
                    exons: Vec::new(),
                    utr: Vec::new(),
                };
                all_transcripts.insert(transcript_id, transcript);
            },
            "exon" => {
                let transcript_id = attributes.get("transcript_id").unwrap().to_string();
                let set = exons.entry(transcript_id).or_insert(BTreeSet::new());
                set.insert((left, right));
            },
            "utr" => {
                let transcript_id = attributes.get("transcript_id").unwrap().to_string();
                let set = utrs.entry(transcript_id).or_insert(BTreeSet::new());
                set.insert((left, right));
            },
            "gene" => {},
            _ => {},
        }
    });
    exons.drain().for_each(|(k, v)| {
        if let Some(x) = all_transcripts.get_mut(&k) {
            x.exons = v.into_iter().collect();
        }
    });
    utrs.drain().for_each(|(k, v)| {
        if let Some(x) = all_transcripts.get_mut(&k) {
            x.utr = v.into_iter().collect();
        }
    });
    all_transcripts
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
            transcript_name: "DDX11L1-202".to_string(),
            transcript_id: "ENST00000456328.2".to_string(),
            gene_name: "DDX11L1".to_string(),
            gene_id: "ENSG00000223972.5".to_string(),
            is_coding: Some(false),
            chrom: "chr1".to_string(),
            left: 11869,
            right: 14409,
            strand: Strand::Forward,
            exons: vec![(11869, 12227), (12613, 12721), (13221, 14409)],
            utr: Vec::new(),
        };
        assert_eq!(read_transcripts(input.as_bytes()).into_values().collect::<Vec<_>>()[0], expected)
    }

}