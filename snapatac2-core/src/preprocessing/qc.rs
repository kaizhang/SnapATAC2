use std::{io::{Read, BufRead, BufReader}, ops::Div, collections::HashMap};
use anndata::data::CsrNonCanonical;
use bed_utils::bed::{
    GenomicRange, BEDLike, tree::{GenomeRegions, BedTree, SparseBinnedCoverage},
    ParseError, Strand, BED,
};
use anyhow::Result;
use serde::{Serialize, Deserialize};
use extsort::sorter::Sortable;
use bincode;

pub type CellBarcode = String;

/// Fragments from single-cell ATAC-seq experiment. Each fragment is represented
/// by a genomic coordinate, cell barcode and a integer value.
#[derive(Serialize, Deserialize, Debug)] 
pub struct Fragment {
    pub chrom: String,
    pub start: u64,
    pub end: u64,
    pub barcode: CellBarcode,
    pub count: u32,
    pub strand: Option<Strand>,
}

impl Sortable for Fragment {
    fn encode<W: std::io::Write>(&self, writer: &mut W) {
        bincode::serialize_into(writer, self)
            .unwrap_or_else(|e| panic!("Failed to serialize fragment: {}", e));
    }

    fn decode<R: std::io::Read>(reader: &mut R) -> Option<Self> {
        bincode::deserialize_from(reader).ok()
    }
}

impl Fragment {
    pub fn to_insertions(&self) -> Vec<GenomicRange> {
        match self.strand {
            None => vec![
                GenomicRange::new(self.chrom.clone(), self.start, self.start + 1),
                GenomicRange::new(self.chrom.clone(), self.end - 1, self.end),
            ],
            Some(Strand::Forward) => vec![
                GenomicRange::new(self.chrom.clone(), self.start, self.start + 1)
            ],
            Some(Strand::Reverse) => vec![
                GenomicRange::new(self.chrom.clone(), self.end - 1, self.end)
            ],
        }
    }
}

impl BEDLike for Fragment {
    fn chrom(&self) -> &str { &self.chrom }
    fn set_chrom(&mut self, chrom: &str) -> &mut Self {
        self.chrom = chrom.to_string();
        self
    }
    fn start(&self) -> u64 { self.start }
    fn set_start(&mut self, start: u64) -> &mut Self {
        self.start = start;
        self
    }
    fn end(&self) -> u64 { self.end }
    fn set_end(&mut self, end: u64) -> &mut Self {
        self.end = end;
        self
    }
    fn name(&self) -> Option<&str> { None }
    fn score(&self) -> Option<bed_utils::bed::Score> { None }
    fn strand(&self) -> Option<Strand> { None }
}

impl std::str::FromStr for Fragment {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut fields = s.split('\t');
        let chrom = fields.next().ok_or(ParseError::MissingReferenceSequenceName)?.to_string();
        let start = fields.next().ok_or(ParseError::MissingStartPosition)
            .and_then(|s| lexical::parse(s).map_err(ParseError::InvalidStartPosition))?;
        let end = fields.next().ok_or(ParseError::MissingEndPosition)
            .and_then(|s| lexical::parse(s).map_err(ParseError::InvalidEndPosition))?;
        let barcode = fields.next().ok_or(ParseError::MissingName)
            .map(|s| s.into())?;
        let count = fields.next().map_or(Ok(1), |s| if s == "." {
            Ok(1)
        } else {
            lexical::parse(s).map_err(ParseError::InvalidStartPosition)
        })?;
        let strand = fields.next().map_or(Ok(None), |s| if s == "." {
            Ok(None)
        } else {
            s.parse().map(Some).map_err(ParseError::InvalidStrand)
        })?;
        Ok(Fragment { chrom, start, end, barcode, count, strand })
    }
}

/// Chromatin interactions from single-cell Hi-C experiment. 
#[derive(Serialize, Deserialize, Debug)] 
pub struct Contact {
    pub chrom1: String,
    pub start1: u64,
    pub chrom2: String,
    pub start2: u64,
    pub barcode: CellBarcode,
    pub count: u32,
}

impl Sortable for Contact {
    fn encode<W: std::io::Write>(&self, writer: &mut W) {
        bincode::serialize_into(writer, self)
            .unwrap_or_else(|e| panic!("Failed to serialize fragment: {}", e));
    }

    fn decode<R: std::io::Read>(reader: &mut R) -> Option<Self> {
        bincode::deserialize_from(reader).ok()
    }
}

impl std::str::FromStr for Contact {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut fields = s.split('\t');
        let barcode = fields.next().ok_or(ParseError::MissingName)
            .map(|s| s.into())?;
        let chrom1 = fields.next().ok_or(ParseError::MissingReferenceSequenceName)?.to_string();
        let start1 = fields.next().ok_or(ParseError::MissingStartPosition)
            .and_then(|s| lexical::parse(s).map_err(ParseError::InvalidStartPosition))?;
        let chrom2 = fields.next().ok_or(ParseError::MissingReferenceSequenceName)?.to_string();
        let start2 = fields.next().ok_or(ParseError::MissingStartPosition)
            .and_then(|s| lexical::parse(s).map_err(ParseError::InvalidStartPosition))?;
        let count = fields.next().map_or(Ok(1), |s| if s == "." {
            Ok(1)
        } else {
            lexical::parse(s).map_err(ParseError::InvalidStartPosition)
        })?;
        Ok(Contact { barcode, chrom1, start1, chrom2, start2, count })
    }
}


#[derive(Clone, Debug, PartialEq)]
pub struct QualityControl {
    pub tss_enrichment: f64,
    pub num_unique_fragment: u64,
    pub frac_mitochondrial: f64,
    pub frac_duplicated: f64,
}

pub(crate) struct FragmentSummary<'a> {
    promoter_insertion_count: [u64; 4001],
    pub(crate) num_unique_fragment: u64,
    num_total_fragment: u64, 
    num_mitochondrial : u64,
    promoter: &'a BedTree<bool>,
}

impl<'a> FragmentSummary<'a> {
    pub(crate) fn new(promoter: &'a BedTree<bool>) -> Self {
        FragmentSummary {
            promoter_insertion_count: [0; 4001],
            num_unique_fragment: 0,
            num_total_fragment: 0,
            num_mitochondrial: 0,
            promoter,
        }
    }

    pub(crate) fn update(&mut self, fragment: &Fragment) {
        self.num_total_fragment += fragment.count as u64;
        match fragment.chrom.as_str() {
            "chrM" | "M" => self.num_mitochondrial += 1,
            _ => {
                self.num_unique_fragment += 1;
                for ins in fragment.to_insertions() {
                    for (entry, data) in self.promoter.find(&ins) {
                        let pos: u64 =
                            if *data {
                                ins.start() - entry.start()
                            } else {
                                4000 - (entry.end() - 1 - ins.start())
                            };
                        self.promoter_insertion_count[pos as usize] += 1;
                    }
                }
            }
        }
    }

    pub(crate) fn get_qc(self) -> QualityControl {
        let bg_count: f64 =
            ( self.promoter_insertion_count[ .. 100].iter().sum::<u64>() +
            self.promoter_insertion_count[3901 .. 4001].iter().sum::<u64>() ) as f64 /
            200.0 + 0.1;
        let tss_enrichment = moving_average(5, &self.promoter_insertion_count)
            .max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap().div(bg_count);
        let frac_duplicated = 1.0 -
            (self.num_unique_fragment + self.num_mitochondrial) as f64 /
            self.num_total_fragment as f64;
        let frac_mitochondrial = self.num_mitochondrial as f64 /
            (self.num_unique_fragment + self.num_mitochondrial) as f64;
        QualityControl {
            tss_enrichment,
            num_unique_fragment: self.num_unique_fragment,
            frac_mitochondrial,
            frac_duplicated,
        }
    }
}

fn moving_average(half_window: usize, arr: &[u64]) -> impl Iterator<Item = f64> + '_ {
    let n = arr.len();
    (0 .. n).map(move |i| {
        let r = i.saturating_sub(half_window) .. std::cmp::min(i + half_window + 1, n);
        let l = r.len() as f64;
        arr[r].iter().sum::<u64>() as f64 / l
    })
}

/// Read tss from a gtf or gff file
pub fn read_tss<R: Read>(file: R) -> impl Iterator<Item = (String, u64, bool)> {
    let reader = BufReader::new(file);
    reader.lines().filter_map(|line| {
        let chr_idx: usize = 0;
        let type_idx: usize = 2;
        let start_idx: usize = 3;
        let end_idx: usize = 4;
        let strand_idx: usize = 6;
        let l = line.unwrap();
        if l.as_bytes()[0] as char == '#' {
            return None;
        }
        let elements: Vec<&str> = l.split('\t').collect();
        if elements[type_idx] == "transcript" {
            let chr = elements[chr_idx].to_string();
            let is_fwd = elements[strand_idx] != "-";
            let tss: u64 = 
                if is_fwd {
                    elements[start_idx].parse::<u64>().unwrap() - 1
                } else {
                    elements[end_idx].parse::<u64>().unwrap() - 1
                };
            Some((chr, tss, is_fwd))
        } else {
            None
        }
    })
}


pub fn make_promoter_map<I: Iterator<Item = (String, u64, bool)>>(iter: I) -> BedTree<bool> {
    iter
        .map( |(chr, tss, is_fwd)| {
            let b = GenomicRange::new(chr, tss.saturating_sub(2000), tss + 2001);
            (b, is_fwd)
        }).collect()
}

pub fn compute_qc_count<B>(
    fragments: Vec<Fragment>,
    promoter: &BedTree<bool>,
    regions: &GenomeRegions<B>,
    min_n_fragment: u64,
    min_tsse: f64,
    ) -> Option<(QualityControl, Vec<(usize, u8)>)>
where
    B: BEDLike,
{

    let mut summary = FragmentSummary::new(promoter);
    fragments.iter().for_each(|frag| summary.update(frag));
    let qc = summary.get_qc();
    if qc.num_unique_fragment < min_n_fragment || qc.tss_enrichment < min_tsse {
        None
    } else {
        let mut binned_coverage = SparseBinnedCoverage::new(regions, 1);
        fragments.iter().for_each(|fragment| {
            fragment.to_insertions().into_iter().for_each(|x| binned_coverage.insert(&x, 1));
        });
        let count: Vec<(usize, u8)> = binned_coverage.get_coverage()
            .iter().map(|(k, v)| (*k, *v)).collect();
        Some((qc, count))
    }
}

/// barcode counting.
pub fn get_barcode_count<I>(fragments: I) -> HashMap<String, u64>
where
    I: Iterator<Item = Fragment>,
{
    let mut barcodes = HashMap::new();
    fragments.for_each(|frag| {
        let key = frag.barcode.clone();
        *barcodes.entry(key).or_insert(0) += 1;
    });
    barcodes
}

/// Compute the fragment size distribution.
/// The result is stored in a vector where each element represents the number of fragments
/// and the index represents the fragment length. The first posision of the vector is
/// reserved for fragments with length larger than the maximum length.
pub fn fragment_size_distribution<I>(data: I, max_size: usize) -> Vec<usize>
  where
    I: Iterator<Item = CsrNonCanonical<u32>>,
{
    let mut size_dist = vec![0; max_size+1];
    data.for_each(|csr| {
        let values = csr.values();
        values.iter().for_each(|&v| {
            let v = v as usize;
            if v <= max_size {
                size_dist[v] += 1;
            } else {
                size_dist[0] += 1;
            }
        });
    });
    size_dist
}

/// Count the fraction of the records in the given regions.
pub fn fraction_in_region<'a, I, D>(
    iter: I, regions: &'a Vec<BedTree<D>>,
) -> impl Iterator<Item = (Vec<Vec<f64>>, usize, usize)> + 'a
where
    I: Iterator<Item = (Vec<Vec<BED<6>>>, usize, usize)> + 'a,
{
    let k = regions.len();
    iter.map(move |(beds, start, end)| {
        let frac = beds.into_iter().map(|xs| {
            let sum = xs.len() as f64;
            let mut counts = vec![0.0; k];
            xs.into_iter().for_each(|x|
                regions.iter().enumerate().for_each(|(i, r)| {
                    if r.is_overlapped(&x) {
                        counts[i] += 1.0;
                    }
                })
            );
            counts.iter_mut().for_each(|x| *x /= sum);
            counts
        }).collect::<Vec<_>>();
        (frac, start, end)
    })
}