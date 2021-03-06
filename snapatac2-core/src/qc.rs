use anndata_rs::{
    anndata::AnnData,
    anndata_trait::{DataIO, DataPartialIO},
    iterator::CsrIterator,
};
use polars::prelude::{NamedFrom, DataFrame, Series};

use std::{
    io, io::prelude::*, io::BufReader, ops::Div,
    collections::{HashSet, HashMap},
};
use bed_utils::bed::{
    ParseError, GenomicRange, BEDLike,
    tree::{GenomeRegions, BedTree, SparseBinnedCoverage}, Strand,
};
use itertools::Itertools;
use anyhow::Result;
use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelIterator;

pub type CellBarcode = String;

/// Fragments from single-cell ATAC-seq experiment. Each fragment is represented
/// by a genomic coordinate, cell barcode and a integer value.
pub struct Fragment {
    pub chrom: String,
    pub start: u64,
    pub end: u64,
    pub barcode: CellBarcode,
    pub count: u32,
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
        Ok(Fragment { chrom, start, end, barcode, count })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct QualityControl {
    pub tss_enrichment: f64,
    pub num_unique_fragment: u64,
    pub frac_mitochondrial: f64,
    pub frac_duplicated: f64,
}

/*
// Store length distribution in an array,
// TODO: use [u64; N+1] when const generic arithmetic is implemented in rust
pub struct FragmentSizeDistribution<const N: usize> {
    counts: [u64; N]),

}

impl<const N: usize> FragmentSizeDistribution<N> {
    pub fn new() -> Self { Self([0; N]) }

    /// Get the frequency of fragment size
    pub fn get(&self, i: usize) -> u64 {
        if i <= N {
            self.0[i]
        } else {
            self.0[N]
        }
    }

    pub fn add<B: BEDLike>(&mut self, bed: B) {
        let i = bed.len() as usize;
        if i <= N {
            self.0[i - 1] += 1;
        } else {
            self.0[N as usize] += 1;
        }
    }
}

impl<const N: u32> FromIterator for FragmentSizeDistribution<N> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = B>,
        B: BEDLike,
    {
        let mut distribution = FragmentSizeDistribution::new();
        iter.for_each(|x| { distribution.update(x); });
        distribution
    }

}
*/

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
                for ins in get_insertions(fragment) {
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

fn qc_to_df(
    cells: Vec<CellBarcode>,
    qc: Vec<QualityControl>,
) -> DataFrame {
    DataFrame::new(vec![
        Series::new("Cell", cells),
        Series::new(
            "tsse", 
            qc.iter().map(|x| x.tss_enrichment).collect::<Series>(),
        ),
        Series::new(
            "n_fragment", 
            qc.iter().map(|x| x.num_unique_fragment).collect::<Series>(),
        ),
        Series::new(
            "frac_dup", 
            qc.iter().map(|x| x.frac_duplicated).collect::<Series>(),
        ),
        Series::new(
            "frac_mito", 
            qc.iter().map(|x| x.frac_mitochondrial).collect::<Series>(),
        ),
    ]).unwrap()
}

fn moving_average(half_window: usize, arr: &[u64]) -> impl Iterator<Item = f64> + '_ {
    let n = arr.len();
    (0 .. n).map(move |i| {
        let r = i.saturating_sub(half_window) .. std::cmp::min(i + half_window + 1, n);
        let l = r.len() as f64;
        arr[r].iter().sum::<u64>() as f64 / l
    })
}

/// Read tss from a gtf file
pub fn read_tss<R: Read>(file: R) -> impl Iterator<Item = (String, u64, bool)> {
    let reader = BufReader::new(file);
    let parse_line = |line: io::Result<String>| {
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
    };
    reader.lines().filter_map(parse_line)
}


pub fn make_promoter_map<I: Iterator<Item = (String, u64, bool)>>(iter: I) -> BedTree<bool> {
    iter
        .map( |(chr, tss, is_fwd)| {
            let b = GenomicRange::new(chr, tss.saturating_sub(2000), tss + 2001);
            (b, is_fwd)
        }).collect()
}

fn get_insertions(rec: &Fragment) -> [GenomicRange; 2] {
    [ GenomicRange::new(rec.chrom.clone(), rec.start, rec.start + 1)
    , GenomicRange::new(rec.chrom.clone(), rec.end - 1, rec.end) ]
}

pub fn import_fragments<B, I>(
    anndata: &AnnData,
    fragments: I,
    promoter: &BedTree<bool>,
    regions: &GenomeRegions<B>,
    white_list: Option<&HashSet<String>>,
    min_num_fragment: u64,
    min_tsse: f64,
    fragment_is_sorted_by_name: bool,
    chunk_size: usize,
    ) -> Result<()>
where
    B: BEDLike + Clone + std::marker::Sync,
    I: Iterator<Item = Fragment>,
{
    let num_features = SparseBinnedCoverage::<_, u8>::new(regions, 1).len;
    let mut saved_barcodes = Vec::new();
    let mut qc = Vec::new();

    if fragment_is_sorted_by_name {
        let mut scanned_barcodes = HashSet::new();
        anndata.get_obsm().inner().insert_from_row_iter(
            "insertion",
            CsrIterator {
                iterator: fragments
                .group_by(|x| { x.barcode.clone() }).into_iter()
                .filter(|(key, _)| white_list.map_or(true, |x| x.contains(key)))
                .chunks(chunk_size).into_iter().map(|chunk| {
                    let data: Vec<(String, Vec<Fragment>)> = chunk
                        .map(|(barcode, x)| (barcode, x.collect())).collect();
                    let result: Vec<_> = data.into_par_iter()
                        .map(|(barcode, x)| (
                            barcode,
                            compute_qc_count(x, promoter, regions, min_num_fragment, min_tsse)
                        )).collect();
                    result.into_iter().filter_map(|(barcode, r)| {
                        if !scanned_barcodes.insert(barcode.clone()) {
                            panic!("Please sort fragment file by barcodes");
                        }
                        match r {
                            None => None,
                            Some((q, count)) => {
                                saved_barcodes.push(barcode);
                                qc.push(q);
                                Some(count)
                            },
                        }
                    }).collect::<Vec<_>>()
                }),
                num_cols: num_features,
            }
        )?;
    } else {
        let mut scanned_barcodes = HashMap::new();
        fragments
        .filter(|frag| white_list.map_or(true, |x| x.contains(frag.barcode.as_str())))
        .for_each(|frag| {
            let key = frag.barcode.as_str();
            let ins = get_insertions(&frag);
            match scanned_barcodes.get_mut(key) {
                None => {
                    let mut summary= FragmentSummary::new(promoter);
                    let mut counts = SparseBinnedCoverage::new(regions, 1);
                    summary.update(&frag);
                    counts.insert(&ins[0], 1);
                    counts.insert(&ins[1], 1);
                    scanned_barcodes.insert(key.to_string(), (summary, counts));
                },
                Some((summary, counts)) => {
                    summary.update(&frag);
                    counts.insert(&ins[0], 1);
                    counts.insert(&ins[1], 1);
                }
            }
        });
        let csr_matrix: Box<dyn DataPartialIO> = Box::new(CsrIterator {
            iterator: scanned_barcodes.drain()
            .filter_map(|(barcode, (summary, binned_coverage))| {
                let q = summary.get_qc();
                if q.num_unique_fragment < min_num_fragment || q.tss_enrichment < min_tsse {
                    None
                } else {
                    saved_barcodes.push(barcode);
                    qc.push(q);
                    let count: Vec<(usize, u8)> = binned_coverage.get_coverage()
                        .iter().map(|(k, v)| (*k, *v)).collect();
                    Some(count)
                }
            }),
            num_cols: num_features,
        }.to_csr_matrix());
        anndata.get_obsm().inner().add_data("insertion", &csr_matrix)?;
    }

    let chrom_sizes: Box<dyn DataIO> = Box::new(DataFrame::new(vec![
        Series::new(
            "reference_seq_name",
            regions.regions.iter().map(|x| x.chrom()).collect::<Series>(),
        ),
        Series::new(
            "reference_seq_length",
            regions.regions.iter().map(|x| x.end()).collect::<Series>(),
        ),
    ]).unwrap());
    anndata.get_uns().inner().add_data("reference_sequences", &chrom_sizes)?;
    anndata.set_obs(Some(&qc_to_df(saved_barcodes, qc)))?;
    Ok(())
}

fn compute_qc_count<B>(
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
            let ins = get_insertions(fragment);
            binned_coverage.insert(&ins[0], 1);
            binned_coverage.insert(&ins[1], 1);
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