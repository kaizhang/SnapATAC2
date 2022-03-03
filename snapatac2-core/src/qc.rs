use crate::utils::{
    Insertions,
    anndata::{AnnDataElement, SparseRowIter, StrVec, AnnDataIO, create_obs, write_csr_rows},
};

use std::io;
use std::io::prelude::*;
use std::io::BufReader;                                                                                                                                           
use bed_utils::bed::{
    GenomicRange, BED, BEDLike, io::Reader,
    tree::{GenomeRegions, BedTree, SparseBinnedCoverage},
};
use itertools::{Itertools, GroupBy};
use std::ops::Div;
use std::collections::HashSet;
use std::collections::HashMap;
use hdf5::{File, Group, Result};
use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelIterator;
use nalgebra_sparse::csr;

pub type CellBarcode = String;

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

    pub(crate) fn update(&mut self, fragment: &BED<5>) {
        self.num_total_fragment += *fragment.score().unwrap() as u64;
        match fragment.chrom() {
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

pub fn get_insertions(rec: &BED<5>) -> [GenomicRange; 2] {
    [ GenomicRange::new(rec.chrom().to_string(), rec.start(), rec.start() + 1)
    , GenomicRange::new(rec.chrom().to_string(), rec.end() - 1, rec.end()) ]
}

/// Read and group fragments according to cell barcodes.
pub fn read_fragments<R>(r: R) -> GroupBy<CellBarcode, impl Iterator<Item = BED<5>>, impl FnMut(&BED<5>) -> CellBarcode>
where
    R: Read,
{
    Reader::new(r, None).into_records().map(Result::unwrap)
        .group_by(|x: &BED<5>| { x.name().unwrap().to_string() })
}

pub fn import_fragments<B, I>(
    file: &File,
    fragments: I,
    promoter: &BedTree<bool>,
    regions: &GenomeRegions<B>,
    white_list: Option<&HashSet<String>>,
    min_num_fragment: u64,
    min_tsse: f64,
    fragment_is_sorted_by_name: bool,
    ) -> Result<()>
where
    B: BEDLike + Clone + std::marker::Sync,
    I: Iterator<Item = BED<5>>,
{
    let num_features = SparseBinnedCoverage::<_, u8>::new(regions, 1).len;
    let mut saved_barcodes = Vec::new();
    let mut qc = Vec::new();
    let obsm = file.create_group("obsm")?;

    let mat = if fragment_is_sorted_by_name {
        let mut scanned_barcodes = HashSet::new();
        write_csr_rows(
            fragments
            .group_by(|x| { x.name().unwrap().to_string() }).into_iter()
            .filter(|(key, _)| white_list.map_or(true, |x| x.contains(key)))
            .chunks(2000).into_iter().map(|chunk| {
                let data: Vec<(String, Vec<BED<5>>)> = chunk
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
            }).flatten(),
            num_features,
            &obsm,
            "base_count",
            "base_count",
            "0.1.0",
        )?
    } else {
        let mut scanned_barcodes = HashMap::new();
        fragments
        .filter(|frag| white_list.map_or(true, |x| x.contains(frag.name().unwrap())))
        .for_each(|frag| {
            let key = frag.name().unwrap();
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
        let csr_rows = scanned_barcodes.drain()
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
            });
        write_csr_rows(csr_rows, num_features, &obsm, "base_count", "base_count", "0.1.0")?
    };

    StrVec(regions.regions.iter().map(|x| x.chrom().to_string()).collect())
        .write(&mat, "reference_seq_name")?;
    regions.regions.iter().map(|x| x.end()).collect::<Vec<_>>()
        .write(&mat, "reference_seq_length")?;
    create_obs(file, saved_barcodes, Some(qc))?;
    Ok(())
}

fn compute_qc_count<B>(
    fragments: Vec<BED<5>>,
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
    I: Iterator<Item = BED<5>>,
{
    let mut barcodes = HashMap::new();
    fragments.for_each(|frag| {
        let key = frag.name().unwrap().to_string();
        *barcodes.entry(key).or_insert(0) += 1;
    });
    barcodes
}

pub struct IntoInsertionIter {
    data: AnnDataElement<csr::CsrMatrix<u32>, Group>, 
    chrom_index: Vec<(String, u64)>,
}

impl IntoInsertionIter {
    pub fn iter<'a>(&'a self) -> InsertionIter<'a> {
        InsertionIter {
            data: self,
            iter: self.data.row_iter(),
        }
    }
}

pub struct InsertionIter<'a> {
    data: &'a IntoInsertionIter,
    iter: SparseRowIter<'a, u32>,
}

impl<'a> Iterator for InsertionIter<'a>
{
    type Item = Insertions;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|item| {
            let ins = item.into_iter().map(|(i, x)| {
                let locus = match self.data.chrom_index.binary_search_by_key(&i, |s| s.1.try_into().unwrap()) {
                    Ok(i_) => GenomicRange::new(self.data.chrom_index[i_].0.clone(), 0, 1),
                    Err(i_) => {
                        let (chr, p) = self.data.chrom_index[i_ - 1].clone();
                        GenomicRange::new(chr, i as u64 - p, i as u64 - p + 1)
                    },
                };
                (locus, x)
            }).collect();
            Insertions(ins)
        })
    }
}

pub fn read_insertions<'a>(file: Group) -> Result<IntoInsertionIter> {
    let chrs: StrVec = AnnDataIO::read(&file.dataset("reference_seq_name")?)?;
    let chr_sizes: Vec<u64> = AnnDataIO::read(&file.dataset("reference_seq_length")?)?;
    let chrom_index = chrs.0.into_iter().zip(
        std::iter::once(0).chain(chr_sizes.into_iter().scan(0, |state, x| {
            *state = *state + x;
            Some(*state)
        }))
    ).collect();
    Ok(IntoInsertionIter{
        data: AnnDataElement::new(file),
        chrom_index
    })
}


#[cfg(test)]
mod tests {
    use super::*;

    use flate2::read::GzDecoder;
    use std::fs::File;

    #[test]
    fn test_tsse() {
        let f = GzDecoder::new(File::open("../data/fragments.bed.gz").expect("xx"));
        let gencode = File::open("../data/gencode.gtf.gz").expect("xx");
        let promoter = make_promoter_map(read_tss(GzDecoder::new(gencode)));
        let expected = vec![12.702366127023662, 1.8181818181818181,
            6.1688311688311686, 1.8181818181818181, 0.0, 0.0,
            0.9090909090909091, 8.333333333333332, 0.9090909090909091,
            6.0606060606060606, 5.483405483405483, 6.28099173553719,
            8.869179600886916];

        let result: Vec<f64> = read_fragments(f).into_iter().map(|(_, fragments)| {
            let mut summary = FragmentSummary::new(&promoter);
            fragments.for_each(|frag| { summary.update(&frag); });
            summary.get_qc().tss_enrichment
        }).collect();
        assert_eq!(expected, result);
    }

}