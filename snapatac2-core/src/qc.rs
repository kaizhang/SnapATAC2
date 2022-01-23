use std::io;
use std::io::prelude::*;
use std::io::BufReader;                                                                                                                                           
use bed_utils::bed::{GenomicRange, BED, BEDLike, tree::BedTree, io::Reader};
use itertools::{Itertools, GroupBy};
use std::ops::Div;

pub type CellBarcode = String;

#[derive(Clone, Debug, PartialEq)]
pub struct QualityControl {
    pub tss_enrichment: f64,
    pub num_unique_fragment: u64,
    pub frac_mitochondrial: f64,
    pub frac_duplicated: f64,
}

pub(crate) struct FragmentSummary {
    promoter_insertion_count: [u64; 4001],
    pub(crate) num_unique_fragment: u64,
    num_total_fragment: u64, 
    num_mitochondrial : u64,
}

impl FragmentSummary {
    pub(crate) fn new() -> Self { FragmentSummary {
        promoter_insertion_count: [0; 4001],
        num_unique_fragment: 0,
        num_total_fragment: 0,
        num_mitochondrial: 0 }
    }

    pub(crate) fn update(&mut self, promoter: &BedTree<bool>, fragment: &BED<5>) {
        self.num_unique_fragment += 1;
        self.num_total_fragment += *fragment.score().unwrap() as u64;
        if fragment.chrom() == "chrM" || fragment.chrom() == "M" {
            self.num_mitochondrial += 1;
        }
        for ins in get_insertions(fragment) {
            for (entry, data) in promoter.find(&ins) {
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

    pub(crate) fn get_qc(self) -> QualityControl {
        let bg_count: f64 =
            ( self.promoter_insertion_count[ .. 100].iter().sum::<u64>() +
            self.promoter_insertion_count[3901 .. 4001].iter().sum::<u64>() ) as f64 /
            200.0 + 0.1;
        let tss_enrichment = moving_average(5, &self.promoter_insertion_count)
            .max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap().div(bg_count);
        let frac_duplicated = 1.0 -
            self.num_unique_fragment as f64 / self.num_total_fragment as f64;
        let frac_mitochondrial = self.num_mitochondrial as f64 /
            self.num_unique_fragment as f64;
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
    Reader::new(r).into_records().map(Result::unwrap)
        .group_by(|x: &BED<5>| { x.name().unwrap().to_string() })
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
            let mut summary = FragmentSummary::new();
            fragments.for_each(|frag| { summary.update(&promoter, &frag); });
            summary.get_qc().tss_enrichment
        }).collect();
        assert_eq!(expected, result);
    }

}