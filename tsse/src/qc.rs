use std::io;
use std::io::prelude::*;
use std::io::BufReader;                                                                                                                                           
use bed_utils::bed::{BED, BEDLike, tree::BedTree, io::Reader};
use itertools::{Itertools, GroupBy};

pub type CellBarcode = String;

#[derive(Clone, Debug, PartialEq)]
pub struct QualityControl {
    pub tss_enrichment: f64,
    pub num_unique_fragment: u64,
    pub frac_mitochondrial: f64,
    pub frac_duplicated: f64,
}

/// Compute QC metrics.
pub fn get_qc<I>(promoter: &BedTree<bool>, fragments: I) -> QualityControl
where
    I: Iterator<Item = BED<5>>,
{
    let mut tss_insertion_count: [f64; 4001] = [0.0; 4001];
    let mut num_unique_fragment: u64 = 0;
    let mut num_total_fragment: u64 = 0;
    let mut num_mitochondrial : u64 = 0;
    let mut update_tss_insertion_count = |ins: BED<3>| {
        for (entry, data) in promoter.find(&ins) {
            let pos: u64 =
                if *data {
                    ins.start() - entry.start()
                } else {
                    4000 - (entry.end() - 1 - ins.start())
                };
            tss_insertion_count[pos as usize] += 1.0;
        }
    };

    for fragment in fragments {
        let (ins1, ins2) = get_insertions(&fragment);
        update_tss_insertion_count(ins1);
        update_tss_insertion_count(ins2);

        num_unique_fragment += 1;
        num_total_fragment += *fragment.score().unwrap() as u64;
        if fragment.chrom() == "chrM" || fragment.chrom() == "M" { num_mitochondrial += 1 };
    }

    let bg_count: f64 = (tss_insertion_count[ .. 100].iter().sum::<f64>() +
            tss_insertion_count[3901 .. 4001].iter().sum::<f64>()) / 200.0 + 0.1;
    for i in 0 .. 4001 {
        tss_insertion_count[i] /= bg_count;
    }
    let tss_enrichment = *moving_average(5, &tss_insertion_count).iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let frac_duplicated: f64 = 1.0 - num_unique_fragment as f64 / num_total_fragment as f64;
    let frac_mitochondrial : f64 = num_mitochondrial as f64 / num_unique_fragment as f64;
    QualityControl { tss_enrichment, num_unique_fragment, frac_mitochondrial, frac_duplicated }
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
            let b: BED<3> = BED::new_bed3(chr, tss.saturating_sub(2000), tss + 2001);
            (b, is_fwd)
        }).collect()
}

fn get_insertions(rec: &BED<5>) -> (BED<3>, BED<3>) {
    ( BED::new_bed3(rec.chrom().to_string(), rec.start() + 75, rec.start() + 76)
    , BED::new_bed3(rec.chrom().to_string(), rec.end() - 76, rec.end() - 75) )
}

fn moving_average(half_window: usize, arr: &[f64]) -> Vec<f64> {
    let n = arr.len();
    let f = |i: usize| {
        let r = i.saturating_sub(half_window) .. std::cmp::min(i + half_window + 1, n);
        let l = r.len() as f64;
        arr[r].iter().sum::<f64>() / l
    };
    (0 .. arr.len()).map(f).collect()
}

/// Read and group fragments according to cell barcodes.
pub fn read_fragments<R>(r: R) -> GroupBy<CellBarcode, impl Iterator<Item = BED<5>>, impl FnMut(&BED<5>) -> CellBarcode>
where
    R: Read,
{
    group_cells_by_barcode(Reader::new(r).into_records().map(Result::unwrap))
}

pub fn group_cells_by_barcode<I>(fragments: I) -> GroupBy<CellBarcode, I, impl FnMut(&BED<5>) -> CellBarcode>
where
    I: Iterator<Item = BED<5>>,
{
    fragments.group_by(|x| { x.name().unwrap().to_string() })
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
        let expected = vec![11.857707509881424, 2.727272727272727, 6.583072100313478
                , 2.727272727272727, 0.0, 0.0, 1.8181818181818181, 6.1633281972265
                , 0.9090909090909091, 6.220095693779905, 5.965909090909091
                , 7.204116638078901, 9.312638580931262];

        let result: Vec<f64> = read_fragments(f).into_iter()
                .map(|(_, fragments)| get_qc(&promoter, fragments).tss_enrichment)
                .collect();
        assert_eq!(expected, result);
    }

}