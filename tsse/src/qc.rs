use std::io;
use std::io::prelude::*;
use std::io::BufReader;                                                                                                                                           
use bed_utils::bed::{BED, BEDLike, tree::BedTree, io::Reader};

fn moving_average(half_window: usize, arr: &[f64]) -> Vec<f64> {
    let n = arr.len();
    let f = |i: usize| {
        let r = i.saturating_sub(half_window) .. std::cmp::min(i + half_window + 1, n);
        let l = r.len() as f64;
        arr[r].iter().sum::<f64>() / l
    };
    (0 .. arr.len()).map(f).collect()
}

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

pub fn get_insertions(rec: BED<4>) -> [BED<3>; 2] {
    [ BED::new_bed3(rec.chrom().to_string(), rec.chrom_start() + 75, rec.chrom_start() + 76)
    , BED::new_bed3(rec.chrom().to_string(), rec.chrom_end() - 76, rec.chrom_end() - 75) ]
}

pub fn tsse<I: Iterator<Item = BED<3>>>(promoter: &BedTree<bool>, insertions: I) -> f64 {
    let mut counts: [f64; 4001] = [0.0; 4001];
    for ins in insertions {
        for (entry, data) in promoter.find(&ins) {
            let pos: u64 =
                if *data {
                    ins.chrom_start() - entry.chrom_start()
                } else {
                    4000 - (entry.chrom_end() - 1 - ins.chrom_start())
                };
            counts[pos as usize] += 1.0;
        }
    }
    let bg_count: f64 = (counts[ .. 100].iter().sum::<f64>() + counts[3901 .. 4001].iter().sum::<f64>()) / 200.0 + 0.1;
    for i in 0 .. 4001 {
        counts[i] /= bg_count;
    }
    *moving_average(5, &counts).iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    use flate2::read::GzDecoder;
    use std::fs::File;
    use itertools::Itertools;

    #[test]
    fn test_tsse() {
        let f = GzDecoder::new(File::open("../data/fragments.bed.gz").expect("xx"));
        let gencode = File::open("../data/gencode.gtf.gz").expect("xx");
        let promoter = make_promoter_map(read_tss(GzDecoder::new(gencode)));
        let expected = vec![11.857707509881424, 2.727272727272727, 6.583072100313478
                , 2.727272727272727, 0.0, 0.0, 1.8181818181818181, 6.1633281972265
                , 0.9090909090909091, 6.220095693779905, 5.965909090909091
                , 7.204116638078901, 9.312638580931262];

        let result: Vec<f64> = Reader::new(f).records::<BED<4>>().map(Result::unwrap)
                .group_by(|x| { x.name().unwrap().to_string() })
                .into_iter()
                .map(|(_, fragments)| tsse(&promoter, fragments.map(get_insertions).flatten()))
                .collect();
        assert_eq!(expected, result);
    }

}