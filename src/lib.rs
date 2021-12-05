pub mod tss_enrichment {
    use std::io;
    use std::io::prelude::*;
    use std::fs::File;
    use bio::io::bed;
    use std::io::BufReader;                                                                                                                                           
    use itertools::Itertools;
    use flate2::read::GzDecoder;
    use std::collections::HashMap;
    use std::collections::HashSet;
    use bio::data_structures::interval_tree;

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

    type PromoterMap = HashMap<String, interval_tree::IntervalTree<u64, bool>>;

    pub fn make_promoter_map<I: Iterator<Item = (String, u64, bool)>>(iter: I) -> PromoterMap {
        let mut hmap: HashMap<String, HashSet<(std::ops::Range<u64>, bool)>> = HashMap::new();
        for (chr, tss, is_fwd) in iter {
            let set = hmap.entry(chr).or_insert(HashSet::new());
            set.insert((tss.saturating_sub(2000) .. tss + 2001, is_fwd));
        }
        hmap.into_iter().map(|(chr, set)| (chr, interval_tree::IntervalTree::from_iter(set))).collect()
    }

    pub fn get_insertions(rec: bed::Record) -> [(String, u64); 2] {
        [(rec.chrom().to_string(), rec.start() + 75), (rec.chrom().to_string(), rec.end() - 76)]
    }

    pub fn read_fragments<R: Read>(file: R) -> impl Iterator<Item = bed::Record> {
        bed::Reader::new(file).into_records().map(Result::unwrap)
    }

    pub fn tsse<I: Iterator<Item = (String, u64)>>(promoter: &PromoterMap, insertions: I) -> f64 {
        let mut counts: [f64; 4001] = [0.0; 4001];
        for (chr, ins) in insertions {
            if (*promoter).contains_key(&chr) {
                for entry in (*promoter)[&chr].find(ins .. ins+1) {
                    let pos: u64 =
                        if *entry.data() {
                            ins - entry.interval().start
                        } else {
                            4000 - (entry.interval().end - 1 - ins)
                        };
                    counts[pos as usize] += 1.0;
                }
            }
    }
    let bg_count: f64 = (counts[ .. 100].iter().sum::<f64>() + counts[3901 .. 4001].iter().sum::<f64>()) / 200.0 + 0.1;
    for i in 0 .. 4001 {
        counts[i] /= bg_count;
    }
    *moving_average(5, &counts).iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
    }

    #[cfg(test)]
    #[test]
    fn it_works() {
        let f = GzDecoder::new(File::open("data/fragments.bed.gz").expect("xx"));
        let gencode = File::open("data/gencode.gtf.gz").expect("xx");
        let promoter = make_promoter_map(read_tss(GzDecoder::new(gencode)));
        let expected = vec![12.834224598930483, 1.8181818181818181,6.583072100313481,
                1.8181818181818181,0.0,0.0, 1.8181818181818181,5.956112852664577,
                0.9090909090909091, 5.896805896805896,5.965909090909091,
                6.734006734006734,9.312638580931264];
        let result: Vec<f64> = read_fragments(f).group_by(|x| x.name().unwrap().to_string()).into_iter()
                .map(|(_, fragments)| tsse(&promoter, fragments.map(get_insertions).flatten()))
                .collect();
        assert_eq!(expected, result);
    }
}