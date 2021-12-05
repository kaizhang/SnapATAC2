extern crate ProjectA;

use std::env;
use ProjectA::tss_enrichment::*;
use std::fs::File;
use flate2::read::GzDecoder;
use itertools::Itertools;

fn main() {
    let args: Vec<String> = env::args().collect();
    let gencode = GzDecoder::new(File::open(&args[1]).expect("xx"));
    let f = GzDecoder::new(File::open(&args[2]).expect("xx"));
    let promoter = make_promoter_map(read_tss(gencode));
    for (bc, fragments) in read_fragments(f).group_by(|x| x.name().unwrap().to_string()).into_iter() {
        let t: f64 = tsse(&promoter, fragments.map(get_insertions).flatten());
        println!("{}\t{}", bc, t);
    }
}