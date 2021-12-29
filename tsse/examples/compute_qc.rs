use std::env;
use project_a::qc::*;
use std::fs::File;
use flate2::read::GzDecoder;
use itertools::Itertools;

fn main() {
    let args: Vec<String> = env::args().collect();
    let gencode = GzDecoder::new(File::open(&args[1]).expect("xx"));
    let f = GzDecoder::new(File::open(&args[2]).expect("xx"));
    let promoter = make_promoter_map(read_tss(gencode));
    for (bc, fragments) in read_fragments(f).into_iter() {
        let qc = get_qc(&promoter, fragments);
        println!("{}\t{}\t{}\t{}\t{}", bc, qc.tss_enrichment, qc.num_unique_fragment, qc.frac_mitochondrial, qc.frac_duplicated);
    }
}