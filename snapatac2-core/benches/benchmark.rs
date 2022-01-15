use criterion::{black_box, criterion_group, criterion_main, Criterion};
use criterion::BenchmarkId;
use crate::qc::*;
use std::fs::File;
use flate2::read::GzDecoder;
use itertools::Itertools;
use bed_utils::bed::{BED, BEDLike, tree::BedTree, io::Reader};

fn criterion_benchmark(c: &mut Criterion) {
    let gencode = File::open("../data/gencode.gtf.gz").expect("xx");
    let input = ( "../data/fragments.bed.gz"
                , make_promoter_map(read_tss(GzDecoder::new(gencode))));
    c.bench_with_input(BenchmarkId::new("TSSe", ""), &input, |b, i| {
        b.iter(|| {
            let f = GzDecoder::new(File::open(i.0).expect("xx"));

            for (bc, fragments) in read_fragments(f).into_iter() {
                let qc = get_qc(&promoter, fragments);
                println!("{}\t{}\t{}\t{}\t{}", bc, qc.tss_enrichment, qc.num_unique_fragment, qc.frac_mitochondrial, qc.frac_duplicated);

            for (bc, fragments) in Reader::new(f).records::<BED<4>>().map(Result::unwrap).group_by(|x| x.name().unwrap().to_string()).into_iter() {
                tsse(&i.1, fragments.map(get_insertions).flatten());
            }
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);