use criterion::{black_box, criterion_group, criterion_main, Criterion};
use criterion::BenchmarkId;
use project_a::qc::*;
use std::fs::File;
use flate2::read::GzDecoder;
use itertools::Itertools;

fn criterion_benchmark(c: &mut Criterion) {
    let gencode = File::open("data/gencode.gtf.gz").expect("xx");
    let input = ( "data/fragments.bed.gz"
                , make_promoter_map(read_tss(GzDecoder::new(gencode))));
    c.bench_with_input(BenchmarkId::new("TSSe", ""), &input, |b, i| {
        b.iter(|| {
            let f = GzDecoder::new(File::open(i.0).expect("xx"));
            for (bc, fragments) in read_fragments(f).group_by(|x| x.name().unwrap().to_string()).into_iter() {
                tsse(&i.1, fragments.map(get_insertions).flatten());
            }
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);