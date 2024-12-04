pub mod similarity;
pub mod knn;

use std::path::Path;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::str::FromStr;
use anyhow::{Result, Context};

use bed_utils::bed::{BEDLike, NarrowPeak, merge_sorted_bed_with};
use bed_utils::extsort::ExternalSorterBuilder;

pub fn merge_peaks<I>(peaks: I, half_window_size: u64) -> impl Iterator<Item = Vec<NarrowPeak>>
where
    I: Iterator<Item = NarrowPeak>,
{
    fn iterative_merge(mut peaks: Vec<NarrowPeak>) -> Vec<NarrowPeak> {
        let mut result = Vec::new();
        while !peaks.is_empty() {
            let best_peak = peaks.iter()
                .max_by(|a, b| a.p_value.partial_cmp(&b.p_value).unwrap()).unwrap()
                .clone();
            peaks = peaks.into_iter().filter(|x| x.n_overlap(&best_peak) == 0).collect();
            result.push(best_peak);
        }
        result
    }

    let input = peaks.map(move |mut x| {
        let summit = x.start() + x.peak;
        x.start = summit.saturating_sub(half_window_size);
        x.end = summit + half_window_size + 1;
        x.peak = summit - x.start;
        x
    });
    let input = ExternalSorterBuilder::new()
        .with_compression(2)
        .build().unwrap()
        .sort_by(input, BEDLike::compare).unwrap()
        .map(|x| x.unwrap());
    merge_sorted_bed_with(input, iterative_merge)
}

pub fn clip_peak(mut peak: NarrowPeak, chrom_sizes: &crate::genome::ChromSizes) -> NarrowPeak {
    let chr = peak.chrom();
    let max_len = chrom_sizes.get(chr).expect(&format!("Size missing for chromosome: {}", chr));
    let new_start = peak.start().max(0).min(max_len);
    let new_end = peak.end().min(max_len);
    peak.set_start(new_start);
    peak.set_end(new_end);
    peak.peak = (new_start + peak.peak).min(new_end) - new_start;
    peak
}

#[derive(Debug, Clone, Copy)]
pub enum Compression {
    Gzip,
    Zstd,
}

impl FromStr for Compression {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "gzip" => Ok(Compression::Gzip),
            "zstd" | "zstandard" => Ok(Compression::Zstd),
            _ => Err(format!("unsupported compression: {}", s)),
        }
    }
}

pub fn open_file_for_write<P: AsRef<Path>>(
    filename: P,
    compression: Option<Compression>,
    compression_level: Option<u32>,
) -> Result<Box<dyn Write + Send>> {
    let buffer = BufWriter::new(
        File::create(&filename).with_context(|| format!("cannot create file: {}", filename.as_ref().display()))?
    );
    let writer: Box<dyn Write + Send> = match compression {
        None => Box::new(buffer),
        Some(Compression::Gzip) => Box::new(flate2::write::GzEncoder::new(buffer, flate2::Compression::new(compression_level.unwrap_or(6)))),
        Some(Compression::Zstd) => {
            let mut zstd = zstd::stream::Encoder::new(buffer, compression_level.unwrap_or(3) as i32)?;
            zstd.multithread(8)?;
            Box::new(zstd.auto_finish())
        },
    };
    Ok(writer)
}

/// Open a file, possibly compressed. Supports gzip and zstd.
pub fn open_file_for_read<P: AsRef<Path>>(file: P) -> Box<dyn std::io::Read> {
    match detect_compression(file.as_ref()) {
        Some(Compression::Gzip) => Box::new(flate2::read::MultiGzDecoder::new(File::open(file.as_ref()).unwrap())),
        Some(Compression::Zstd) => {
            let r = zstd::stream::read::Decoder::new(File::open(file.as_ref()).unwrap()).unwrap();
            Box::new(r)
        },
        None => Box::new(File::open(file.as_ref()).unwrap()),
    }
}

/// Determine the file compression type. Supports gzip and zstd.
fn detect_compression<P: AsRef<Path>>(file: P) -> Option<Compression> {
    if flate2::read::MultiGzDecoder::new(File::open(file.as_ref()).unwrap()).header().is_some() {
        Some(Compression::Gzip)
    } else if let Some(ext) = file.as_ref().extension() {
        if ext == "zst" {
            Some(Compression::Zstd)
        } else {
            None
        }
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bed_utils::bed::io::Reader;

    #[test]
    fn test_merge_peaks() {
        let input = "chr1\t9977\t16487\ta\t1000\t.\t74.611\t290.442\t293.049\t189
chr1\t9977\t16487\tb\t1000\t.\t74.1871\t295.33\t290.939\t425
chr1\t9977\t16487\tb\t1000\t.\t74.1871\t295\t290.939\t425
chr1\t9977\t16487\tb\t1000\t.\t74.1871\t295\t290.939\t625
chr1\t9977\t16487\tb\t1000\t.\t74.1871\t290\t290.939\t925
chr2\t9977\t16487\tb\t1000\t.\t74.1871\t295\t290.939\t625
chr2\t9977\t16487\tb\t1000\t.\t74.1871\t290\t290.939\t325
chr2\t9977\t16487\tb\t1000\t.\t74.1871\t290\t290.939\t525
chr2\t9977\t16487\tb\t1000\t.\t74.1871\t290\t290.939\t725
chr3\t0\t1164\tb\t1000\t.\t74.1871\t290\t290.939\t100
";
        let output = "chr1\t10202\t10603\tb\t1000\t.\t74.1871\t295.33\t290.939\t200
chr1\t10702\t11103\tb\t1000\t.\t74.1871\t290\t290.939\t200
chr2\t10402\t10803\tb\t1000\t.\t74.1871\t295\t290.939\t200
chr3\t0\t301\tb\t1000\t.\t74.1871\t290\t290.939\t100
";

        let expected: Vec<NarrowPeak> = Reader::new(output.as_bytes(), None)
            .into_records().map(|x| x.unwrap()).collect();
        let result: Vec<NarrowPeak> = merge_peaks(
            Reader::new(input.as_bytes(), None).into_records().map(|x| x.unwrap()),
            200
        ).flatten().collect();

        assert_eq!(expected, result);
    }
}