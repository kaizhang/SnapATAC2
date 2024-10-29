use crate::{
    preprocessing::{
        count_data::{ChromSizes, SnapData},
        Fragment,
    },
    utils::{self, Compression},
};

use anyhow::{bail, ensure, Context, Result};
use bed_utils::bed::{
    io,
    map::{GIntervalIndexSet, GIntervalMap},
    merge_sorted_bed_with, BEDLike, BedGraph, GenomicRange,
};
use bed_utils::coverage::SparseBinnedCoverage;
use bigtools::BigWigWrite;
use indexmap::IndexMap;
use indicatif::{style::ProgressStyle, ParallelProgressIterator, ProgressIterator};
use itertools::Itertools;
use log::info;
use rayon::iter::{IntoParallelIterator, ParallelBridge, ParallelIterator};
use std::{
    collections::{HashMap, HashSet},
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use tempfile::Builder;

#[derive(Debug, Clone, Copy)]
pub enum CoverageOutputFormat {
    BedGraph,
    BigWig,
}

impl std::str::FromStr for CoverageOutputFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "BEDGRAPH" => Ok(CoverageOutputFormat::BedGraph),
            "BIGWIG" => Ok(CoverageOutputFormat::BigWig),
            _ => Err(format!("unknown output format: {}", s)),
        }
    }
}

impl<T> Exporter for T where T: SnapData {}

pub trait Exporter: SnapData {
    fn export_fragments<P: AsRef<Path>>(
        &self,
        barcodes: Option<&Vec<&str>>,
        group_by: &Vec<&str>,
        selections: Option<HashSet<&str>>,
        min_fragment_length: Option<u64>,
        max_fragment_length: Option<u64>,
        dir: P,
        prefix: &str,
        suffix: &str,
        compression: Option<Compression>,
        compression_level: Option<u32>,
    ) -> Result<HashMap<String, PathBuf>> {
        ensure!(self.n_obs() == group_by.len(), "lengths differ");
        let mut groups: HashSet<&str> = group_by.iter().map(|x| *x).unique().collect();
        if let Some(select) = selections {
            groups.retain(|x| select.contains(x));
        }
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("cannot create directory: {}", dir.as_ref().display()))?;
        let files = groups
            .into_iter()
            .map(|x| {
                let filename = prefix.to_string() + x + suffix;
                if !sanitize_filename::is_sanitized(&filename) {
                    bail!("invalid filename: {}", filename);
                }
                let filename = dir.as_ref().join(filename);
                let writer = utils::open_file_for_write(&filename, compression, compression_level)?;
                Ok((x, (filename, Arc::new(Mutex::new(writer)))))
            })
            .collect::<Result<HashMap<_, _>>>()?;

        let style = ProgressStyle::with_template(
            "[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} (eta: {eta})",
        )?;
        let mut counts = self.get_count_iter(1000)?;
        if let Some(min_len) = min_fragment_length {
            counts = counts.min_fragment_size(min_len);
        }
        if let Some(max_len) = max_fragment_length {
            counts = counts.max_fragment_size(max_len);
        }

        counts
            .into_fragments()
            .map(move |(vals, start, _)| {
                let mut ordered = HashMap::new();
                vals.into_iter().enumerate().for_each(|(i, xs)| {
                    let k = group_by[start + i];
                    let entry = ordered.entry(k).or_insert_with(Vec::new);
                    if let Some(barcodes_) = barcodes {
                        let bc = barcodes_[start + i];
                        entry.extend(xs.into_iter().map(|mut x| {
                            x.barcode = Some(bc.to_string());
                            x
                        }));
                    } else {
                        entry.extend(xs.into_iter());
                    }
                });
                ordered
            })
            .progress_with_style(style)
            .try_for_each(|vals| {
                vals.into_iter().par_bridge().try_for_each(|(k, beds)| {
                    if let Some((_, fl)) = files.get(k) {
                        let mut fl = fl.lock().unwrap();
                        beds.into_iter().try_for_each(|x| writeln!(fl, "{}", x))?;
                    }
                    anyhow::Ok(())
                })
            })?;
        Ok(files
            .into_iter()
            .map(|(k, (v, _))| (k.to_string(), v))
            .collect())
    }

    fn export_coverage<P: AsRef<Path> + std::marker::Sync>(
        &self,
        group_by: &Vec<&str>,
        selections: Option<HashSet<&str>>,
        resolution: usize,
        blacklist_regions: Option<&GIntervalMap<()>>,
        normalization: Option<Normalization>,
        ignore_for_norm: Option<&HashSet<&str>>,
        min_fragment_length: Option<u64>,
        max_fragment_length: Option<u64>,
        smooth_length: Option<u16>,
        dir: P,
        prefix: &str,
        suffix: &str,
        format: CoverageOutputFormat,
        compression: Option<Compression>,
        compression_level: Option<u32>,
        temp_dir: Option<P>,
        num_threads: Option<usize>,
    ) -> Result<HashMap<String, PathBuf>> {
        // Create directory
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("cannot create directory: {}", dir.as_ref().display()))?;

        let temp_dir = if let Some(tmp) = temp_dir {
            Builder::new()
                .tempdir_in(tmp)
                .expect("failed to create tmperorary directory")
        } else {
            Builder::new()
                .tempdir()
                .expect("failed to create tmperorary directory")
        };

        info!("Exporting fragments...");
        let fragment_files = self.export_fragments(
            None,
            group_by,
            selections,
            min_fragment_length,
            max_fragment_length,
            temp_dir.path(),
            "",
            "",
            Some(Compression::Gzip),
            Some(1),
        )?;

        let chrom_sizes = self.read_chrom_sizes()?;
        info!("Creating coverage files...");
        let style = ProgressStyle::with_template(
            "[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} (eta: {eta})",
        )
        .unwrap();

        let pool = if let Some(n) = num_threads {
            rayon::ThreadPoolBuilder::new().num_threads(n)
        } else {
            rayon::ThreadPoolBuilder::new()
        };
        pool.build().unwrap().install(|| {
            fragment_files
                .into_iter()
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|(grp, filename)| {
                    let output = dir
                        .as_ref()
                        .join(prefix.to_string() + grp.replace("/", "+").as_str() + suffix);

                    // Make BedGraph
                    let bedgraph = create_bedgraph_from_fragments(
                        io::Reader::new(utils::open_file_for_read(filename), None)
                            .into_records()
                            .map(Result::unwrap),
                        &chrom_sizes,
                        resolution as u64,
                        smooth_length,
                        blacklist_regions,
                        normalization,
                        ignore_for_norm,
                    );

                    match format {
                        CoverageOutputFormat::BedGraph => {
                            let mut writer = utils::open_file_for_write(
                                &output,
                                compression,
                                compression_level,
                            )?;
                            bedgraph
                                .into_iter()
                                .for_each(|x| writeln!(writer, "{}", x).unwrap());
                        }
                        CoverageOutputFormat::BigWig => {
                            create_bigwig_from_bedgraph(bedgraph, &chrom_sizes, &output)?;
                        }
                    }

                    Ok((grp.to_string(), output))
                })
                .progress_with_style(style)
                .collect()
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Normalization {
    RPKM, // Reads per kilobase per million mapped reads. RPKM (per bin) =
    // number of reads per bin / (number of mapped reads (in millions) * bin length (kb)).
    CPM, // Counts per million mapped reads. CPM (per bin) =
    // number of reads per bin / number of mapped reads (in millions).
    BPM, // Bins Per Million mapped reads, same as TPM in RNA-seq. BPM (per bin) =
    // number of reads per bin / sum of all reads per bin (in millions).
    RPGC, // Reads per genomic content. RPGC (per bin) =
          // number of reads per bin / scaling factor for 1x average coverage.
}

impl std::str::FromStr for Normalization {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "RPKM" => Ok(Normalization::RPKM),
            "CPM" => Ok(Normalization::CPM),
            "BPM" => Ok(Normalization::BPM),
            "RPGC" => Ok(Normalization::RPGC),
            _ => Err(format!("unknown normalization method: {}", s)),
        }
    }
}

/// Create a BedGraph file from fragments.
///
/// The values represent the sequence coverage (or sequencing depth), which refers
/// to the number of reads that include a specific nucleotide of a reference genome.
/// For paired-end data, the coverage is computed as the number of times a base
/// is read or spanned by paired ends or mate paired reads
///
/// # Arguments
///
/// * `fragments` - iterator of fragments
/// * `chrom_sizes` - chromosome sizes
/// * `bin_size` - Size of the bins, in bases, for the output of the bigwig/bedgraph file.
/// * `smooth_length` - Length of the smoothing window for the output of the bigwig/bedgraph file.
///                     For example, if the bin_size is set to 20 and the smooth_length is set to 3,
///                     then, for each bin, the average of the bin and its left and right neighbors
///                     is considered (the total of 60 bp).
/// * `blacklist_regions` - Blacklist regions to be ignored.
/// * `normalization` - Normalization method.
/// * `ignore_for_norm` - Chromosomes to be ignored for normalization.
fn create_bedgraph_from_fragments<I>(
    fragments: I,
    chrom_sizes: &ChromSizes,
    bin_size: u64,
    smooth_length: Option<u16>,
    blacklist_regions: Option<&GIntervalMap<()>>,
    normalization: Option<Normalization>,
    ignore_for_norm: Option<&HashSet<&str>>,
) -> Vec<BedGraph<f32>>
where
    I: Iterator<Item = Fragment>,
{
    let genome: GIntervalIndexSet = chrom_sizes
        .into_iter()
        .map(|(k, v)| GenomicRange::new(k, 0, *v))
        .collect();
    let mut counter = SparseBinnedCoverage::new(&genome, bin_size);
    let mut total_count = 0.0;
    fragments.for_each(|frag| {
        let not_in_blacklist = blacklist_regions.map_or(true, |bl| !bl.is_overlapped(&frag));
        if not_in_blacklist {
            if ignore_for_norm.map_or(true, |x| !x.contains(frag.chrom())) {
                total_count += 1.0;
            }
            counter.insert(&frag, 1.0);
        }
    });

    let norm_factor = match normalization {
        None => 1.0,
        Some(Normalization::RPKM) => total_count * bin_size as f32 / 1e9,
        Some(Normalization::CPM) => total_count / 1e6,
        Some(Normalization::BPM) => todo!(),
        Some(Normalization::RPGC) => todo!(),
    };

    let counts = counter
        .get_coverage()
        .iter()
        .map(|(i, val)| (counter.get_region(*i).unwrap(), *val / norm_factor));

    let counts: Box<dyn Iterator<Item = _>> = if let Some(smooth_length) = smooth_length {
        let smooth_left = (smooth_length - 1) / 2;
        let smooth_right = smooth_left + (smooth_left - 1) % 2;
        Box::new(smooth_bedgraph(counts, bin_size, smooth_left, smooth_right, chrom_sizes))
    } else {
        Box::new(counts)
    };

    let chunks = counts
        .map(|(region, count)| BedGraph::from_bed(&region, count))
        .chunk_by(|x| x.value);
    chunks.into_iter()
        .flat_map(|(_, groups)| {
            merge_sorted_bed_with(groups, |beds| {
                let mut iter = beds.into_iter();
                let mut first = iter.next().unwrap();
                if let Some(last) = iter.last() {
                    first.set_end(last.end());
                }
                first
            })
        })
        .collect()
}

fn smooth_bedgraph<'a, I>(
    input: I,
    bin_size: u64,
    left_window_len: u16,
    right_window_len: u16,
    chr_sizes: &'a ChromSizes,
) -> impl Iterator<Item = (GenomicRange, f32)> + 'a
where
    I: Iterator<Item = (GenomicRange, f32)> + 'a,
{
    let left_bases = left_window_len as u64 * bin_size;
    let right_bases = right_window_len as u64 * bin_size;
    get_overlapped_chunks(input, left_bases, right_bases).flat_map(move |chunk| {
        let size = chr_sizes.get(chunk[0].0.chrom()).unwrap();
        moving_average(&chunk, left_bases, right_bases, size)
    })
}

fn get_overlapped_chunks<I>(
    mut input: I,
    left_bases: u64,
    right_bases: u64,
) -> impl Iterator<Item = Vec<(GenomicRange, f32)>>
where
    I: Iterator<Item = (GenomicRange, f32)>,
{
    let mut buffer = vec![input.next().unwrap()];
    std::iter::from_fn(move || {
        while let Some((cur_region, cur_val)) = input.next() {
            let (prev_region, _) = buffer.last().unwrap();
            if cur_region.chrom() == prev_region.chrom()
                && prev_region.end() + right_bases > cur_region.start().saturating_sub(left_bases)
            {
                buffer.push((cur_region, cur_val));
            } else {
                let result = Some(buffer.clone());
                buffer = vec![(cur_region, cur_val)];
                return result;
            }
        }
        if buffer.is_empty() {
            None
        } else {
            let result = Some(buffer.clone());
            buffer = Vec::new();
            return result;
        }
    })
}

fn moving_average(
    chunk: &Vec<(GenomicRange, f32)>,
    left_bases: u64,
    right_bases: u64,
    chrom_size: u64,
) -> impl Iterator<Item = (GenomicRange, f32)> {
    let bin_size = chunk[0].0.len() as u64;
    let n_left = (left_bases / bin_size) as usize;
    let n_right = (right_bases / bin_size) as usize;
    let chrom = chunk.first().unwrap().0.chrom();
    let chunk_start = chunk.first().unwrap().0.start();
    let chunk_end = (chunk.last().unwrap().0.end() + right_bases).min(chrom_size);
    let mut regions = GenomicRange::new(chrom, chunk_start.saturating_sub(left_bases), chunk_start)
        .rsplit_by_len(bin_size)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .chain(GenomicRange::new(chrom, chunk_start, chunk_end).split_by_len(bin_size))
        .map(|x| (x, 0.0))
        .collect::<IndexMap<_, _>>();
    // Add values
    chunk
        .iter()
        .for_each(|(gr, val)| *regions.get_mut(gr).unwrap() = *val);
    let len = regions.len();
    // Compute the average
    (0..len).map(move |i| {
        let s: f32 = (i.saturating_sub(n_left)..(i + n_right + 1).min(len))
            .map(|x| regions[x])
            .sum();
        let val = s / (n_right + n_left + 1) as f32;
        (regions.get_index(i).unwrap().0.clone(), val)
    })
}

/// Create a bigwig file from BedGraph records.
fn create_bigwig_from_bedgraph<P: AsRef<Path>>(
    mut bedgraph: Vec<BedGraph<f32>>,
    chrom_sizes: &ChromSizes,
    filename: P,
) -> Result<()> {
    // perform clipping to make sure the end of each region is within the range.
    bedgraph
        .iter_mut()
        .chunk_by(|x| x.chrom().to_string())
        .into_iter()
        .for_each(|(chr, groups)| {
            let size = chrom_sizes
                .get(&chr)
                .expect(&format!("chromosome not found: {}", chr));
            let bed = groups.last().unwrap();
            if bed.end() > size {
                bed.set_end(size);
            }
        });

    // write to bigwig file
    BigWigWrite::create_file(
        filename.as_ref().to_str().unwrap().to_string(),
        chrom_sizes
            .into_iter()
            .map(|(k, v)| (k.to_string(), *v as u32))
            .collect(),
    )?
    .write(
        bigtools::beddata::BedParserStreamingIterator::wrap_iter(
            bedgraph.into_iter().map(|x| {
                let val = bigtools::Value {
                    start: x.start() as u32,
                    end: x.end() as u32,
                    value: x.value,
                };
                let res: Result<_, bigtools::bed::bedparser::BedValueError> =
                    Ok((x.chrom().to_string(), val));
                res
            }),
            false,
        ),
        tokio::runtime::Runtime::new().unwrap(),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smoothing() {
        let input = vec![
            (GenomicRange::new("chr1", 0, 100), 1.0),
            (GenomicRange::new("chr1", 200, 300), 2.0),
            (GenomicRange::new("chr1", 500, 600), 3.0),
            (GenomicRange::new("chr1", 1000, 1100), 5.0),
        ];

        assert_eq!(
            smooth_bedgraph(
                input.into_iter(),
                100,
                2,
                2,
                &[("chr1", 10000000)].into_iter().collect(),
            ).collect::<Vec<_>>(),
            vec![
                (GenomicRange::new("chr1", 0, 100), 0.6),
                (GenomicRange::new("chr1", 100, 200), 0.6),
                (GenomicRange::new("chr1", 200, 300), 0.6),
                (GenomicRange::new("chr1", 300, 400), 1.0),
                (GenomicRange::new("chr1", 400, 500), 1.0),
                (GenomicRange::new("chr1", 500, 600), 0.6),
                (GenomicRange::new("chr1", 600, 700), 0.6),
                (GenomicRange::new("chr1", 700, 800), 0.6),
                (GenomicRange::new("chr1", 800, 900), 1.0),
                (GenomicRange::new("chr1", 900, 1000), 1.0),
                (GenomicRange::new("chr1", 1000, 1100), 1.0),
                (GenomicRange::new("chr1", 1100, 1200), 1.0),
                (GenomicRange::new("chr1", 1200, 1300), 1.0),
            ],
        );
    }
}