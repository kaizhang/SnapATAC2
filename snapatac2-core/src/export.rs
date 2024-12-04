use crate::genome::ChromSizes;
use crate::SnapData;
use crate::{
    preprocessing::Fragment,
    utils::{self, Compression},
};

use anyhow::{bail, ensure, Context, Result};
use bed_utils::{
    bed::{io, map::GIntervalMap, merge_sorted_bedgraph, BEDLike, BedGraph, GenomicRange},
    extsort::ExternalSorterBuilder,
};
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
        let mut fragment_data = self.get_fragment_iter(1000)?;
        if let Some(min_len) = min_fragment_length {
            fragment_data = fragment_data.min_fragment_size(min_len);
        }
        if let Some(max_len) = max_fragment_length {
            fragment_data = fragment_data.max_fragment_size(max_len);
        }

        fragment_data
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
        include_for_norm: Option<&GIntervalMap<()>>,
        exclude_for_norm: Option<&GIntervalMap<()>>,
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

                    let fragments = io::Reader::new(utils::open_file_for_read(filename), None)
                        .into_records::<Fragment>()
                        .map(Result::unwrap);
                    let fragments = ExternalSorterBuilder::new()
                        .with_tmp_dir(temp_dir.path())
                        .build()?
                        .sort_by(fragments, |a, b| a.compare(b))?
                        .map(Result::unwrap);

                    // Make BedGraph
                    let bedgraph = create_bedgraph_from_sorted_fragments(
                        fragments,
                        &chrom_sizes,
                        resolution as u64,
                        smooth_length,
                        blacklist_regions,
                        normalization,
                        include_for_norm,
                        exclude_for_norm,
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
/// * `include_for_norm` - If specified, only the regions that overlap with these intervals will be used for normalization.
/// * `exclude_for_norm` - If specified, the regions that overlap with these intervals will be
///                        excluded from normalization. If a region is in both "include_for_norm" and
///                        "exclude_for_norm", it will be excluded.
fn create_bedgraph_from_sorted_fragments<I>(
    fragments: I,
    chrom_sizes: &ChromSizes,
    bin_size: u64,
    smooth_length: Option<u16>,
    blacklist_regions: Option<&GIntervalMap<()>>,
    normalization: Option<Normalization>,
    include_for_norm: Option<&GIntervalMap<()>>,
    exclude_for_norm: Option<&GIntervalMap<()>>,
) -> Vec<BedGraph<f32>>
where
    I: Iterator<Item = Fragment>,
{
    let mut norm_factor = 0.0;
    let bedgraph = fragments.flat_map(|frag| {
        if blacklist_regions.map_or(false, |bl| bl.is_overlapped(&frag)) {
            None
        } else {
            if include_for_norm.map_or(true, |x| x.is_overlapped(&frag))
                && !exclude_for_norm.map_or(false, |x| x.is_overlapped(&frag))
            {
                norm_factor += 1.0;
            }
            let mut frag = BedGraph::from_bed(&frag, 1.0);
            fit_to_bin(&mut frag, bin_size);
            clip_bed(frag, chrom_sizes)
        }
    });
    let mut bedgraph: Vec<_> = merge_sorted_bedgraph(bedgraph).collect();

    norm_factor = match normalization {
        None => 1.0,
        Some(Normalization::RPKM) => norm_factor * bin_size as f32 / 1e9,
        Some(Normalization::CPM) => norm_factor / 1e6,
        Some(Normalization::BPM) => bedgraph.iter().map(|x| x.value).sum::<f32>() / 1e6,
        Some(Normalization::RPGC) => todo!(),
    };

    bedgraph.iter_mut().for_each(|x| x.value /= norm_factor);

    if let Some(smooth_length) = smooth_length {
        let smooth_left = (smooth_length - 1) / 2;
        let smooth_right = smooth_left + (smooth_left - 1) % 2;
        bedgraph = smooth_bedgraph(
            bedgraph.into_iter(),
            bin_size,
            smooth_left,
            smooth_right,
            chrom_sizes,
        )
        .collect();
    }

    bedgraph
}

fn smooth_bedgraph<'a, I>(
    input: I,
    bin_size: u64,
    left_window_len: u16,
    right_window_len: u16,
    chr_sizes: &'a ChromSizes,
) -> impl Iterator<Item = BedGraph<f32>> + 'a
where
    I: Iterator<Item = BedGraph<f32>> + 'a,
{
    let left_bases = left_window_len as u64 * bin_size;
    let right_bases = right_window_len as u64 * bin_size;
    get_overlapped_chunks(input, left_bases, right_bases).flat_map(move |chunk| {
        let size = chr_sizes.get(chunk[0].chrom()).unwrap();
        moving_average(&chunk, left_bases, right_bases, size)
    })
}

fn get_overlapped_chunks<I>(
    mut input: I,
    left_bases: u64,
    right_bases: u64,
) -> impl Iterator<Item = Vec<BedGraph<f32>>>
where
    I: Iterator<Item = BedGraph<f32>>,
{
    let mut buffer = vec![input.next().unwrap()];
    std::iter::from_fn(move || {
        while let Some(cur) = input.next() {
            let prev = buffer.last().unwrap();
            if cur.chrom() == prev.chrom()
                && prev.end() + right_bases > cur.start().saturating_sub(left_bases)
            {
                buffer.push(cur);
            } else {
                let result = Some(buffer.clone());
                buffer = vec![cur];
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
    chunk: &Vec<BedGraph<f32>>,
    left_bases: u64,
    right_bases: u64,
    chrom_size: u64,
) -> impl Iterator<Item = BedGraph<f32>> {
    let bin_size = chunk[0].len() as u64;
    let n_left = (left_bases / bin_size) as usize;
    let n_right = (right_bases / bin_size) as usize;
    let chrom = chunk.first().unwrap().chrom();
    let chunk_start = chunk.first().unwrap().start();
    let chunk_end = (chunk.last().unwrap().end() + right_bases).min(chrom_size);
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
        .for_each(|gr| *regions.get_mut(&gr.to_genomic_range()).unwrap() = gr.value);
    let len = regions.len();
    // Compute the average
    (0..len).map(move |i| {
        let s: f32 = (i.saturating_sub(n_left)..(i + n_right + 1).min(len))
            .map(|x| regions[x])
            .sum();
        let val = s / (n_right + n_left + 1) as f32;
        BedGraph::from_bed(regions.get_index(i).unwrap().0, val)
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

fn clip_bed<B: BEDLike>(mut bed: B, chr_size: &ChromSizes) -> Option<B> {
    let size = chr_size.get(bed.chrom())?;
    if bed.start() >= size {
        return None;
    }
    if bed.end() > size {
        bed.set_end(size);
    }
    Some(bed)
}

fn fit_to_bin<B: BEDLike>(x: &mut B, bin_size: u64) {
    if bin_size > 1 {
        if x.start() % bin_size != 0 {
            x.set_start(x.start() - x.start() % bin_size);
        }
        if x.end() % bin_size != 0 {
            x.set_end(x.end() + bin_size - x.end() % bin_size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bedgraph1() {
        let fragments: Vec<Fragment> = vec![
            Fragment::new("chr1", 0, 10),
            Fragment::new("chr1", 3, 13),
            Fragment::new("chr1", 5, 41),
            Fragment::new("chr1", 8, 18),
            Fragment::new("chr1", 15, 25),
            Fragment::new("chr1", 22, 24),
            Fragment::new("chr1", 23, 33),
            Fragment::new("chr1", 29, 40),
        ];
        let genome: ChromSizes = [("chr1", 50)].into_iter().collect();

        let output: Vec<_> = create_bedgraph_from_sorted_fragments(
            fragments.clone().into_iter(),
            &genome,
            3,
            None,
            None,
            None,
            None,
            None,
        )
        .into_iter()
        .map(|x| x.value)
        .collect();
        let expected = vec![1.0, 3.0, 4.0, 3.0, 2.0, 4.0, 3.0, 2.0];
        assert_eq!(output, expected);

        let output: Vec<_> = create_bedgraph_from_sorted_fragments(
            fragments.clone().into_iter(),
            &genome,
            5,
            None,
            None,
            None,
            None,
            None,
        )
        .into_iter()
        .map(|x| x.value)
        .collect();
        let expected = vec![2.0, 4.0, 3.0, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(output, expected);
    }

    #[test]
    fn test_bedgraph2() {
        let reader = crate::utils::open_file_for_read("test/fragments.tsv.gz");
        let mut reader = bed_utils::bed::io::Reader::new(reader, None);
        let fragments: Vec<Fragment> = reader.records().map(|x| x.unwrap()).collect();

        let reader = crate::utils::open_file_for_read("test/coverage.bdg.gz");
        let mut reader = bed_utils::bed::io::Reader::new(reader, None);
        let mut expected: Vec<BedGraph<f32>> = reader.records().map(|x| x.unwrap()).collect();

        let output = create_bedgraph_from_sorted_fragments(
            fragments.clone().into_iter(),
            &[("chr1", 248956422), ("chr2", 242193529)]
                .into_iter()
                .collect(),
            1,
            None,
            None,
            None,
            None,
            None,
        );
        assert_eq!(
            output,
            expected,
            "Left: {:?}\n\n{:?}",
            &output[..10],
            &expected[..10]
        );

        let output = create_bedgraph_from_sorted_fragments(
            fragments.into_iter(),
            &[("chr1", 248956422), ("chr2", 242193529)]
                .into_iter()
                .collect(),
            1,
            None,
            None,
            Some(Normalization::BPM),
            None,
            None,
        );
        let scale_factor: f32 = expected.iter().map(|x| x.value).sum::<f32>() / 1e6;
        expected = expected
            .into_iter()
            .map(|mut x| {
                x.value = x.value / scale_factor;
                x
            })
            .collect::<Vec<_>>();
        assert_eq!(
            output,
            expected,
            "Left: {:?}\n\n{:?}",
            &output[..10],
            &expected[..10]
        );
    }

    #[test]
    fn test_smoothing() {
        let input = vec![
            BedGraph::new("chr1", 0, 100, 1.0),
            BedGraph::new("chr1", 200, 300, 2.0),
            BedGraph::new("chr1", 500, 600, 3.0),
            BedGraph::new("chr1", 1000, 1100, 5.0),
        ];

        assert_eq!(
            smooth_bedgraph(
                input.into_iter(),
                100,
                2,
                2,
                &[("chr1", 10000000)].into_iter().collect(),
            )
            .collect::<Vec<_>>(),
            vec![
                BedGraph::new("chr1", 0, 100, 0.6),
                BedGraph::new("chr1", 100, 200, 0.6),
                BedGraph::new("chr1", 200, 300, 0.6),
                BedGraph::new("chr1", 300, 400, 1.0),
                BedGraph::new("chr1", 400, 500, 1.0),
                BedGraph::new("chr1", 500, 600, 0.6),
                BedGraph::new("chr1", 600, 700, 0.6),
                BedGraph::new("chr1", 700, 800, 0.6),
                BedGraph::new("chr1", 800, 900, 1.0),
                BedGraph::new("chr1", 900, 1000, 1.0),
                BedGraph::new("chr1", 1000, 1100, 1.0),
                BedGraph::new("chr1", 1100, 1200, 1.0),
                BedGraph::new("chr1", 1200, 1300, 1.0),
            ],
        );
    }
}
