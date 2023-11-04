use crate::{preprocessing::{count_data::{SnapData, GenomeCoverage, CoverageType, ChromSizes}, Fragment}, utils::{self, Compression}};

use anyhow::{Context, Result, ensure};
use itertools::Itertools;
use log::info;
use std::{
    sync::{Arc, Mutex}, io::Write,
    path::{Path, PathBuf}, collections::{BTreeMap, HashMap, HashSet},
};
use rayon::iter::{ParallelBridge, ParallelIterator, IntoParallelIterator};
use bed_utils::bed::{
    BEDLike, BedGraph, merge_sorted_bed_with, GenomicRange, io,
    tree::{GenomeRegions, SparseBinnedCoverage, BedTree}
};
use bigtools::{bbi::bigwigwrite::BigWigWrite, bed::bedparser::BedParser};
use futures::executor::ThreadPool;
use indicatif::{ProgressIterator, style::ProgressStyle, ParallelProgressIterator};
use tempfile::Builder;

impl<T> Exporter for T where T: SnapData {}

pub trait Exporter: SnapData {
    fn export_fragments<P: AsRef<Path>>(
        &self,
        barcodes: Option<&Vec<&str>>,
        group_by: &Vec<&str>,
        selections: Option<HashSet<&str>>,
        dir: P,
        prefix: &str,
        suffix:&str,
        compression: Option<Compression>,
        compression_level: Option<u32>,
    ) -> Result<HashMap<String, PathBuf>> {
        ensure!(self.n_obs() == group_by.len(), "lengths differ");
        let mut groups: HashSet<&str> = group_by.iter().map(|x| *x).unique().collect();
        if let Some(select) = selections { groups.retain(|x| select.contains(x)); }
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("cannot create directory: {}", dir.as_ref().display()))?;
        let files = groups.into_iter().map(|x| {
            let filename = dir.as_ref().join(
                prefix.to_string() + x.replace("/", "+").as_str() + suffix
            );
            let writer = utils::open_file_for_write(&filename, compression, compression_level)?;
            Ok((x, (filename, Arc::new(Mutex::new(writer)))))
        }).collect::<Result<HashMap<_, _>>>()?;

        let style = ProgressStyle::with_template("[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} (eta: {eta})")?;
        self.get_count_iter(1000)?.into_raw()
            .map(move |(vals, start, _)| {
                let mut ordered = HashMap::new();
                vals.into_iter().enumerate().for_each(|(i, xs)| {
                    let k = group_by[start + i];
                    let entry = ordered
                        .entry(k)
                        .or_insert_with(Vec::new);
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
            .try_for_each(|vals| vals.into_iter().par_bridge().try_for_each(|(k, beds)| {
                if let Some((_, fl)) = files.get(k) {
                    let mut fl = fl.lock().unwrap();
                    beds.into_iter().try_for_each(|x| writeln!(fl, "{}", x))?;
                }
                anyhow::Ok(())
            }))?;
        Ok(files.into_iter().map(|(k, (v, _))| (k.to_string(), v)).collect())
    }

    fn export_bedgraph<P: AsRef<Path> + std::marker::Sync>(
        &self,
        group_by: &Vec<&str>,
        selections: Option<HashSet<&str>>,
        resolution: usize,
        smooth_length: Option<u64>,
        blacklist_regions: Option<&BedTree<()>>,
        normalization: Option<Normalization>,
        ignore_for_norm: Option<&HashSet<&str>>,
        min_frag_length: Option<u64>,
        max_frag_length: Option<u64>,
        dir: P,
        prefix: &str,
        suffix:&str,
        compression: Option<Compression>,
        compression_level: Option<u32>,
        temp_dir: Option<P>,
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
            None, group_by, selections, temp_dir.path(), "", "", Some(Compression::Gzip), Some(1)
        )?;


        let chrom_sizes = self.read_chrom_sizes()?;
        info!("Export fragments as bedgraph files...");
        let style = ProgressStyle::with_template(
            "[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} (eta: {eta})"
        ).unwrap();
        fragment_files.into_iter().collect::<Vec<_>>().into_par_iter().map(|(grp, filename)| {
            let output = dir.as_ref().join(
                prefix.to_string() + grp.replace("/", "+").as_str() + suffix
            );
            let mut writer = utils::open_file_for_write(&output, compression, compression_level)?;

            // Make BedGraph
            create_bedgraph_from_fragments(
                io::Reader::new(utils::open_file_for_read(filename), None).into_records().map(Result::unwrap),
                &chrom_sizes,
                resolution as u64,
                smooth_length,
                blacklist_regions,
                normalization,
                ignore_for_norm,
                min_frag_length,
                max_frag_length,
            ).into_iter().for_each(|x| writeln!(writer, "{}", x).unwrap());

            Ok((grp.to_string(), output))
        }).progress_with_style(style).collect()
    }

    fn export_bigwig<P: AsRef<Path>>(
        &self,
        group_by: &Vec<&str>,
        selections: Option<HashSet<&str>>,
        resolution: usize,
        dir: P,
        prefix: &str,
        suffix:&str,
    ) -> Result<HashMap<String, PathBuf>> {
        export_insertions_as_bigwig(
            self.get_count_iter(500)?, group_by, selections, resolution, dir, prefix, suffix,
        )
    }
}

/// Export TN5 insertions as bigwig files
/// 
/// # Arguments
/// 
/// * `insertions` - TN5 insertion matrix
/// * `genome_index` - 
/// * `chrom_sizes` - 
fn export_insertions_as_bigwig<P, I>(
    mut coverage: GenomeCoverage<I>,
    group_by: &Vec<&str>,
    selections: Option<HashSet<&str>>,
    resolution: usize,
    dir: P,
    prefix: &str,
    suffix:&str,
) -> Result<HashMap<String, PathBuf>>
where
    I: ExactSizeIterator<Item = (CoverageType, usize, usize)>,
    P: AsRef<Path>,
{
    // Create directory
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("cannot create directory: {}", dir.as_ref().display()))?;

    let style = ProgressStyle::with_template(
        "[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} (eta: {eta})"
    ).unwrap();

    coverage = coverage.with_resolution(resolution);
    let index = coverage.get_gindex();
    let chrom_sizes: HashMap<String, u32> = index.chrom_sizes().map(|(k, v)| (k.to_string(), v as u32)).collect();

    // Collect groups
    let mut groups: HashSet<&str> = group_by.iter().map(|x| *x).unique().collect();
    if let Some(select) = selections { groups.retain(|x| select.contains(x)); }

    // Collect counts
    let mut counts: HashMap<&str, BTreeMap<usize, u32>> =
        groups.into_iter().map(|grp| (grp, BTreeMap::new())).collect();
    info!("Compute coverage for {} groups...", counts.len());
    coverage.into_values::<u32>().progress_with_style(style.clone()).for_each(|(csr, start, end)|
        (start..end).zip(csr.row_iter()).for_each(|(row_idx, row)|
            if let Some(count) = counts.get_mut(group_by[row_idx]) {
                row.col_indices().iter().zip(row.values()).for_each(|(i, v)|
                    *count.entry(*i).or_insert(0) += *v
                );
            }
        )
    );

    // Exporting
    info!("Exporting bigwig files...");
    counts.into_iter().progress_with_style(style).map(|(grp, count)| {
        let filename = dir.as_ref().join(
            prefix.to_string() + grp.replace("/", "+").as_str() + suffix
        );

        // compute normalization factor
        let total_count: f64 = count.values().map(|x| *x as f64).sum();
        let norm_factor = total_count * resolution as f64 / 1e9;

        // Make BedGraph
        let bedgraph: Vec<BedGraph<f32>> = count.into_iter().map(|(k, v)| {
            let mut region = index.get_region(k);
            region.set_end(region.start() + resolution as u64);
            BedGraph::from_bed(&region, (v as f64 / norm_factor) as f32)
        }).group_by(|x| x.value).into_iter().flat_map(|(_, groups)|
            merge_sorted_bed_with(
                groups,
                |beds| {
                    let mut iter = beds.into_iter();
                    let mut first = iter.next().unwrap();
                    if let Some(last) = iter.last() {
                        first.set_end(last.end());
                    }
                    first
                },
            )
        ).collect();

        create_bigwig_from_bedgraph(bedgraph, &chrom_sizes, filename.as_path())?;

        Ok((grp.to_string(), filename))
    }).collect()
}

#[derive(Debug, Clone, Copy)]
pub enum Normalization {
    RPKM,  // Reads per kilobase per million mapped reads. RPKM (per bin) =
           // number of reads per bin / (number of mapped reads (in millions) * bin length (kb)).
    CPM,   // Counts per million mapped reads. CPM (per bin) =
           // number of reads per bin / number of mapped reads (in millions). 
    BPM,   // Bins Per Million mapped reads, same as TPM in RNA-seq. BPM (per bin) =
           // number of reads per bin / sum of all reads per bin (in millions).
    RPGC,  // Reads per genomic content. RPGC (per bin) =
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
/// * `smooth_length` - Length of the smoothing window, in bases, for the output of the bigwig/bedgraph file.
///                     For example, if the bin_size is set to 20 and the smooth_length is set to 60,
///                     then, for each bin, the average of the bin and its left and right neighbors
///                     is considered. Any value smaller than bin_size will be ignored and no
///                     smoothing will be applied.
/// * `blacklist_regions` - Blacklist regions to be ignored.
/// * `normalization` - Normalization method.
/// * `ignore_for_norm` - Chromosomes to be ignored for normalization.
/// * `min_frag_length` - Minimum fragment length to be considered. No effect on single-end data.
/// * `max_frag_length` - Maximum fragment length to be considered. No effect on single-end data.
fn create_bedgraph_from_fragments<I>(
    fragments: I,
    chrom_sizes: &ChromSizes,
    bin_size: u64,
    smooth_length: Option<u64>,
    blacklist_regions: Option<&BedTree<()>>,
    normalization: Option<Normalization>,
    ignore_for_norm: Option<&HashSet<&str>>,
    min_frag_length: Option<u64>,
    max_frag_length: Option<u64>,
) -> Vec<BedGraph<f64>>
where
    I: Iterator<Item = Fragment>,
{
    let genome: GenomeRegions<GenomicRange> = chrom_sizes.into_iter()
        .map(|(k, v)| GenomicRange::new(k, 0, *v)).collect();
    let mut counter = SparseBinnedCoverage::new(&genome, bin_size);
    let mut total_count = 0.0;
    fragments.for_each(|frag| {
        let not_in_blacklist = blacklist_regions.map_or(true, |bl| !bl.is_overlapped(&frag));
        let is_single = frag.strand().is_some();
        let not_too_short = min_frag_length.map_or(true, |x| frag.len() >= x);
        let not_too_long = max_frag_length.map_or(true, |x| frag.len() <= x);
        if not_in_blacklist && (is_single || (not_too_short && not_too_long)) {
            if ignore_for_norm.map_or(true, |x| !x.contains(frag.chrom())) {
                total_count += 1.0;
            }
            counter.insert(&frag, 1.0);
        }
    });

    let norm_factor = match normalization {
        None => 1.0,
        Some(Normalization::RPKM) => total_count * bin_size as f64 / 1e9,
        Some(Normalization::CPM) => total_count / 1e6,
        Some(Normalization::BPM) => todo!(),
        Some(Normalization::RPGC) => todo!(),
    };
        
    let groups = counter.get_region_coverage().map(|(region, count)|
        BedGraph::from_bed(&region, count / norm_factor)
    ).group_by(|x| x.value);
    groups.into_iter().flat_map(|(_, groups)| merge_sorted_bed_with(
        groups,
        |beds| {
            let mut iter = beds.into_iter();
            let mut first = iter.next().unwrap();
            if let Some(last) = iter.last() {
                first.set_end(last.end());
            }
            first
        },
    )).collect()
}

/// Create a bigwig file from BedGraph records.
fn create_bigwig_from_bedgraph<P: AsRef<Path>>(
    mut bedgraph: Vec<BedGraph<f32>>,
    chrom_sizes: &HashMap<String, u32>,
    filename: P,
) -> Result<()> {
    // perform clipping to make sure the end of each region is within the range.
    bedgraph.iter_mut().group_by(|x| x.chrom().to_string()).into_iter().for_each(|(chr, groups)| {
        let size = *chrom_sizes.get(&chr).expect(&format!("chromosome not found: {}", chr)) as u64;
        let bed = groups.last().unwrap();
        if bed.end() > size {
            bed.set_end(size);
        }
    });

    // write to bigwig file
    BigWigWrite::create_file(filename.as_ref().to_str().unwrap().to_string()).write(
        chrom_sizes.clone(),
        bigtools::bbi::bedchromdata::BedParserStreamingIterator::new(
            BedParser::wrap_iter(bedgraph.into_iter().map(|x| {
                let val = bigtools::bbi::Value {
                    start: x.start() as u32,
                    end: x.end() as u32,
                    value: x.value,
                };
                let res: Result<_, bigtools::bed::bedparser::BedValueError> = Ok((x.chrom().to_string(), val));
                res
            })),
            false,
        ),
        ThreadPool::new().unwrap(),
    ).unwrap();
    Ok(())
}