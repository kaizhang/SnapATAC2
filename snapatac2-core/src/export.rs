use crate::{preprocessing::{count_data::{SnapData, GenomeCoverage, CoverageType, ChromSizes}, Fragment}, utils::open_file_for_write};

use anyhow::{Context, Result, ensure};
use itertools::Itertools;
use log::info;
use std::{
    sync::{Arc, Mutex}, io::Write,
    path::{Path, PathBuf}, collections::{BTreeMap, HashMap, HashSet},
};
use rayon::iter::{ParallelBridge, ParallelIterator};
use bed_utils::bed::{BEDLike, BedGraph, merge_sorted_bed_with, GenomicRange, tree::{GenomeRegions, SparseBinnedCoverage}};
use bigtools::{bbi::bigwigwrite::BigWigWrite, bed::bedparser::BedParser};
use futures::executor::ThreadPool;
use indicatif::{ProgressIterator, style::ProgressStyle};

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
        compression: Option<&str>,
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
            let writer = open_file_for_write(&filename, compression, compression_level)?;
            Ok((x, (filename, Arc::new(Mutex::new(writer)))))
        }).collect::<Result<HashMap<_, _>>>()?;

        let style = ProgressStyle::with_template("[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} (eta: {eta})")?;
        self.get_count_iter(1000)?.into_raw()
            .progress_with_style(style)
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
            .try_for_each(|vals| vals.into_iter().par_bridge().try_for_each(|(k, beds)| {
                if let Some((_, fl)) = files.get(k) {
                    let mut fl = fl.lock().unwrap();
                    beds.into_iter().try_for_each(|x| writeln!(fl, "{}", x))?;
                }
                anyhow::Ok(())
            }))?;
        Ok(files.into_iter().map(|(k, (v, _))| (k.to_string(), v)).collect())
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

    fn get_counts(
        &self,
        group_by: &Vec<&str>,
        selections: Option<HashSet<&str>>,
    ) -> Result<()> {
        todo!()
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

fn create_bedgraph_from_fragments<I>(fragments: I, chrom_sizes: &ChromSizes, bin_size: u64)
where
    I: Iterator<Item = Fragment>,
{
    let genome: GenomeRegions<GenomicRange> = chrom_sizes.into_iter()
        .map(|(k, v)| GenomicRange::new(k, 0, *v)).collect();
    let mut coverage = SparseBinnedCoverage::new(&genome, bin_size);
    fragments.for_each(|frag| frag.to_reads().into_iter().for_each(|read|
        coverage.insert(&read, 1.0)
    ));
    todo!()
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