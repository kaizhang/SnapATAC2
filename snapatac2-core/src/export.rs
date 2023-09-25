use crate::preprocessing::count_data::{SnapData, GenomeCoverage, CoverageType};

use anyhow::{Context, Result, ensure};
use flate2::{Compression, write::GzEncoder};
use itertools::Itertools;
use log::info;
use std::{
    fs::File, io::{BufReader, BufWriter, BufRead, Write},
    path::{Path, PathBuf}, collections::{BTreeMap, HashMap, HashSet}, process::Command,
};
use tempfile::Builder;
use rayon::iter::{ParallelIterator, IntoParallelIterator};
use which::which;
use bed_utils::bed::{BEDLike, BedGraph, merge_sorted_bed_with};
use bigtools::{bbi::bigwigwrite::BigWigWrite, bed::bedparser::BedParser};
use futures::executor::ThreadPool;
use indicatif::{ProgressIterator, style::ProgressStyle};

impl<T> Exporter for T where T: SnapData {}

pub trait Exporter: SnapData {
    fn export_bed<P: AsRef<Path>>(
        &self,
        barcodes: Option<&Vec<&str>>,
        group_by: &Vec<&str>,
        selections: Option<HashSet<&str>>,
        dir: P,
        prefix: &str,
        suffix:&str,
    ) -> Result<HashMap<String, PathBuf>> {
        ensure!(self.n_obs() == group_by.len(), "lengths differ");
        let mut groups: HashSet<&str> = group_by.iter().map(|x| *x).unique().collect();
        if let Some(select) = selections { groups.retain(|x| select.contains(x)); }
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("cannot create directory: {}", dir.as_ref().display()))?;
        let mut files = groups.into_iter().map(|x| {
            let filename = dir.as_ref().join(
                prefix.to_string() + x.replace("/", "+").as_str() + suffix
            );
            let f = BufWriter::with_capacity(
                1024*1024,
                File::create(&filename).with_context(|| format!("cannot create file: {}", filename.display()))?,
            );
            let e: Box<dyn Write> = if filename.extension().unwrap() == "gz" {
                Box::new(GzEncoder::new(f, Compression::default()))
            } else {
                Box::new(f)
            };
            Ok((x, (filename, e)))
        }).collect::<Result<HashMap<_, _>>>()?;

        let style = ProgressStyle::with_template("[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} (eta: {eta})")?;
        self.get_count_iter(500)?.into_raw().progress_with_style(style).try_for_each(|(beds, start, _)|
            beds.into_iter().enumerate().try_for_each::<_, Result<_>>(|(i, xs)| {
                if let Some((_, fl)) = files.get_mut(group_by[start + i]) {
                    xs.into_iter().try_for_each(|mut bed| {
                        if let Some(barcodes_) = barcodes {
                            bed.name = Some(barcodes_[start + i].to_string());
                        }
                        writeln!(fl, "{}", bed)
                    })?;
                }
                Ok(())
            })
        )?;
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

    fn call_peaks<P: AsRef<Path> + std::marker::Sync>(
        &self,
        q_value: f64,
        group_by: &Vec<&str>,
        selections: Option<HashSet<&str>>,
        dir: P,
        prefix: &str,
        suffix:&str,
        nolambda: bool,
        shift: i64,
        extension_size: i64,
    ) -> Result<HashMap<String, PathBuf>>
    {
        // Check if the command is in the PATH
        ensure!(
            which("macs2").is_ok(),
            "Cannot find macs2; please make sure macs2 has been installed"
        );

        std::fs::create_dir_all(&dir)?;
        info!("Exporting data...");
        let files = self.export_bed(None, group_by, selections, &dir, "", "_insertion.bed.gz")?;
        let genome_size = self.read_chrom_sizes()?.into_iter().map(|(_, v)| v).sum();
        info!("Calling peaks for {} groups ...", files.len());
        files.into_par_iter().map(|(key, fl)| {
            let out_file = dir.as_ref().join(
                prefix.to_string() + key.as_str().replace("/", "+").as_str() + suffix
            );
            macs2(fl, q_value, genome_size, nolambda, shift, extension_size, &dir, &out_file)?;
            Ok((key, out_file))
        }).collect()
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
        let mut bedgraph: Vec<BedGraph<f32>> = count.into_iter().map(|(k, v)| {
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

        // perform clipping to make sure the end of each region is within the range.
        bedgraph.iter_mut().group_by(|x| x.chrom().to_string()).into_iter().for_each(|(chr, groups)| {
            let size = *chrom_sizes.get(&chr).expect(&format!("chromosome not found: {}", chr)) as u64;
            let bed = groups.last().unwrap();
            if bed.end() > size {
                bed.set_end(size);
            }
        });

        // write to bigwig file
        BigWigWrite::create_file(filename.as_path().to_str().unwrap().to_string()).write(
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

        Ok((grp.to_string(), filename))
    }).collect()
}

/// Call peaks using macs2.
fn macs2<P1, P2, P3>(
    bed_file: P1,
    q_value: f64,
    genome_size: u64,
    nolambda: bool,
    shift: i64,
    extension_size: i64,
    tmp_dir: P2,
    out_file: P3,
) -> Result<()>
where
    P1: AsRef<Path>,
    P2: AsRef<Path>,
    P3: AsRef<Path>,
{
    let dir = Builder::new().tempdir_in(tmp_dir)?;

    let cmd_out = Command::new("macs2").args([
        "callpeak",
        "-f", "BED",
        "-t", bed_file.as_ref().to_str().unwrap(),
        "--keep-dup", "all",
        "--outdir", format!("{}", dir.path().display()).as_str(),
        "--qvalue", format!("{}", q_value).as_str(),
        "-g", format!("{}", (genome_size as f64 * 0.9).round()).as_str(),
        "--call-summits",
        "--nomodel",
        "--shift", format!("{}", shift).as_str(),
        "--extsize", format!("{}", extension_size).as_str(),
        if nolambda { "--nolambda" } else { "" },
        "--tempdir", format!("{}", dir.path().display()).as_str(),
    ]).output().context("failed to execute macs2 command")?;

    ensure!(
        cmd_out.status.success(),
        format!("macs2 error:\n{}", std::str::from_utf8(&cmd_out.stderr).unwrap()),
    );

    let peak_file = dir.path().join("NA_peaks.narrowPeak");
    let reader = BufReader::new(File::open(&peak_file)
        .context(format!("Cannot open file for read: {}", peak_file.display()))?
    );
    let mut writer: Box<dyn Write> = if out_file.as_ref().extension().unwrap() == "gz" {
        Box::new(BufWriter::new(GzEncoder::new(
            File::create(out_file)?,
            Compression::default(),
        )))
    } else {
        Box::new(BufWriter::new(File::create(out_file)?))
    };
    for x in reader.lines() {
        let x_ = x?;
        let mut strs: Vec<_> = x_.split("\t").collect();
        if strs[4].parse::<u64>().unwrap() > 1000 {
            strs[4] = "1000";
        }
        write!(writer, "{}\n", strs.into_iter().join("\t"))?;
    }
    Ok(())
}
 