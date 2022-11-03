use anndata_rs::anndata::{AnnData, AnnDataSet};
use nalgebra_sparse::CsrMatrix;
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
use bigtools::{bigwig::bigwigwrite::BigWigWrite, bed::bedparser::BedParser};
use futures::executor::ThreadPool;
use indicatif::{ProgressIterator, style::ProgressStyle};

use crate::preprocessing::{ChromValues, ChromValuesReader, GenomeIndex, GBaseIndex};

pub trait Exporter: ChromValuesReader {
    fn export_bed<P: AsRef<Path>>(
        &self,
        barcodes: Option<&Vec<&str>>,
        group_by: &Vec<&str>,
        selections: Option<HashSet<&str>>,
        dir: P,
        prefix: &str,
        suffix:&str,
    ) -> Result<HashMap<String, PathBuf>>;

    fn export_bigwig<P: AsRef<Path>>(
        &self,
        group_by: &Vec<&str>,
        selections: Option<HashSet<&str>>,
        resolution: usize,
        dir: P,
        prefix: &str,
        suffix:&str,
    ) -> Result<HashMap<String, PathBuf>>;

    fn call_peaks<P: AsRef<Path> + std::marker::Sync>(
        &self,
        q_value: f64,
        group_by: &Vec<&str>,
        selections: Option<HashSet<&str>>,
        dir: P,
        prefix: &str,
        suffix:&str,
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
        let genome_size = self.get_reference_seq_info()?.into_iter().map(|(_, v)| v).sum();
        info!("Calling peaks for {} groups ...", files.len());
        files.into_par_iter().map(|(key, fl)| {
            let out_file = dir.as_ref().join(
                prefix.to_string() + key.as_str().replace("/", "+").as_str() + suffix
            );
            macs2(fl, q_value, genome_size, &dir, &out_file)?;
            Ok((key, out_file))
        }).collect()
    }
}

impl Exporter for AnnData {
    fn export_bed<P: AsRef<Path>>(
        &self,
        barcodes: Option<&Vec<&str>>,
        group_by: &Vec<&str>,
        selections: Option<HashSet<&str>>,
        dir: P,
        prefix: &str,
        suffix:&str,
    ) -> Result<HashMap<String, PathBuf>> {
        export_insertions_as_bed(
            &mut self.raw_count_iter(500)?,
            barcodes, group_by, selections, dir, prefix, suffix,
        )
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
        // Read genome information
        let chrom_sizes = self.get_reference_seq_info()?.into_iter()
            .map(|(a, b)| (a, b as u32)).collect();
        let genome_index = GBaseIndex::read_from_anndata(&mut self.get_uns().inner())?;

        let obsm = self.get_obsm().inner();
        let insertion_elem = obsm.get("insertion")
            .expect(".obsm does not contain key: insertion");
        export_insertions_as_bigwig(
            insertion_elem.chunked(500).map(|chunk|
                chunk.into_any().downcast::<CsrMatrix<u8>>().unwrap()
            ),
            &chrom_sizes, &genome_index, group_by,
            selections, resolution, dir, prefix, suffix,
        )
    }
}

impl Exporter for AnnDataSet {
    fn export_bed<P: AsRef<Path>>(
        &self,
        barcodes: Option<&Vec<&str>>,
        group_by: &Vec<&str>,
        selections: Option<HashSet<&str>>,
        dir: P,
        prefix: &str,
        suffix:&str,
    ) -> Result<HashMap<String, PathBuf>> {
        export_insertions_as_bed(
            &mut self.raw_count_iter(500)?,
            barcodes, group_by, selections, dir, prefix, suffix,
        )
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
        // Read genome information
        let chrom_sizes = self.get_reference_seq_info()?.into_iter()
            .map(|(a, b)| (a, b as u32)).collect();
        let genome_index = GBaseIndex::read_from_anndata(
            &mut self.anndatas.inner().iter().next().unwrap().1.get_uns().inner()
        )?;

        let adatas = self.anndatas.inner();
        let insertion_elem = adatas.obsm.data.get("insertion").unwrap();
        export_insertions_as_bigwig(
            insertion_elem.chunked(500).map(|chunk|
                chunk.into_any().downcast::<CsrMatrix<u8>>().unwrap()
            ),
            &chrom_sizes, &genome_index, group_by,
            selections, resolution, dir, prefix, suffix,
        )
    }
}


/// Export TN5 insertion sites to bed files with following fields:
///     1. chromosome
///     2. start
///     3. end (which is start + 1)
///     4. cell ID
fn export_insertions_as_bed<I, P>(
    insertions: &mut I,
    barcodes: Option<&Vec<&str>>,
    group_by: &Vec<&str>,
    selections: Option<HashSet<&str>>,
    dir: P,
    prefix: &str,
    suffix:&str,
) -> Result<HashMap<String, PathBuf>>
where
    I: Iterator<Item = Vec<ChromValues<u8>>> + ExactSizeIterator,
    P: AsRef<Path>,
{
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

    let style = ProgressStyle::with_template(
        "[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} (eta: {eta})"
    ).unwrap();
    let total_records = insertions.progress_with_style(style).try_fold::<_, _, Result<_>>(0, |accum, x| {
        let n_records = x.len();
        x.into_iter().enumerate().try_for_each::<_, Result<_>>(|(i, ins)| {
            if let Some((_, fl)) = files.get_mut(group_by[accum + i]) {
                ins.into_iter().try_for_each(|x| {
                    let bed = match barcodes {
                        None => format!("{}\t{}\t{}", x.chrom(), x.start(), x.end()),
                        Some(barcodes_) => format!("{}\t{}\t{}\t{}", x.chrom(), x.start(), x.end(), barcodes_[accum + i]),
                    };
                    (0..(x.value as usize)).try_for_each(|_| writeln!(fl, "{}", bed))
                })?;
            }
            Ok(())
        })?;
        Ok(accum + n_records)
    })?;
    ensure!(
        total_records == group_by.len(),
        "length of group differs",
    );
    Ok(files.into_iter().map(|(k, (v, _))| (k.to_string(), v)).collect())
}

/// Export TN5 insertions as bigwig files
/// 
/// # Arguments
/// 
/// * `insertions` - TN5 insertion matrix
/// * `genome_index` - 
/// * `chrom_sizes` - 
fn export_insertions_as_bigwig<P, I>(
    iter: I,
    chrom_sizes: &HashMap<String, u32>,
    genome_index: &GBaseIndex,
    group_by: &Vec<&str>,
    selections: Option<HashSet<&str>>,
    resolution: usize,
    dir: P,
    prefix: &str,
    suffix:&str,
) -> Result<HashMap<String, PathBuf>>
where
    I: Iterator<Item = Box<CsrMatrix<u8>>> + ExactSizeIterator,
    P: AsRef<Path>,
{
    // Create directory
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("cannot create directory: {}", dir.as_ref().display()))?;

    let style = ProgressStyle::with_template(
        "[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} (eta: {eta})"
    ).unwrap();

    // Collect groups
    let mut groups: HashSet<&str> = group_by.iter().map(|x| *x).unique().collect();
    if let Some(select) = selections { groups.retain(|x| select.contains(x)); }

    // Collect counts
    let mut counts: HashMap<&str, BTreeMap<usize, u32>> =
        groups.into_iter().map(|grp| (grp, BTreeMap::new())).collect();
    info!("Compute coverage for {} groups...", counts.len());
    let mut cur_row_idx = 0;
    iter.progress_with_style(style.clone()).for_each(|csr| csr.row_iter().for_each(|row| {
        if let Some(count) = counts.get_mut(group_by[cur_row_idx]) {
            row.col_indices().iter().zip(row.values()).for_each(|(i, v)| {
                let e = count.entry(
                    genome_index.index_downsampled(*i, resolution)
                ).or_insert(0);
                *e += *v as u32;
            })
        }
        cur_row_idx += 1;
    }));
    ensure!(cur_row_idx == group_by.len(), "the length of group differs");

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
        let mut bedgraph: Vec<BedGraph<f32>> = count.into_iter().map(move |(k, v)| {
            let mut region = genome_index.lookup_region(k);
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
            bigtools::bed::bedparser::BedParserStreamingIterator::new(
                BedParser::wrap_iter(bedgraph.into_iter().map(|x| {
                    let val = bigtools::bigwig::Value {
                        start: x.start() as u32,
                        end: x.end() as u32,
                        value: x.value,
                    };
                    Ok((x.chrom().to_string(), val))
                })),
                chrom_sizes.clone(),
                false,
            ),
            ThreadPool::new().unwrap(),
        ).unwrap();

        Ok((grp.to_string(), filename))
    }).collect()
}

fn macs2<P1, P2, P3>(
    bed_file: P1,
    q_value: f64,
    genome_size: u64,
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
        "--nomodel", "--shift", "-100", "--extsize", "200",
        "--nolambda",
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
        let line: String = strs.into_iter().intersperse("\t").collect();
        write!(writer, "{}\n", line)?;
    }
    Ok(())
}
 