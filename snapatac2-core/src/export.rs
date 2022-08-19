use crate::utils::{ChromValues, ChromValuesReader};

use anndata_rs::anndata::{AnnData, AnnDataSet};
use anyhow::{Context, Result, ensure};
use flate2::Compression;
use flate2::write::GzEncoder;
use itertools::Itertools;
use std::{
    fs::File,
    io::{BufReader, BufWriter, BufRead, Write},
    path::{Path, PathBuf},
    collections::{HashMap, HashSet},
    process::Command,
};
use tempfile::Builder;
use rayon::iter::{ParallelIterator, IntoParallelIterator};
use which::which;

pub trait Exporter: ChromValuesReader {
    fn export_bed<P: AsRef<Path>>(
        &self,
        barcodes: &Vec<&str>,
        group_by: &Vec<&str>,
        selections: Option<HashSet<&str>>,
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
        let tmp_dir = Builder::new().tempdir_in(&dir)
            .context("failed to create tmperorary directory")?;

        eprintln!("preparing input...");
        let files = self.export_bed(
            group_by, group_by, selections, &tmp_dir, "", ".bed.gz"
        ).with_context(|| format!("cannot save bed file to {}", tmp_dir.path().display()))?;
        let genome_size = self.get_reference_seq_info()?.into_iter().map(|(_, v)| v).sum();
        eprintln!("calling peaks for {} groups...", files.len());
        files.into_par_iter().map(|(key, fl)| {
            let out_file = dir.as_ref().join(prefix.to_string() + &key.as_str().replace("/", "+") + suffix);
            macs2(fl, q_value, genome_size, &tmp_dir, &out_file)?;
            eprintln!("group {}: done!", key);
            Ok((key, out_file))
        }).collect()
    }
}

impl Exporter for AnnData {
    fn export_bed<P: AsRef<Path>>(
        &self,
        barcodes: &Vec<&str>,
        group_by: &Vec<&str>,
        selections: Option<HashSet<&str>>,
        dir: P,
        prefix: &str,
        suffix:&str,
    ) -> Result<HashMap<String, PathBuf>> {
        export_insertions_as_bed(
            &mut self.read_insertions(500)?,
            barcodes, group_by, selections, dir, prefix, suffix,
        )
    }
}

impl Exporter for AnnDataSet {
    fn export_bed<P: AsRef<Path>>(
        &self,
        barcodes: &Vec<&str>,
        group_by: &Vec<&str>,
        selections: Option<HashSet<&str>>,
        dir: P,
        prefix: &str,
        suffix:&str,
    ) -> Result<HashMap<String, PathBuf>> {
        export_insertions_as_bed(
            &mut self.read_insertions(500)?,
            barcodes, group_by, selections, dir, prefix, suffix,
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
    barcodes: &Vec<&str>,
    group_by: &Vec<&str>,
    selections: Option<HashSet<&str>>,
    dir: P,
    prefix: &str,
    suffix:&str,
) -> Result<HashMap<String, PathBuf>>
where
    I: Iterator<Item = Vec<ChromValues>>,
    P: AsRef<Path>,
{
    let mut groups: HashSet<&str> = group_by.iter().map(|x| *x).unique().collect();
    if let Some(select) = selections { groups.retain(|x| select.contains(x)); }
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("cannot create directory: {}", dir.as_ref().display()))?;
    let mut files = groups.into_iter().map(|x| {
        let filename = dir.as_ref().join(prefix.to_string() + &x.replace("/", "+") + suffix);
        let f = File::create(&filename)
            .with_context(|| format!("cannot create file: {}", filename.display()))?;
        let e: Box<dyn Write> = if filename.ends_with(".gz") {
            Box::new(GzEncoder::new(BufWriter::new(f), Compression::default()))
        } else {
            Box::new(BufWriter::new(f))
        };
        Ok((x, (filename, e)))
    }).collect::<Result<HashMap<_, _>>>()?;

    insertions.try_fold::<_, _, Result<_>>(0, |accum, x| {
        let n_records = x.len();
        x.into_iter().enumerate().try_for_each::<_, Result<_>>(|(i, ins)| {
            if let Some((_, fl)) = files.get_mut(group_by[accum + i]) {
                let bc = barcodes[accum + i];
                ins.to_bed(bc).try_for_each(|o| writeln!(fl, "{}", o))?;
            }
            Ok(())
        })?;
        Ok(accum + n_records)
    })?;
    Ok(files.into_iter().map(|(k, (v, _))| (k.to_string(), v)).collect())
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

    Command::new("macs2").args([
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
    ]).output().context("macs2 command did not exit properly")?;

    let reader = BufReader::new(File::open(
        dir.path().join("NA_peaks.narrowPeak"))
            .context("NA_peaks.narrowPeak: cannot find the peak file")?
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
 
/*
fn export_insertions_as_bigwig<I, P>(
    insertions: &mut I,
    group_by: &Vec<&str>,
    selections: Option<HashSet<&str>>,
    dir: P,
    prefix: &str,
    suffix:&str,
) -> Result<HashMap<String, PathBuf>>
where
    I: Iterator<Item = Vec<ChromValues>>,
    P: AsRef<Path>,
{
    todo!()
}
*/

