use crate::utils::{Insertions, read_insertions, read_insertions_from_anndataset};

use anndata_rs::anndata::{AnnData, AnnDataSet};
use anyhow::Result;
use flate2::Compression;
use flate2::write::GzEncoder;
use itertools::Itertools;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    collections::{HashMap, HashSet},
};

pub trait Exporter {
    fn export_bed<P: AsRef<Path> + Clone>(
        &self,
        barcodes: &Vec<&str>,
        group_by: &Vec<&str>,
        selections: Option<HashSet<&str>>,
        dir: P,
        prefix: &str,
        suffix:&str,
    ) -> Result<HashMap<String, PathBuf>>;
}

impl Exporter for AnnData {
    fn export_bed<P: AsRef<Path> + Clone>(
        &self,
        barcodes: &Vec<&str>,
        group_by: &Vec<&str>,
        selections: Option<HashSet<&str>>,
        dir: P,
        prefix: &str,
        suffix:&str,
    ) -> Result<HashMap<String, PathBuf>> {
        export_insertions_as_bed(
            &mut read_insertions(self)?,
            barcodes, group_by, selections, dir, prefix, suffix,
        )
    }
}

impl Exporter for AnnDataSet {
    fn export_bed<P: AsRef<Path> + Clone>(
        &self,
        barcodes: &Vec<&str>,
        group_by: &Vec<&str>,
        selections: Option<HashSet<&str>>,
        dir: P,
        prefix: &str,
        suffix:&str,
    ) -> Result<HashMap<String, PathBuf>> {
        export_insertions_as_bed(
            &mut read_insertions_from_anndataset(self)?,
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
    I: Iterator<Item = Vec<Insertions>>,
    P: AsRef<Path> + Clone,
{
    let mut groups: HashSet<&str> = group_by.iter().map(|x| *x).unique().collect();
    if let Some(select) = selections { groups.retain(|x| select.contains(x)); }
    std::fs::create_dir_all(dir.clone())?;
    let mut files = groups.into_iter().map(|x| {
        let filename = dir.as_ref().join(prefix.to_string() + x + suffix);
        let f = File::create(&filename)?;
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