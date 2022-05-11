use crate::utils::{ChromValues, FeatureCounter, ChromValuesReader};
use crate::utils::gene::{Promoters, PromoterCoverage, Transcript};

use anndata_rs::{
    anndata::AnnData,
    iterator::CsrIterator,
};
use polars::prelude::{NamedFrom, DataFrame, Series};
use anyhow::Result;
use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelIterator;

use bed_utils::bed::{
    GenomicRange,
    tree::{GenomeRegions, SparseCoverage, SparseBinnedCoverage},
};

/// Create cell by feature matrix, and compute qc matrix.
/// 
/// # Arguments
/// 
/// * `file` - 
/// * `fragments` -
/// * `promoter` -
/// * `region` -
/// * `bin_size` -
/// * `min_num_fragment` -
/// * `min_tsse` -
pub fn create_feat_matrix<C, I>(
    anndata: &AnnData,
    insertions: I,
    feature_counter: C,
    ) -> Result<()>
where
    C: FeatureCounter<Value = u32> + Clone + std::marker::Sync,
    I: Iterator<Item = Vec<ChromValues>>,
{
    let features = feature_counter.get_feature_ids();
    anndata.set_x_from_row_iter(CsrIterator {
        iterator: insertions.map(|chunk|
            chunk.into_par_iter().map(|ins| {
                let mut counter = feature_counter.clone();
                counter.inserts(ins);
                counter.get_counts()
            }).collect::<Vec<_>>()
        ),
        num_cols: features.len(),
    })?;

    anndata.set_var(Some(
        &DataFrame::new(vec![Series::new("Feature_ID", features)]).unwrap()
    ))?;

    Ok(())
}

/// Create cell by bin matrix, and compute qc matrix.
/// 
/// # Arguments
/// 
/// * `file` - 
/// * `fragments` -
/// * `promoter` -
/// * `region` -
/// * `bin_size` -
/// * `min_num_fragment` -
/// * `min_tsse` -
pub fn create_tile_matrix(
    anndata: &AnnData,
    bin_size: u64,
    chunk_size: usize,
    ) -> Result<()>
where
{
    let df: Box<DataFrame> = {
        anndata.get_uns().inner().get_mut("reference_sequences").unwrap()
            .read()?.into_any().downcast().unwrap()
    };
    let regions = df.column("reference_seq_length")
        .unwrap().u64().unwrap().into_iter()
        .zip(df.column("reference_seq_name").unwrap().utf8().unwrap())
        .map(|(s, chr)| GenomicRange::new(chr.unwrap(), 0, s.unwrap())).collect();
    let feature_counter: SparseBinnedCoverage<'_, _, u32> =
        SparseBinnedCoverage::new(&regions, bin_size);
    create_feat_matrix(
        anndata,
        anndata.read_insertions(chunk_size)?,
        feature_counter,
    )?;
    Ok(())
}

pub fn create_peak_matrix<I>(
    output: &str,
    fragments: I,
    peaks: &GenomeRegions<GenomicRange>,
    ) -> Result<AnnData>
where
    I: Iterator<Item = Vec<ChromValues>>,
{
    let mut anndata = AnnData::new(output, 0, 0)?;
    let feature_counter: SparseCoverage<'_, _, u32> = SparseCoverage::new(&peaks);
    create_feat_matrix(&mut anndata, fragments, feature_counter)?;
    Ok(anndata)
}

pub fn create_gene_matrix<I>(
    output: &str,
    fragments: I,
    transcripts: Vec<Transcript>,
    ) -> Result<AnnData>
where
    I: Iterator<Item = Vec<ChromValues>>,
{
    let mut anndata = AnnData::new(output, 0, 0)?;
    let promoters = Promoters::new(transcripts, 2000);
    let feature_counter: PromoterCoverage<'_> = PromoterCoverage::new(&promoters);
    create_feat_matrix(&mut anndata, fragments, feature_counter)?;
    Ok(anndata)
}