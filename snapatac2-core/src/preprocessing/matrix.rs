use crate::utils::{ChromValues, FeatureCounter, ChromValuesReader};
use crate::utils::gene::{Promoters, TranscriptCount, GeneCount, Transcript};

use anndata_rs::{
    anndata::AnnData,
    iterator::CsrIterator,
};
use polars::prelude::{NamedFrom, DataFrame, Series};
use anyhow::Result;
use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelIterator;
use indicatif::ProgressIterator;
use indicatif::style::ProgressStyle;

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
    I: Iterator<Item = Vec<ChromValues>> + ExactSizeIterator,
{
    let features = feature_counter.get_feature_ids();
    let style = ProgressStyle::with_template(
        "[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} (eta: {eta})"
    ).unwrap();
    anndata.set_x_from_row_iter(CsrIterator {
        iterator: insertions.progress_with_style(style).map(|chunk|
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
        anndata.get_uns().inner().get_mut("reference_sequences")
            .expect("No reference sequence information is available in the anndata object")
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
        anndata.raw_count_iter(chunk_size)?,
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
    I: Iterator<Item = Vec<ChromValues>> + ExactSizeIterator,
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
    id_type: &str, 
    ) -> Result<AnnData>
where
    I: Iterator<Item = Vec<ChromValues>> + ExactSizeIterator,
{
    let mut anndata = AnnData::new(output, 0, 0)?;
    let promoters = Promoters::new(transcripts, 2000, 0, true);

    match id_type {
        "transcript" => {
            let feature_counter: TranscriptCount<'_> = TranscriptCount::new(&promoters);
            let gene_names: Vec<String> = feature_counter.gene_names().iter()
                .map(|x| x.clone()).collect();
            create_feat_matrix(&mut anndata, fragments, feature_counter)?;
            let mut var = anndata.get_var().read()?;
            var.insert_at_idx(1, Series::new("gene_name", gene_names))?;
            anndata.set_var(Some(&var))?;
        },
        "gene" => {
            let feature_counter: GeneCount<'_> = GeneCount::new(
                TranscriptCount::new(&promoters)
            );
            create_feat_matrix(&mut anndata, fragments, feature_counter)?;
        },
        _ => panic!("id_type must be 'transcript' or 'gene'"),
    }

    Ok(anndata)
}