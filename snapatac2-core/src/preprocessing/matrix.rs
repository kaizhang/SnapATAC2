use crate::preprocessing::genome::{Promoters, Transcript, SnapData};
use crate::preprocessing::counter::{FeatureCounter, TranscriptCount, GeneCount};

use anndata::AnnDataOp;
use indicatif::{ProgressIterator, ProgressStyle};
use polars::prelude::{NamedFrom, DataFrame, Series};
use anyhow::Result;
use bed_utils::bed::{BEDLike, tree::{GenomeRegions, SparseCoverage}};


/// Create cell by bin matrix.
/// 
/// # Arguments
/// 
/// * `anndata` - 
pub fn create_tile_matrix<A, B>(
    adata: &A,
    bin_size: usize,
    chunk_size: usize,
    out: Option<&B>,
    ) -> Result<()>
where
    A: SnapData,
    B: AnnDataOp,
{
    let style = ProgressStyle::with_template(
        "[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} (eta: {eta})"
    ).unwrap();
    let counts = adata.raw_count_iter(chunk_size)?.with_resolution(bin_size);
    let feature_names = counts.index.to_ranges().map(|x| x.pretty_show()).collect();
    let data_iter = counts.into_values::<u32>().map(|x| x.0).progress_with_style(style);
    if let Some(adata_out) =  out {
        adata_out.set_x_from_iter(data_iter)?;
        adata_out.set_var_names(feature_names)?;
    } else {
        adata.set_x_from_iter(data_iter)?;
        adata.set_var_names(feature_names)?;
    }
    Ok(())
}

pub fn create_peak_matrix<A, I, D, B>(
    adata: &A,
    peaks: I,
    chunk_size: usize,
    out: Option<&B>,
    ) -> Result<()>
where
    A: SnapData,
    I: Iterator<Item = D>,
    D: BEDLike + Send + Sync + Clone,
    B: AnnDataOp,
{
    let regions: GenomeRegions<D> = peaks.collect();
    let counter = SparseCoverage::new(&regions);
    let feature_names = counter.get_feature_ids();
    let data = adata.raw_count_iter(chunk_size)?;
    if let Some(adata_out) =  out {
        adata_out.set_x_from_iter(data.aggregate_by(counter).map(|x| x.0))?;
        adata_out.set_var_names(feature_names.into())?;
    } else {
        adata.set_x_from_iter(data.aggregate_by(counter).map(|x| x.0))?;
        adata.set_var_names(feature_names.into())?;
    }
    Ok(())
}

pub fn create_gene_matrix<A, B>(
    adata: &A,
    transcripts: Vec<Transcript>,
    id_type: &str, 
    chunk_size: usize,
    out: Option<&B>,
    ) -> Result<()>
where
    A: SnapData,
    B: AnnDataOp,
{
    let promoters = Promoters::new(transcripts, 2000, 0, true);
    let transcript_counter: TranscriptCount<'_> = TranscriptCount::new(&promoters);
    let data = adata.raw_count_iter(chunk_size)?;
    match id_type {
        "transcript" => {
            let gene_names: Vec<String> = transcript_counter.gene_names().iter().map(|x| x.clone()).collect();
            let ids = transcript_counter.get_feature_ids();
            if let Some(adata_out) = out {
                adata_out.set_x_from_iter(data.aggregate_by(transcript_counter).map(|x| x.0))?;
                adata_out.set_var_names(ids.into())?;
                adata_out.set_var(DataFrame::new(vec![Series::new("gene_name", gene_names)])?)?;
            } else {
                adata.set_x_from_iter(data.aggregate_by(transcript_counter).map(|x| x.0))?;
                adata.set_var_names(ids.into())?;
                adata.set_var(DataFrame::new(vec![Series::new("gene_name", gene_names)])?)?;
            }
        },
        "gene" => {
            let gene_counter: GeneCount<'_> = GeneCount::new(transcript_counter);
            let ids = gene_counter.get_feature_ids();
            if let Some(adata_out) = out {
                adata_out.set_x_from_iter(data.aggregate_by(gene_counter).map(|x| x.0))?;
                adata_out.set_var_names(ids.into())?;
            } else {
                adata.set_x_from_iter(data.aggregate_by(gene_counter).map(|x| x.0))?;
                adata.set_var_names(ids.into())?;
            }
        },
        _ => panic!("id_type must be 'transcript' or 'gene'"),
    }
    Ok(())
}