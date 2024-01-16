use crate::preprocessing::count_data::{
    SnapData,
    FeatureCounter, TranscriptCount, GeneCount,
    Promoters, Transcript,
};

use anndata::{AnnDataOp, AxisArraysOp};
use indicatif::{ProgressIterator, ProgressStyle};
use polars::prelude::{NamedFrom, DataFrame, Series};
use anyhow::Result;
use bed_utils::bed::{BEDLike, tree::{GenomeRegions, SparseCoverage}};


/// Create cell by bin matrix.
/// 
/// # Arguments
/// 
/// * `adata` - The input anndata object.
/// * `bin_size` - The bin size.
/// * `chunk_size` - The chunk size.
/// * `exclude_chroms` - The chromosomes to exclude.
/// * `min_fragment_size` - The minimum fragment size.
/// * `max_fragment_size` - The maximum fragment size.
/// * `count_frag_as_reads` - Whether to treat fragments as reads during counting.
/// * `out` - The output anndata object.
pub fn create_tile_matrix<A, B>(
    adata: &A,
    bin_size: usize,
    chunk_size: usize,
    exclude_chroms: Option<&[&str]>,
    min_fragment_size: Option<u64>,
    max_fragment_size: Option<u64>,
    count_frag_as_reads: bool,
    out: Option<&B>,
    ) -> Result<()>
where
    A: SnapData,
    B: AnnDataOp,
{
    let style = ProgressStyle::with_template(
        "[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} (eta: {eta})"
    ).unwrap();

    if adata.obsm().keys().contains(&"contact".into()) {
        let data_iter = adata
            .contact_count_iter(chunk_size)?
            .with_resolution(bin_size)
            .into_values::<u32>().map(|x| x.0).progress_with_style(style);
        if let Some(adata_out) =  out {
            adata_out.set_x_from_iter(data_iter)?;
            adata_out.set_obs_names(adata.obs_names())?;
            adata_out.set_var_names(adata_out.n_vars().into())?;
        } else {
            adata.set_x_from_iter(data_iter)?;
            adata.set_var_names(adata.n_vars().into())?;
        }
    } else {
        let mut counts = adata
            .get_count_iter(chunk_size)?
            .with_resolution(bin_size)
            .count_fragment_as_reads(count_frag_as_reads);

        if let Some(exclude_chroms) = exclude_chroms {
            counts = counts.exclude(exclude_chroms);
        }
        if let Some(min_fragment_size) = min_fragment_size {
            counts = counts.min_fragment_size(min_fragment_size);
        }
        if let Some(max_fragment_size) = max_fragment_size {
            counts = counts.max_fragment_size(max_fragment_size);
        }

        let feature_names = counts.get_gindex().to_index().into();
        let data_iter = counts.into_counts::<u32>().map(|x| x.0).progress_with_style(style);
        if let Some(adata_out) =  out {
            adata_out.set_x_from_iter(data_iter)?;
            adata_out.set_obs_names(adata.obs_names())?;
            adata_out.set_var_names(feature_names)?;
        } else {
            adata.set_x_from_iter(data_iter)?;
            adata.set_var_names(feature_names)?;
        }
    }
    Ok(())
}

pub fn create_peak_matrix<A, I, D, B>(
    adata: &A,
    peaks: I,
    chunk_size: usize,
    count_frag_as_reads: bool,
    min_fragment_size: Option<u64>,
    max_fragment_size: Option<u64>,
    out: Option<&B>,
    use_x: bool,
    ) -> Result<()>
where
    A: SnapData,
    I: Iterator<Item = D>,
    D: BEDLike + Send + Sync + Clone,
    B: AnnDataOp,
{
    let style = ProgressStyle::with_template(
        "[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} (eta: {eta})"
    ).unwrap();
    let regions: GenomeRegions<D> = peaks.collect();
    let counter = SparseCoverage::new(&regions);
    let feature_names = counter.get_feature_ids();
    let data: Box<dyn ExactSizeIterator<Item = _>> = if use_x {
        Box::new(adata.read_chrom_values(chunk_size)?
            .aggregate_by(counter).map(|x| x.0))
    } else {
        let mut counts = adata.get_count_iter(chunk_size)?.count_fragment_as_reads(count_frag_as_reads);
        if let Some(min_fragment_size) = min_fragment_size {
            counts = counts.min_fragment_size(min_fragment_size);
        }
        if let Some(max_fragment_size) = max_fragment_size {
            counts = counts.max_fragment_size(max_fragment_size);
        }
        Box::new(counts.aggregate_counts_by(counter).map(|x| x.0))
    };
    if let Some(adata_out) =  out {
        adata_out.set_x_from_iter(data.progress_with_style(style))?;
        adata_out.set_obs_names(adata.obs_names())?;
        adata_out.set_var_names(feature_names.into())?;
    } else {
        adata.set_x_from_iter(data.progress_with_style(style))?;
        adata.set_var_names(feature_names.into())?;
    }
    Ok(())
}

pub fn create_gene_matrix<A, B>(
    adata: &A,
    transcripts: Vec<Transcript>,
    id_type: &str, 
    chunk_size: usize,
    count_frag_as_reads: bool,
    min_fragment_size: Option<u64>,
    max_fragment_size: Option<u64>,
    out: Option<&B>,
    use_x: bool,
    ) -> Result<()>
where
    A: SnapData,
    B: AnnDataOp,
{
    let promoters = Promoters::new(transcripts, 2000, 0, true);
    let transcript_counter: TranscriptCount<'_> = TranscriptCount::new(&promoters);
    match id_type {
        "transcript" => {
            let gene_names: Vec<String> = transcript_counter.gene_names().iter().map(|x| x.clone()).collect();
            let ids = transcript_counter.get_feature_ids();
            let data: Box<dyn ExactSizeIterator<Item = _>> = if use_x {
                Box::new(adata.read_chrom_values(chunk_size)?
                    .aggregate_by(transcript_counter).map(|x| x.0))
            } else {
                let mut counts = adata.get_count_iter(chunk_size)?.count_fragment_as_reads(count_frag_as_reads);
                if let Some(min_fragment_size) = min_fragment_size {
                    counts = counts.min_fragment_size(min_fragment_size);
                }
                if let Some(max_fragment_size) = max_fragment_size {
                    counts = counts.max_fragment_size(max_fragment_size);
                }
                Box::new(counts.aggregate_counts_by(transcript_counter).map(|x| x.0))
            };
            if let Some(adata_out) = out {
                adata_out.set_x_from_iter(data)?;
                adata_out.set_obs_names(adata.obs_names())?;
                adata_out.set_var_names(ids.into())?;
                adata_out.set_var(DataFrame::new(vec![Series::new("gene_name", gene_names)])?)?;
            } else {
                adata.set_x_from_iter(data)?;
                adata.set_var_names(ids.into())?;
                adata.set_var(DataFrame::new(vec![Series::new("gene_name", gene_names)])?)?;
            }
        },
        "gene" => {
            let gene_counter: GeneCount<'_> = GeneCount::new(transcript_counter);
            let ids = gene_counter.get_feature_ids();
            let data: Box<dyn ExactSizeIterator<Item = _>> = if use_x {
                Box::new(adata.read_chrom_values(chunk_size)?
                    .aggregate_by(gene_counter).map(|x| x.0))
            } else {
                let mut counts = adata.get_count_iter(chunk_size)?.count_fragment_as_reads(count_frag_as_reads);
                if let Some(min_fragment_size) = min_fragment_size {
                    counts = counts.min_fragment_size(min_fragment_size);
                }
                if let Some(max_fragment_size) = max_fragment_size {
                    counts = counts.max_fragment_size(max_fragment_size);
                }
                Box::new(counts.aggregate_counts_by(gene_counter).map(|x| x.0))
            };
            if let Some(adata_out) = out {
                adata_out.set_x_from_iter(data)?;
                adata_out.set_obs_names(adata.obs_names())?;
                adata_out.set_var_names(ids.into())?;
            } else {
                adata.set_x_from_iter(data)?;
                adata.set_var_names(ids.into())?;
            }
        },
        _ => panic!("id_type must be 'transcript' or 'gene'"),
    }
    Ok(())
}