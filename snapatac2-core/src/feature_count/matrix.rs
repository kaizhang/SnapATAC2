use super::counter::{CountingStrategy, FeatureCounter, GeneCount, RegionCounter, TranscriptCount};
use super::ValueType;
use crate::genome::{Promoters, Transcript};
use crate::feature_count::SnapData;
use crate::preprocessing::SummaryType;

use anndata::{data::DataFrameIndex, AnnDataOp, ArrayData};
use anyhow::{bail, Result};
use bed_utils::bed::{map::GIntervalIndexSet, BEDLike};
use indicatif::{ProgressIterator, ProgressStyle};
use polars::prelude::{DataFrame, NamedFrom, Series};

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
/// * `val_type` - Which kind of value to use: numerator, denominator or ratio. Only used for base data.
/// * `out` - The output anndata object.
pub fn create_tile_matrix<A, B>(
    adata: &A,
    bin_size: usize,
    chunk_size: usize,
    exclude_chroms: Option<&[&str]>,
    min_fragment_size: Option<u64>,
    max_fragment_size: Option<u64>,
    counting_strategy: CountingStrategy,
    val_type: ValueType,
    summary_type: SummaryType,
    out: Option<&B>,
) -> Result<()>
where
    A: SnapData,
    B: AnnDataOp,
{
    let style = ProgressStyle::with_template(
        "[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} (eta: {eta})",
    )
    .unwrap();

    let data_iter: Box<dyn ExactSizeIterator<Item = ArrayData>>;
    let feature_names: DataFrameIndex;

    if let Ok(mut fragments) = adata.get_fragment_iter(chunk_size) {
        fragments = fragments
            .with_resolution(bin_size)
            .set_counting_strategy(counting_strategy);

        if let Some(exclude_chroms) = exclude_chroms {
            fragments = fragments.exclude(exclude_chroms);
        }
        if let Some(min_fragment_size) = min_fragment_size {
            fragments = fragments.min_fragment_size(min_fragment_size);
        }
        if let Some(max_fragment_size) = max_fragment_size {
            fragments = fragments.max_fragment_size(max_fragment_size);
        }

        feature_names = fragments.get_gindex().to_index().into();
        data_iter = Box::new(fragments.into_array_iter().map(|x| ArrayData::from(x.0)));
    } else if let Ok(mut values) = adata.get_base_iter(chunk_size) {
        values = values.with_resolution(bin_size);

        if let Some(exclude_chroms) = exclude_chroms {
            values = values.exclude(exclude_chroms);
        }

        feature_names = values.get_gindex().to_index().into();
        data_iter = Box::new(values.into_array_iter(val_type, summary_type).map(|x| ArrayData::from(x.0)));
    } else {
        bail!("No fragment or base data found in the anndata object");
    };

    let n_feat = feature_names.len();
    let data_iter = data_iter.progress_with_style(style);
    if let Some(adata_out) = out {
        adata_out.set_n_vars(n_feat)?;
        adata_out.set_x_from_iter(data_iter)?;
        adata_out.set_obs_names(adata.obs_names())?;
        adata_out.set_var_names(feature_names)?;
    } else {
        adata.set_n_vars(n_feat)?;
        adata.set_x_from_iter(data_iter)?;
        adata.set_var_names(feature_names)?;
    }
    Ok(())
}

pub fn create_peak_matrix<A, I, D, B>(
    adata: &A,
    peaks: I,
    chunk_size: usize,
    counting_strategy: CountingStrategy,
    val_type: ValueType,
    summary_type: SummaryType,
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
        "[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} (eta: {eta})",
    )
    .unwrap();
    let regions: GIntervalIndexSet = peaks.collect();

    let data_iter: Box<dyn ExactSizeIterator<Item = ArrayData>>;
    let feature_names: Vec<String>;

    if use_x {
        let counter = RegionCounter::new(&regions);
        feature_names = counter.get_feature_ids();
        let data = adata
            .read_chrom_values(chunk_size)?
            .aggregate_by(counter)
            .map(|x| x.0.into());
        data_iter = Box::new(data);
    } else if let Ok(mut fragments) = adata.get_fragment_iter(chunk_size) {
        let counter = RegionCounter::new(&regions);
        feature_names = counter.get_feature_ids();
        fragments = fragments.set_counting_strategy(counting_strategy);
        if let Some(min_fragment_size) = min_fragment_size {
            fragments = fragments.min_fragment_size(min_fragment_size);
        }
        if let Some(max_fragment_size) = max_fragment_size {
            fragments = fragments.max_fragment_size(max_fragment_size);
        }
        data_iter = Box::new(
            fragments
                .into_aggregated_array_iter(counter)
                .map(|x| x.0.into()),
        );
    } else if let Ok(values) = adata.get_base_iter(chunk_size) {
        let counter = RegionCounter::new(&regions);
        feature_names = counter.get_feature_ids();
        data_iter = Box::new(
            values
                .into_aggregated_array_iter(counter, val_type, summary_type)
                .map(|x| x.0.into()),
        );
    } else {
        bail!("No fragment data found in the anndata object");
    }

    let n_feat = feature_names.len();
    let data_iter = data_iter.progress_with_style(style);
    if let Some(adata_out) = out {
        adata_out.set_n_vars(n_feat)?;
        adata_out.set_x_from_iter(data_iter)?;
        adata_out.set_obs_names(adata.obs_names())?;
        adata_out.set_var_names(feature_names.into())?;
    } else {
        adata.set_n_vars(n_feat)?;
        adata.set_x_from_iter(data_iter)?;
        adata.set_var_names(feature_names.into())?;
    }

    Ok(())
}

pub fn create_gene_matrix<A, B>(
    adata: &A,
    transcripts: Vec<Transcript>,
    id_type: &str,
    upstream: u64,
    downstream: u64,
    include_gene_body: bool,
    chunk_size: usize,
    counting_strategy: CountingStrategy,
    min_fragment_size: Option<u64>,
    max_fragment_size: Option<u64>,
    out: Option<&B>,
    use_x: bool,
) -> Result<()>
where
    A: SnapData,
    B: AnnDataOp,
{
    let promoters = Promoters::new(transcripts, upstream, downstream, include_gene_body);
    let transcript_counter: TranscriptCount<'_> = TranscriptCount::new(&promoters);
    match id_type {
        "transcript" => {
            let gene_names: Vec<String> = transcript_counter
                .gene_names()
                .iter()
                .map(|x| x.clone())
                .collect();
            let ids = transcript_counter.get_feature_ids();
            let data: Box<dyn ExactSizeIterator<Item = _>> = if use_x {
                Box::new(
                    adata
                        .read_chrom_values(chunk_size)?
                        .aggregate_by(transcript_counter)
                        .map(|x| x.0),
                )
            } else {
                let mut fragments = adata
                    .get_fragment_iter(chunk_size)?
                    .set_counting_strategy(counting_strategy);
                if let Some(min_fragment_size) = min_fragment_size {
                    fragments = fragments.min_fragment_size(min_fragment_size);
                }
                if let Some(max_fragment_size) = max_fragment_size {
                    fragments = fragments.max_fragment_size(max_fragment_size);
                }
                Box::new(
                    fragments
                        .into_aggregated_array_iter(transcript_counter)
                        .map(|x| x.0),
                )
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
        }
        "gene" => {
            let gene_counter: GeneCount<'_> = GeneCount::new(transcript_counter);
            let ids = gene_counter.get_feature_ids();
            let data: Box<dyn ExactSizeIterator<Item = _>> = if use_x {
                Box::new(
                    adata
                        .read_chrom_values(chunk_size)?
                        .aggregate_by(gene_counter)
                        .map(|x| x.0),
                )
            } else {
                let mut fragments = adata
                    .get_fragment_iter(chunk_size)?
                    .set_counting_strategy(counting_strategy);
                if let Some(min_fragment_size) = min_fragment_size {
                    fragments = fragments.min_fragment_size(min_fragment_size);
                }
                if let Some(max_fragment_size) = max_fragment_size {
                    fragments = fragments.max_fragment_size(max_fragment_size);
                }
                Box::new(
                    fragments
                        .into_aggregated_array_iter(gene_counter)
                        .map(|x| x.0),
                )
            };
            if let Some(adata_out) = out {
                adata_out.set_x_from_iter(data)?;
                adata_out.set_obs_names(adata.obs_names())?;
                adata_out.set_var_names(ids.into())?;
            } else {
                adata.set_x_from_iter(data)?;
                adata.set_var_names(ids.into())?;
            }
        }
        _ => panic!("id_type must be 'transcript' or 'gene'"),
    }
    Ok(())
}
