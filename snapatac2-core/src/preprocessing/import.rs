use crate::feature_count::{BaseValue, ContactData, BASE_VALUE, FRAGMENT_PAIRED, FRAGMENT_SINGLE};
use crate::genome::{ChromSizes, GenomeBaseIndex};
use crate::preprocessing::qc::{Contact, Fragment, FragmentQC, FragmentQCBuilder};

use super::qc::BaseValueQC;
use anndata::data::DynCsrMatrix;
use anndata::{
    data::array::utils::{from_csr_data, to_csr_data},
    AnnDataOp, ArrayData, AxisArraysOp, ElemCollectionOp,
};
use anyhow::Result;
use bed_utils::bed::{map::GIntervalIndexSet, BEDLike, Strand};
use indexmap::IndexSet;
use indicatif::{style::ProgressStyle, ProgressBar, ProgressDrawTarget, ProgressIterator};
use itertools::Itertools;
use log::warn;
use nalgebra_sparse::CsrMatrix;
use polars::prelude::{Column, DataFrame, Series};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::collections::{BTreeMap, HashSet};

/// Import fragments
/// Fragments are reprensented as a sparse matrix with rows as barcodes and columns as genomic coordinates.
/// Each entry in the matrix encodes the size of the fragment.
/// Positive values indicate the start of the fragment and negative values indicate the end of the fragment.
/// For example:
/// chr1 2 5
/// will be encoded as:
/// X X 3 X -3 X X X X
/// Note the end coordinate is 5-1=4 as the end coordinate is exclusive.
pub fn import_fragments<A, I>(
    anndata: &A,
    fragments: I,
    mitochrondrial_dna: &HashSet<String>,
    chrom_sizes: &ChromSizes,
    white_list: Option<&HashSet<String>>,
    min_num_fragment: u64,
    chunk_size: usize,
) -> Result<()>
where
    A: AnnDataOp,
    I: Iterator<Item = Fragment>,
{
    let spinner = ProgressBar::with_draw_target(None, ProgressDrawTarget::stderr_with_hz(1))
        .with_style(
            ProgressStyle::with_template(
                "{spinner} Processed {human_pos} barcodes in {elapsed} ({per_sec}) ...",
            )
            .unwrap(),
        );
    let mut fragments = fragments.peekable();
    let is_paired = if let Some(f) = fragments.peek() {
        f.strand.is_none()
    } else {
        false
    };
    let obsm_key = if is_paired {
        FRAGMENT_PAIRED
    } else {
        FRAGMENT_SINGLE
    };

    let genome_index = GenomeBaseIndex::new(chrom_sizes);
    let mut saved_barcodes = Vec::new();
    let mut qc = Vec::new();

    let mut scanned_barcodes = HashSet::new();
    let frag_grouped = fragments
        .filter(|x| x.len() > 0)
        .chunk_by(|x| x.name().unwrap().to_string());
    let frag_chunked = frag_grouped
        .into_iter()
        .progress_with(spinner)
        .filter(|(key, _)| white_list.map_or(true, |x| x.contains(key)))
        .chunks(chunk_size);
    let mut arrays = frag_chunked
        .into_iter()
        .map(|chunk| {
            let data: Vec<(String, Vec<Fragment>)> =
                chunk.map(|(barcode, x)| (barcode, x.collect())).collect();
            if is_paired {
                make_arraydata::<u32>(
                    data,
                    mitochrondrial_dna,
                    &genome_index,
                    min_num_fragment,
                    &mut scanned_barcodes,
                    &mut saved_barcodes,
                    &mut qc,
                )
            } else {
                make_arraydata::<i32>(
                    data,
                    mitochrondrial_dna,
                    &genome_index,
                    min_num_fragment,
                    &mut scanned_barcodes,
                    &mut saved_barcodes,
                    &mut qc,
                )
            }
        })
        .peekable();
    if arrays.peek().is_some() {
        anndata.obsm().add_iter(obsm_key, arrays)?;
        anndata
            .uns()
            .add("reference_sequences", chrom_sizes.to_dataframe())?;
        anndata.set_obs_names(saved_barcodes.into())?;
        anndata.set_obs(qc_to_df(qc))?;
    } else {
        warn!("No barcodes passed the QC filter. No data is imported.");
    }
    Ok(())
}

fn make_arraydata<V>(
    data: Vec<(String, Vec<Fragment>)>,
    mitochrondrial_dna: &HashSet<String>,
    genome_index: &GenomeBaseIndex,
    min_num_fragment: u64,
    scanned_barcodes: &mut HashSet<String>,
    saved_barcodes: &mut Vec<String>,
    qc: &mut Vec<FragmentQC>,
) -> ArrayData
where
    V: TryFrom<i64> + Ord + std::marker::Send,
    ArrayData: From<anndata::data::CsrNonCanonical<V>>,
    ArrayData: From<nalgebra_sparse::CsrMatrix<V>>,
    <V as TryFrom<i64>>::Error: std::fmt::Debug,
{
    let num_features = genome_index.len();
    let result: Vec<_> = data
        .into_par_iter()
        .map(|(barcode, x)| {
            (
                barcode,
                count_fragments::<V>(mitochrondrial_dna, &genome_index, x),
            )
        })
        .collect();
    let counts = result
        .into_iter()
        .filter_map(|(barcode, (q, values))| {
            if !scanned_barcodes.insert(barcode.clone()) {
                panic!("Please sort fragment file by barcodes");
            }
            if q.num_unique_fragment < min_num_fragment {
                return None;
            } else {
                saved_barcodes.push(barcode);
                qc.push(q);
                Some(values)
            }
        })
        .collect::<Vec<_>>();
    let (r, c, offset, ind, data) = to_csr_data(counts, num_features);
    from_csr_data(r, c, offset, ind, data).unwrap()
}

fn count_fragments<V>(
    mitochrondrial_dna: &HashSet<String>,
    genome_index: &GenomeBaseIndex,
    fragments: Vec<Fragment>,
) -> (FragmentQC, Vec<(usize, V)>)
where
    V: TryFrom<i64> + Ord,
    <V as TryFrom<i64>>::Error: std::fmt::Debug,
{
    let mut qc = FragmentQCBuilder::new(mitochrondrial_dna);
    let mut values = Vec::new();
    fragments.into_iter().for_each(|f| {
        let chrom = &f.chrom;
        if genome_index.contain_chrom(chrom) {
            qc.update(&f);
            let start = f.start as i64;
            let end = f.end as i64;
            let size = end - start;
            let pos;
            let shift: V;
            match f.strand {
                Some(Strand::Reverse) => {
                    pos = genome_index.get_position_rev(chrom, (end - 1) as u64);
                    shift = (-size).try_into().expect(
                        format!(
                            "cannot convert size {} to {}",
                            -size,
                            std::any::type_name::<V>()
                        )
                        .as_str(),
                    );
                }
                _ => {
                    pos = genome_index.get_position_rev(chrom, start as u64);
                    shift = size.try_into().expect(
                        format!(
                            "cannot convert size {} to {}",
                            size,
                            std::any::type_name::<V>()
                        )
                        .as_str(),
                    );
                }
            }
            values.push((pos, shift));
        }
    });
    values.sort();
    (qc.finish(), values)
}

fn qc_to_df(qc: Vec<FragmentQC>) -> DataFrame {
    DataFrame::new(vec![
        Column::new(
            "n_fragment".into(),
            qc.iter().map(|x| x.num_unique_fragment).collect::<Series>(),
        ),
        Column::new(
            "frac_dup".into(),
            qc.iter().map(|x| x.frac_duplicated).collect::<Series>(),
        ),
        Column::new(
            "frac_mito".into(),
            qc.iter().map(|x| x.frac_mitochondrial).collect::<Series>(),
        ),
    ])
    .unwrap()
}

/// Import scHi-C contacts into AnnData
pub fn import_contacts<A, I>(
    anndata: &A,
    contacts: I,
    regions: &GIntervalIndexSet,
    bin_size: usize,
    chunk_size: usize,
) -> Result<()>
where
    A: AnnDataOp,
    I: Iterator<Item = Contact>,
{
    let chrom_sizes: ChromSizes = regions.iter().map(|x| (x.chrom(), x.end())).collect();

    let genome_index = GenomeBaseIndex::new(&chrom_sizes);
    let genome_size = genome_index.len();

    let spinner = ProgressBar::with_draw_target(None, ProgressDrawTarget::stderr_with_hz(1))
        .with_style(
            ProgressStyle::with_template(
                "{spinner} Processed {human_pos} barcodes in {elapsed} ({per_sec}) ...",
            )
            .unwrap(),
        );
    let mut scanned_barcodes = IndexSet::new();
    let binding = contacts.chunk_by(|x| x.barcode.clone());
    let binding2 = binding
        .into_iter()
        .progress_with(spinner)
        .chunks(chunk_size);
    let binding3 = binding2.into_iter().map(|chunk| {
        let data: Vec<Vec<Contact>> = chunk
            .map(|(barcode, x)| {
                if !scanned_barcodes.insert(barcode.clone()) {
                    panic!("Please sort fragment file by barcodes");
                }
                x.collect()
            })
            .collect();

        let counts: Vec<_> = data
            .into_par_iter()
            .map(|x| {
                let mut count = BTreeMap::new();
                x.into_iter().for_each(|c| {
                    if genome_index.contain_chrom(&c.chrom1)
                        && genome_index.contain_chrom(&c.chrom2)
                    {
                        let pos1 = genome_index.get_position_rev(&c.chrom1, c.start1);
                        let pos2 = genome_index.get_position_rev(&c.chrom2, c.start2);
                        let i = pos1 * genome_size + pos2;
                        count
                            .entry(i)
                            .and_modify(|x| *x += c.count)
                            .or_insert(c.count);
                    }
                });
                count.into_iter().collect::<Vec<_>>()
            })
            .collect();

        let (r, c, offset, ind, data) = to_csr_data(counts, genome_size * genome_size);
        CsrMatrix::try_from_csr_data(r, c, offset, ind, data).unwrap()
    });
    let contact_map = ContactData::new(chrom_sizes, binding3).with_resolution(bin_size);

    anndata.set_x_from_iter(contact_map.into_values::<u32>())?;
    anndata.set_var_names(anndata.n_vars().into())?;

    anndata.uns().add(
        "reference_sequences",
        DataFrame::new(vec![
            Column::new(
                "reference_seq_name".into(),
                regions.iter().map(|x| x.chrom()).collect::<Series>(),
            ),
            Column::new(
                "reference_seq_length".into(),
                regions.iter().map(|x| x.end()).collect::<Series>(),
            ),
        ])?,
    )?;
    anndata.set_obs_names(scanned_barcodes.into_iter().collect())?;
    Ok(())
}

/// Import values
pub fn import_values<A, I>(
    anndata: &A,
    values: I,
    chrom_sizes: &ChromSizes,
    white_list: Option<&HashSet<String>>,
    chunk_size: usize,
) -> Result<()>
where
    A: AnnDataOp,
    I: Iterator<Item = (String, BaseValue)>,
{
    fn helper<T: TryFrom<BaseValue, Error = anyhow::Error> + Send>(
        chunk: Vec<Vec<BaseValue>>,
        genome_index: &GenomeBaseIndex,
        genome_size: usize,
    ) -> (Vec<BaseValueQC>, CsrMatrix<T>) {
        let (qc, counts): (Vec<_>, Vec<_>) = chunk
            .into_par_iter()
            .map(|cell_data| {
                let mut qc = BaseValueQC::new();
                let mut count = cell_data
                    .into_iter()
                    .flat_map(|value| {
                        let chrom = &value.chrom;
                        if genome_index.contain_chrom(chrom) {
                            qc.add();
                            let pos = genome_index.get_position_rev(chrom, value.pos);
                            Some((pos, T::try_from(value).unwrap()))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                count.sort_by(|x, y| x.0.cmp(&y.0));
                (qc, count)
            })
            .unzip();
        let (r, c, offset, ind, csr_data) = to_csr_data(counts, genome_size);
        (
            qc,
            CsrMatrix::try_from_csr_data(r, c, offset, ind, csr_data).unwrap(),
        )
    }

    let spinner = ProgressBar::with_draw_target(None, ProgressDrawTarget::stderr_with_hz(1))
        .with_style(
            ProgressStyle::with_template(
                "{spinner} Processed {human_pos} barcodes in {elapsed} ({per_sec}) ...",
            )
            .unwrap(),
        );

    let genome_index = GenomeBaseIndex::new(chrom_sizes);
    let genome_size = genome_index.len();
    let mut scanned_barcodes = IndexSet::new();

    let mut qc_metrics = Vec::new();
    let chunked_values = values.chunk_by(|x| x.0.clone());
    let chunked_values = chunked_values
        .into_iter()
        .filter(|(key, _)| white_list.map_or(true, |x| x.contains(key)))
        .progress_with(spinner)
        .chunks(chunk_size);
    let arrays = chunked_values.into_iter().map(|chunk| {
        // Collect into vector for parallel processing
        let chunk: Vec<Vec<_>> = chunk
            .map(|(barcode, data)| {
                if !scanned_barcodes.insert(barcode.clone()) {
                    panic!("Please sort fragment file by barcodes");
                }
                data.map(|x| x.1).collect()
            })
            .collect();
        let mat: DynCsrMatrix = if chunk[0][0].to_i32().is_some() {
            let (qc, mat) = helper::<i32>(chunk, &genome_index, genome_size);
            qc_metrics.extend(qc);
            mat.into()
        } else {
            let (qc, mat) = helper::<f32>(chunk, &genome_index, genome_size);
            qc_metrics.extend(qc);
            mat.into()
        };
        mat
    });

    anndata.obsm().add_iter(BASE_VALUE, arrays)?;
    anndata
        .uns()
        .add("reference_sequences", chrom_sizes.to_dataframe())?;
    anndata.set_obs_names(scanned_barcodes.into_iter().collect())?;
    anndata.set_obs(DataFrame::new(vec![Column::new(
        "num_values".into(),
        qc_metrics.iter().map(|x| x.num_values).collect::<Series>(),
    )])?)?;
    Ok(())
}
