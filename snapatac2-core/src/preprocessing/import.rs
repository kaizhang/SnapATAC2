use crate::{
    preprocessing::{
        qc::{compute_qc_count, Fragment, Contact, FragmentSummary, QualityControl},
        genome::{ChromSizes, GenomeBaseIndex},
    },
    utils::from_csr_rows,
};

use anndata::{AnnDataOp, AxisArraysOp, ElemCollectionOp};
use anyhow::Result;
use bed_utils::bed::{
    tree::{BedTree, GenomeRegions, SparseBinnedCoverage},
    BEDLike,
};
use indexmap::IndexSet;
use indicatif::{style::ProgressStyle, ProgressBar, ProgressDrawTarget, ProgressIterator};
use itertools::Itertools;
use polars::prelude::{DataFrame, NamedFrom, Series};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::collections::{HashMap, HashSet, BTreeMap};

/// FIXME: Import insertions
pub fn import_insertions<A, B, I>(
    anndata: &A,
    fragments: I,
    promoter: &BedTree<bool>,
    regions: &GenomeRegions<B>,
    white_list: Option<&HashSet<String>>,
    min_num_fragment: u64,
    min_tsse: f64,
    fragment_is_sorted_by_name: bool,
    chunk_size: usize,
) -> Result<()>
where
    A: AnnDataOp,
    B: BEDLike + Clone + std::marker::Sync,
    I: Iterator<Item = Fragment>,
{
    let num_features = SparseBinnedCoverage::<_, u8>::new(regions, 1).len;
    let mut saved_barcodes = Vec::new();
    let mut qc = Vec::new();

    if fragment_is_sorted_by_name {
        let spinner = ProgressBar::with_draw_target(None, ProgressDrawTarget::stderr_with_hz(1))
            .with_style(
                ProgressStyle::with_template(
                    "{spinner} Processed {human_pos} barcodes in {elapsed} ({per_sec}) ...",
                )
                .unwrap(),
            );
        let mut scanned_barcodes = HashSet::new();
        anndata.obsm().add_iter(
            "insertion",
            fragments
                .group_by(|x| x.barcode.clone())
                .into_iter()
                .progress_with(spinner)
                .filter(|(key, _)| white_list.map_or(true, |x| x.contains(key)))
                .chunks(chunk_size)
                .into_iter()
                .map(|chunk| {
                    let data: Vec<(String, Vec<Fragment>)> =
                        chunk.map(|(barcode, x)| (barcode, x.collect())).collect();
                    let result: Vec<_> = data
                        .into_par_iter()
                        .map(|(barcode, x)|
                            (
                                barcode,
                                compute_qc_count(x, promoter, regions, min_num_fragment, min_tsse),
                            )
                        )
                        .collect();
                    let counts = result
                        .into_iter()
                        .filter_map(|(barcode, r)| {
                            if !scanned_barcodes.insert(barcode.clone()) {
                                panic!("Please sort fragment file by barcodes");
                            }
                            match r {
                                None => None,
                                Some((q, count)) => {
                                    saved_barcodes.push(barcode);
                                    qc.push(q);
                                    Some(count)
                                }
                            }
                        })
                        .collect::<Vec<_>>();
                    from_csr_rows(counts, num_features)
                }),
        )?;
    } else {
        let spinner = ProgressBar::with_draw_target(None, ProgressDrawTarget::stderr_with_hz(1))
            .with_style(
                ProgressStyle::with_template(
                    "{spinner} Processed {human_pos} reads in {elapsed} ({per_sec}) ...",
                )
                .unwrap(),
            );
        let mut scanned_barcodes = HashMap::new();
        fragments
            .progress_with(spinner)
            .filter(|frag| white_list.map_or(true, |x| x.contains(frag.barcode.as_str())))
            .for_each(|frag| {
                let key = frag.barcode.as_str();
                match scanned_barcodes.get_mut(key) {
                    None => {
                        let mut summary = FragmentSummary::new(promoter);
                        let mut counts = SparseBinnedCoverage::new(regions, 1);
                        summary.update(&frag);
                        frag.to_insertions()
                            .into_iter()
                            .for_each(|x| counts.insert(&x, 1));
                        scanned_barcodes.insert(key.to_string(), (summary, counts));
                    }
                    Some((summary, counts)) => {
                        summary.update(&frag);
                        frag.to_insertions()
                            .into_iter()
                            .for_each(|x| counts.insert(&x, 1));
                    }
                }
            });
        let csr_matrix = from_csr_rows(
            scanned_barcodes
                .drain()
                .filter_map(|(barcode, (summary, binned_coverage))| {
                    let q = summary.get_qc();
                    if q.num_unique_fragment < min_num_fragment || q.tss_enrichment < min_tsse {
                        None
                    } else {
                        saved_barcodes.push(barcode);
                        qc.push(q);
                        let count: Vec<(usize, u8)> = binned_coverage
                            .get_coverage()
                            .iter()
                            .map(|(k, v)| (*k, *v))
                            .collect();
                        Some(count)
                    }
                })
                .collect::<Vec<_>>(),
            num_features,
        );
        anndata.obsm().add("insertion", csr_matrix)?;
    }

    let chrom_sizes = DataFrame::new(vec![
        Series::new(
            "reference_seq_name",
            regions
                .regions
                .iter()
                .map(|x| x.chrom())
                .collect::<Series>(),
        ),
        Series::new(
            "reference_seq_length",
            regions.regions.iter().map(|x| x.end()).collect::<Series>(),
        ),
    ])?;
    anndata.uns().add("reference_sequences", chrom_sizes)?;
    anndata.set_obs_names(saved_barcodes.into())?;
    anndata.set_obs(qc_to_df(qc))?;
    Ok(())
}



/// Import fragments
pub fn import_fragments<A, I>(
    anndata: &A,
    fragments: I,
    promoter: &BedTree<bool>,
    chrom_sizes: &ChromSizes,
    white_list: Option<&HashSet<String>>,
    min_num_fragment: u64,
    min_tsse: f64,
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

    let genome_index = GenomeBaseIndex::new(chrom_sizes);
    let num_features = genome_index.len();
    let mut saved_barcodes = Vec::new();
    let mut qc = Vec::new();

    let mut scanned_barcodes = HashSet::new();
    anndata.obsm().add_iter(
        "fragment",
        fragments
            .group_by(|x| x.barcode.clone())
            .into_iter()
            .progress_with(spinner)
            .filter(|(key, _)| white_list.map_or(true, |x| x.contains(key)))
            .chunks(chunk_size)
            .into_iter()
            .map(|chunk| {
                let data: Vec<(String, Vec<Fragment>)> =
                    chunk.map(|(barcode, x)| (barcode, x.collect())).collect();
                let result: Vec<_> = data
                    .into_par_iter()
                    .map(|(barcode, x)| (barcode, fragment_to_indices(promoter, &genome_index, x)))
                    .collect();
                let counts = result
                    .into_iter()
                    .filter_map(|(barcode, (q, values))| {
                        if !scanned_barcodes.insert(barcode.clone()) {
                            panic!("Please sort fragment file by barcodes");
                        }
                        if q.num_unique_fragment < min_num_fragment || q.tss_enrichment < min_tsse {
                            return None;
                        } else {
                            saved_barcodes.push(barcode);
                            qc.push(q);
                            Some(values)
                        }
                    })
                    .collect::<Vec<_>>();
                from_csr_rows(counts, num_features)
            }),
    )?;

    anndata.uns().add("reference_sequences", chrom_sizes.to_dataframe())?;
    anndata.set_obs_names(saved_barcodes.into())?;
    anndata.set_obs(qc_to_df(qc))?;
    Ok(())
}

fn fragment_to_indices(
    promoter: &BedTree<bool>,
    genome_index: &GenomeBaseIndex,
    fragments: Vec<Fragment>,
) -> (QualityControl, Vec<(usize, i32)>) {
    let mut qc = FragmentSummary::new(promoter);
    let mut values = Vec::new();
    fragments.into_iter().for_each(|f| {
        qc.update(&f);
        let chrom = &f.chrom;
        if genome_index.contain_chrom(chrom) {
            let start = f.start;
            let end = f.end;
            let size: i32 = (end - start).try_into().unwrap();
            let pos1 = genome_index.get_position(chrom, start);
            let pos2 = genome_index.get_position(chrom, end-1);
            values.push((pos1, size-1));
            values.push((pos2, -size+1));
        }
    });
    values.sort();
    (qc.get_qc(), values)
}

fn qc_to_df(qc: Vec<QualityControl>) -> DataFrame {
    DataFrame::new(vec![
        Series::new(
            "tsse",
            qc.iter().map(|x| x.tss_enrichment).collect::<Series>(),
        ),
        Series::new(
            "n_fragment",
            qc.iter().map(|x| x.num_unique_fragment).collect::<Series>(),
        ),
        Series::new(
            "frac_dup",
            qc.iter().map(|x| x.frac_duplicated).collect::<Series>(),
        ),
        Series::new(
            "frac_mito",
            qc.iter().map(|x| x.frac_mitochondrial).collect::<Series>(),
        ),
    ])
    .unwrap()
}

/// Import scHi-C contacts into AnnData
pub fn import_contacts<A, B, I>(
    anndata: &A,
    contacts: I,
    regions: &GenomeRegions<B>,
    chunk_size: usize,
) -> Result<()>
where
    A: AnnDataOp,
    B: BEDLike + Clone + std::marker::Sync,
    I: Iterator<Item = Contact>,
{
    let chrom_sizes: ChromSizes = regions
        .regions
        .iter()
        .map(|x| x.chrom())
        .zip(regions.regions.iter().map(|x| x.end())).collect();
 
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
    anndata.obsm().add_iter(
        "contact",
        contacts
            .group_by(|x| x.barcode.clone())
            .into_iter()
            .progress_with(spinner)
            .chunks(chunk_size)
            .into_iter()
            .map(|chunk| {
                let data: Vec<Vec<Contact>> = chunk.map(|(barcode, x)| {
                    if !scanned_barcodes.insert(barcode.clone()) {
                        panic!("Please sort fragment file by barcodes");
                    }
                    x.collect()
                }).collect();

                let counts: Vec<_> = data
                    .into_par_iter()
                    .map(|x| {
                        let mut count = BTreeMap::new();
                        x.into_iter().for_each(|c| {
                            let pos1 = genome_index.get_position(&c.chrom1, c.start1);
                            let pos2 = genome_index.get_position(&c.chrom2, c.start2);
                            let i = pos1 * genome_size + pos2; 
                            count.entry(i).and_modify(|x| *x += c.count).or_insert(c.count);
                        });
                        count.into_iter().collect::<Vec<_>>()
                    }).collect();

                from_csr_rows(counts, genome_size*genome_size)
            }),
    )?;

    anndata.uns().add(
        "reference_sequences",
        DataFrame::new(vec![
            Series::new(
                "reference_seq_name",
                regions
                    .regions
                    .iter()
                    .map(|x| x.chrom())
                    .collect::<Series>(),
            ),
            Series::new(
                "reference_seq_length",
                regions.regions.iter().map(|x| x.end()).collect::<Series>(),
            ),
        ])?,
    )?;
    anndata.set_obs_names(scanned_barcodes.into_iter().collect())?;
    Ok(())
}