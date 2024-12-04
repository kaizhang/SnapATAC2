use crate::utils::read_transcripts;

use pyo3::prelude::*;

use snapatac2_core::{
    network::link_region_to_promoter,
    genome::Promoters,
};
use bed_utils::bed::GenomicRange;
use std::{
    str::FromStr,
    collections::HashMap,
};

#[pyfunction]
pub(crate) fn link_region_to_gene(
    regions: Vec<String>,
    annot_fl: &str,
    upstream: u64,
    downstream: u64,
    id_type: &str,
    coding_gene_only: bool,
) -> HashMap<(String, String), Vec<(String, String, u64)>>
{
    let promoters = Promoters::new(
        read_transcripts(annot_fl, &Default::default()).into_iter()
            .filter(|x| if coding_gene_only { x.is_coding.unwrap_or(true) } else { true })
            .collect(),
        upstream,
        downstream,
        false,
    );
    let regions_: Vec<GenomicRange> = regions.into_iter().map(|x| GenomicRange::from_str(&x).unwrap()).collect();
    link_region_to_promoter(&regions_, &promoters,).get_linkages(id_type)
        .into_iter().map(|(k, links)| {
            let data = links.into_iter()
                .map(|(i, x)| (i.to_owned(), "region".to_owned(), x)).collect();
            ((k.to_owned(), "gene".to_owned()), data)
        }).collect()
}
