use crate::utils::open_file;

use pyo3::{
    prelude::*,
    PyResult, Python,
    type_object::PyTypeObject,
    exceptions::PyTypeError,
};

use snapatac2_core::{
    network::link_region_to_promoter,
    utils::gene::{Promoters, read_transcripts},
};
use bed_utils::bed::GenomicRange;
use bed_utils::bed::BEDLike;
use std::io::BufReader;
use std::collections::HashMap;
use std::str::FromStr;

#[pyfunction]
pub(crate) fn link_region_to_gene(
    regions: Vec<String>,
    annot_fl: &str,
    upstream: u64,
    downstream: u64,
    id_type: &str,
    coding_gene_only: bool,
) -> HashMap<String, Vec<String>>
{
    let promoters = Promoters::new(
        read_transcripts(BufReader::new(open_file(annot_fl)))
            .into_iter().map(|(_, v)| v)
            .filter(|x| if coding_gene_only { x.is_coding.unwrap_or(true) } else { true })
            .collect(),
        upstream,
        downstream,
        true,
    );
    let regions_: Vec<GenomicRange> = regions.into_iter().map(|x| GenomicRange::from_str(&x).unwrap()).collect();
    link_region_to_promoter(&regions_, &promoters,).get_linkages(id_type)
        .into_iter().map(|(k, v)|
            (k.to_string(), v.into_iter().collect())
        ).collect()
}
