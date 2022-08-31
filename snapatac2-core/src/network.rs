use crate::utils::gene::Promoters;

use bed_utils::bed::{
    GenomicRange, BEDLike
};
use std::collections::{HashSet, HashMap};

pub struct PromoterLinkage<'a, B> {
    promoters: &'a Promoters,
    links: Vec<Vec<&'a B>>,
}

impl<'a, B> PromoterLinkage<'a, B>
where
    B: BEDLike,
{
    pub fn get_linkages(&self, id_type: &str) -> HashMap<&str, HashSet<String>> {
        match id_type {
            "transcript_id" => self.promoters.transcript_ids.iter().map(|x| x.as_str()).zip(
                self.links.iter().map(|x| x.iter().map(|x| x.pretty_show()).collect())
            ).collect(),
            "gene_id" | "gene_name" => {
                let mut linkages = HashMap::new();
                let ids = if id_type == "gene_id" {
                    self.promoters.gene_ids.iter()
                } else {
                    self.promoters.gene_names.iter()
                };
                ids.map(|x| x.as_str())
                    .zip(self.links.iter()).for_each(|(gene, links)| {
                        let v = linkages.entry(gene).or_insert(HashSet::new());
                        links.iter().for_each(|x| { v.insert(x.pretty_show()); });
                    });
                linkages
            },
            _ => panic!("id_type must be one of transcript_id, gene_id or gene_name"),
        }
    }

}

/// Link genomic regions to genes if they are within the promoter regions.
pub fn link_region_to_promoter<'a, B>(
    regions: &'a [B],
    promoters: &'a Promoters,
) -> PromoterLinkage<'a, B>
where
    B: BEDLike,
{
    let mut assoc_regions = vec![Vec::new(); promoters.regions.len()];
    regions.into_iter().for_each(|x|
        promoters.regions.indices.find(x).for_each(|(_, i)| assoc_regions[*i].push(x))
    );
    PromoterLinkage {
        promoters,
        links: assoc_regions,
    }
}