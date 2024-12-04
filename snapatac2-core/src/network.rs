use crate::genome::Promoters;

use bed_utils::bed::BEDLike;
use std::collections::HashMap;

pub struct PromoterLinkage<'a, B> {
    promoters: &'a Promoters,
    links: Vec<Vec<&'a B>>,
}

impl<'a, B> PromoterLinkage<'a, B>
where
    B: BEDLike,
{
    pub fn get_linkages(&self, id_type: &str) -> HashMap<&str, HashMap<String, u64>> {
        match id_type {
            "transcript_id" => self.promoters.transcripts.iter().zip(self.links.iter())
                .map(|(p, regions)| {
                    let id = p.transcript_id.as_str();
                    let tss = p.get_tss().unwrap() as u64;
                    let links = regions.iter().map(|x| {
                        let d = if tss < x.end() && tss >= x.start() {
                            0
                        } else {
                            x.start().abs_diff(tss).min(x.end().abs_diff(tss))
                        };
                        (x.to_genomic_range().pretty_show(), d)
                    }).collect();
                    (id, links)
                }).collect(),

            // Closest TSS is selected for distance calculation
            "gene_id" | "gene_name" => {
                let mut linkages = HashMap::new();
                self.promoters.transcripts.iter().zip(self.links.iter()).for_each(|(p, regions)| {
                    let id = if id_type == "gene_id" {
                        p.gene_id.as_str()
                    } else {
                        p.gene_name.as_str()
                    };
                    let tss = p.get_tss().unwrap() as u64;
                    let val = linkages.entry(id).or_insert(HashMap::new());
                    regions.iter().for_each(|x| {
                        let d = if tss < x.end() && tss >= x.start() {
                            0
                        } else {
                            x.start().abs_diff(tss).min(x.end().abs_diff(tss))
                        };
                        let v = val.entry(x.to_genomic_range().pretty_show()).or_insert(d);
                        if *v > d { *v = d; }
                    });
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
        promoters.regions.find_index_of(x).for_each(|i| assoc_regions[i].push(x))
    );
    PromoterLinkage {
        promoters,
        links: assoc_regions,
    }
}