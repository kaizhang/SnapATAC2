use std::ops::Range;
use bio::data_structures::interval_tree::*;
use std::collections::HashMap;

use super::{BED, BEDLike};

pub struct BedTree<D>(HashMap<String, IntervalTree<u64, D>>);

impl<D> Default for BedTree<D> {
    fn default() -> Self {
        Self(HashMap::new())
    }
}

impl<D, B: BEDLike> FromIterator<(B, D)> for BedTree<D> {
    fn from_iter<I: IntoIterator<Item = (B, D)>>(iter: I) -> Self {
        let mut hmap: HashMap<String, Vec<(Range<u64>, D)>> = HashMap::new();
        for (bed, data) in iter {
            let chr = bed.chrom();
            let interval = bed.chrom_start() .. bed.chrom_end();
            let vec = hmap.entry(chr.to_string()).or_insert(Vec::new());
            vec.push((interval, data));
        }
        let hm = hmap.into_iter().map(|(chr, vec)| (chr, vec.into_iter().collect())).collect();
        BedTree(hm)
    }
}

/// An `IntervalTreeIterator` is returned by `Intervaltree::find` and iterates over the entries
/// overlapping the query
pub struct BedTreeIterator<'a, D> {
    chrom: String,
    interval_tree_iterator: Option<IntervalTreeIterator<'a, u64, D>>,
}

impl<'a, D: 'a> Iterator for BedTreeIterator<'a, D> {
    type Item = (BED<3>, &'a D);

    fn next(&mut self) -> Option<(BED<3>, &'a D)> {
        match self.interval_tree_iterator {
            None => return None,
            Some(ref mut iter) => match iter.next() {
                None => return None,
                Some(item) => {
                    let bed = BED::new_bed3(self.chrom.to_string(), item.interval().start, item.interval().end);
                    Some((bed, item.data()))
                }
            }
        }
    }
}


impl<D> BedTree<D> {
    pub fn find<B: BEDLike>(&self, bed: &B) -> BedTreeIterator<'_, D> {
        let chr = bed.chrom().to_string();
        let interval = bed.chrom_start() .. bed.chrom_end();
        match self.0.get(&chr) {
            None => BedTreeIterator { chrom: chr, interval_tree_iterator: None },
            Some(tree) => BedTreeIterator { chrom: chr, interval_tree_iterator: Some(tree.find(interval)) }
        }
    }

    pub fn is_overlapped<B: BEDLike>(&self, bed: &B) -> bool {
        self.find(bed).next().is_some()
    }
}

#[cfg(test)]
mod bed_intersect_tests {
    use super::*;

    #[test]
    fn test_intersect() {
        let bed_set1: Vec<BED<3>> = vec![BED::new_bed3("chr1".to_string(), 200, 500), BED::new_bed3("chr1".to_string(), 1000, 2000)];
        let bed_set2: Vec<BED<3>> = vec![BED::new_bed3("chr1".to_string(), 100, 210), BED::new_bed3("chr1".to_string(), 100, 200)];

        let tree: BedTree<()> = bed_set1.into_iter().map(|x| (x, ())).collect();
        let result: Vec<BED<3>> = bed_set2.into_iter().filter(|x| tree.is_overlapped(x)).collect();
        let expected = vec![BED::new_bed3("chr1".to_string(), 100, 210)];
        assert_eq!(result, expected);
    }
}