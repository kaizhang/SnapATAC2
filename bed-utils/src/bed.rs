pub mod score;
pub mod strand;

pub use self::{score::Score, strand::Strand};

use std::{
    error,
    fmt::{self, Write},
    num,
    ops::Deref,
    ops::Range,
    str::FromStr,
};
use bio::data_structures::interval_tree::*;
use std::collections::HashMap;

const DELIMITER: char = '\t';
const MISSING_ITEM : &str = ".";

pub trait BEDLike {
    fn chrom(&self) -> &str;
    fn chrom_start(&self) -> u64;
    fn chrom_end(&self) -> u64;
    fn name(&self) -> Option<String>;
    fn score(&self) -> Option<Score>;
    fn strand(&self) -> Option<Strand>;
}

/// A lightweight BED record with only 3 fields.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BED3(String, u64, u64);

impl BEDLike for BED3 {
    fn chrom(&self) -> &str { &self.0 }
    fn chrom_start(&self) -> u64 { self.1 }
    fn chrom_end(&self) -> u64 { self.2 }
    fn name(&self) -> Option<String> { None }
    fn score(&self) -> Option<Score> { None }
    fn strand(&self) -> Option<Strand> { None }
}

/// Generic BED record optional fields.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct OptionalFields(Vec<String>);

impl Deref for OptionalFields {
    type Target = [String];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for OptionalFields {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, field) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_char(DELIMITER)?;
            }

            f.write_str(field)?;
        }

        Ok(())
    }
}

impl From<Vec<String>> for OptionalFields {
    fn from(fields: Vec<String>) -> Self {
        Self(fields)
    }
}

#[cfg(test)]
mod optional_fields_tests {
    use super::*;

    #[test]
    fn test_fmt() {
        let fields = OptionalFields::default();
        assert_eq!(fields.to_string(), "");

        let fields = OptionalFields::from(vec![String::from("n")]);
        assert_eq!(fields.to_string(), "n");

        let fields = OptionalFields::from(vec![String::from("n"), String::from("d")]);
        assert_eq!(fields.to_string(), "n\td");
    }
}

/// A standard BED record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BED<const N: u8> {
    reference_sequence_name: String,
    chrom_start: u64,
    chrom_end: u64,
    name: Option<String>,
    score: Option<Score>,
    strand: Option<Strand>,
    optional_fields: OptionalFields,
}

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
    type Item = (BED3, &'a D);

    fn next(&mut self) -> Option<(BED3, &'a D)> {
        match self.interval_tree_iterator {
            None => return None,
            Some(ref mut iter) => match iter.next() {
                None => return None,
                Some(item) => {
                    let bed = BED3(self.chrom.to_string(), item.interval().start, item.interval().end);
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
        let bed_set1: Vec<BED3> = vec![BED3("chr1".to_string(), 200, 500), BED3("chr1".to_string(), 1000, 2000)];
        let bed_set2: Vec<BED3> = vec![BED3("chr1".to_string(), 100, 210), BED3("chr1".to_string(), 100, 200)];

        let tree: BedTree<()> = bed_set1.into_iter().map(|x| (x, ())).collect();
        let result: Vec<BED3> = bed_set2.into_iter().filter(|x| tree.is_overlapped(x)).collect();
        let expected = vec![BED3("chr1".to_string(), 100, 210)];
        assert_eq!(result, expected);
    }
}