use noodles_bed as bed;
use bio::data_structures::interval_tree;

use std::collections::HashMap;

pub struct BedTree<T>(HashMap<String, interval_tree::IntervalTree<u64, T>>);

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
