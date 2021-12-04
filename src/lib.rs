use std::io;
use std::io::prelude::*;
use std::fs::File;
use bio::io::bed;
use std::io::BufReader;                                                                                                                                           
use itertools::Itertools;
use itertools::Groups;                                                                                                          
use csv::Reader;

fn read_tss<R: Read>(file: R) {

}

fn read_fragments<R: Read>(file: R) -> impl Iterator<Item = bed::Record> {
    bed::Reader::new(file).into_records().map(Result::unwrap)
}

/*
fn group_fragments_by_name<I>(iter: I) -> itertools::GroupBy<String, I, impl FnMut(&bed::Record) -> String>
    where I: Iterator<Item = bed::Record>
{
    iter.group_by(|x| *x.name().unwrap().to_string())
}
*/

/*
fn group_fragments_by_name<'a, I>(iter: I) -> impl Iterator<Item = u32> + 'a
    where I: Iterator<Item = bed::Record> + 'a
{
    iter.group_by(|x| x.name().unwrap().to_string()).into_iter()
        .map(|(x, y)| 1)
}
*/

/*
fn decode_reader<'a>(file: &'a File) -> impl Iterator<Item=Record> + 'a {
    return bed::Reader::new(GzDecoder::new(file))
        .records().map(|x| x.unwrap())
        //.group_by(|x| x.name().unwrap().to_string())
        //.into_iter()
        //.map(|(x, y)| x);


    /*
    for (k, rec) in &b {
        println!("{}", k);
    }
    */
}
*/

#[cfg(test)]
mod tests {
    use std::fs::File;
    use flate2::read::GzDecoder;

    #[test]
    fn it_works() {
        let f = File::open("data/fragments.bed.gz").expect("xx");

        //for x in super::read_fragments(GzDecoder::new(f)) {
        //    println!("{}", x.chrom());
        //}
    }
}