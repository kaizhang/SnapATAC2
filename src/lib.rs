use std::io;
use std::io::prelude::*;
use std::fs::File;
use bio::io::bed;
use std::io::BufReader;                                                                                                                                           
use itertools::Itertools;
use itertools::Groups;                                                                                                          
use csv::Reader;

fn read_tss<R: Read>(file: R) -> impl Iterator<Item = (String, u32, bool)> {
    let mut reader = BufReader::new(file);
    let parseLine = |line: io::Result<String>| {
        let chr_idx: usize = 0;
        let type_idx: usize = 2;
        let start_idx: usize = 3;
        let end_idx: usize = 4;
        let strand_idx: usize = 6;
        let l = line.unwrap();
        if l.as_bytes()[0] as char == '#' {
            return None;
        }
        let elements: Vec<&str> = l.split('\t').collect();
        if elements[type_idx] == "transcript" {
            let chr = elements[chr_idx].to_string();
            let is_fwd = elements[strand_idx] != "-";
            let tss: u32 = 
                if is_fwd {
                    elements[start_idx].parse::<u32>().unwrap() - 1
                } else {
                    elements[end_idx].parse::<u32>().unwrap() - 1
                };
            Some((chr, tss, is_fwd))
        } else {
            None
        }
    };
    reader.lines().filter_map(parseLine)
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
        let gencode = File::open("data/gencode.gtf.gz").expect("xx");
        for x in super::read_tss(GzDecoder::new(gencode)) {
            println!("{}\t{}\t{}", x.0, x.1, x.2);
        }
    }
}