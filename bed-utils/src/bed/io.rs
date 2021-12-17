use std::io::{self, BufRead};
use std::str::FromStr;
use std::marker::PhantomData;

use super::{BED};

/// An iterator over records of a FASTQ reader.
///
/// This is created by calling [`Reader::records`].
pub struct Records<'a, B, R> {
    inner: &'a mut Reader<R>,
    buf: String,
    phantom: PhantomData<B>,
}

impl<'a, B, R> Records<'a, B, R>
where
    R: BufRead,
    B: FromStr,
{
    pub fn new(inner: &'a mut Reader<R>) -> Self {
        Self {
            inner,
            buf: String::new(),
            phantom: PhantomData,
        }
    }
}

impl<'a, B, R> Iterator for Records<'a, B, R>
where
    R: BufRead,
    B: FromStr,
{
    type Item = io::Result<B>;

    fn next(&mut self) -> Option<Self::Item> {
        self.buf.clear();

        match self.inner.read_record(&mut self.buf) {
            Ok(0) => None,
            Ok(_) => Some(Ok(self.buf.parse().ok()?)),
            Err(e) => Some(Err(e)),
        }
    }
}

/// A BED reader.
pub struct Reader<R> {
    inner: R,
}

impl<R> Reader<R>
where
    R: BufRead,
{
    /// Creates a BED reader.
    pub fn new(inner: R) -> Self {
        Self { inner }
    }

    /// Reads a single raw BED record.
    pub fn read_record(&mut self, buf: &mut String) -> io::Result<usize> {
        read_line(&mut self.inner, buf)
    }

    /// Returns an iterator over records starting from the current stream position.
    ///
    /// The stream is expected to be at the start of a record.
    ///
    pub fn records<B: FromStr>(&mut self) -> Records<'_, B, R> {
        Records::new(self)
    }
}

fn read_line<R>(reader: &mut R, buf: &mut String) -> io::Result<usize>
where
    R: BufRead,
{
    const LINE_FEED: char = '\n';
    const CARRIAGE_RETURN: char = '\r';

    match reader.read_line(buf) {
        Ok(0) => Ok(0),
        Ok(n) => {
            if buf.ends_with(LINE_FEED) {
                buf.pop();

                if buf.ends_with(CARRIAGE_RETURN) {
                    buf.pop();
                }
            }
            Ok(n)
        }
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_line() {
        fn t(buf: &mut String, mut reader: &[u8], expected: &str) {
            buf.clear();
            read_line(&mut reader, buf);
            assert_eq!(buf, expected);
        }

        let mut buf = String::new();

        t(&mut buf, b"noodles\n", "noodles");
        t(&mut buf, b"noodles\r\n", "noodles");
        t(&mut buf, b"noodles", "noodles");
    }

    #[test]
    fn test_read_record() {
        let data = b"\
chr1	200	1000	r1	100	+
chr2	220	2000	r2	2	-
chr10	2000	10000	r3	3	+
" as &[u8];
        let mut reader = Reader::new(data);
        for b in reader.records() {
            let x: BED<6> = b.unwrap();
            println!("{}", x);
        }

        /*
        read_record(&mut reader, &mut record)?;
        assert_eq!(record, Record::new("noodles:1/1", "AGCT", "abcd"));

        read_record(&mut reader, &mut record)?;
        assert_eq!(record, Record::new("noodles:2/1", "TCGA", "dcba"));

        let n = read_record(&mut reader, &mut record)?;
        assert_eq!(n, 0);
        */

    }
}