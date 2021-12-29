use std::{
    fmt,
    io::{self, Error, ErrorKind, Write, Read, BufRead, BufReader},
};
use std::str::FromStr;
use std::marker::PhantomData;

use super::{BED, ParseError};

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
    R: Read,
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
    R: Read,
    B: FromStr<Err = ParseError>,
{
    type Item = io::Result<B>;

    fn next(&mut self) -> Option<Self::Item> {
        self.buf.clear();
        match self.inner.read_record(&mut self.buf) {
            Ok(0) => None,
            Ok(_) => Some(self.buf.parse().map_err(
                |e| Error::new(ErrorKind::Other, format!("{:?}: {}", e, &self.buf))
            )),
            Err(e) => Some(Err(e)),
        }
    }
}

/// An iterator over records of a FASTQ reader.
///
/// This is created by calling [`Reader::records`].
pub struct IntoRecords<B, R> {
    inner: Reader<R>,
    buf: String,
    phantom: PhantomData<B>,
}

impl<B, R> IntoRecords<B, R>
where
    R: Read,
    B: FromStr,
{
    pub fn new(inner: Reader<R>) -> Self {
        Self { inner, buf: String::new(), phantom: PhantomData }
    }
}

impl<B, R> Iterator for IntoRecords<B, R>
where
    R: Read,
    B: FromStr<Err = ParseError>,
{
    type Item = io::Result<B>;

    fn next(&mut self) -> Option<Self::Item> {
        self.buf.clear();
        match self.inner.read_record(&mut self.buf) {
            Ok(0) => None,
            Ok(_) => Some(self.buf.parse().map_err(
                |e| Error::new(ErrorKind::Other, format!("{:?}: {}", e, &self.buf))
            )),
            Err(e) => Some(Err(e)),
        }
    }
}

/// A BED reader.
pub struct Reader<R> {
    inner: BufReader<R>,
}

impl<R> Reader<R>
where
    R: Read,
{
    /// Creates a BED reader.
    pub fn new(inner: R) -> Self {
        Self { inner: BufReader::new(inner) }
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

    pub fn into_records<B: FromStr>(self) -> IntoRecords<B, R> {
        IntoRecords::new(self)
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



/*

/// A BED writer.
pub struct Writer<W> { inner: W }

impl<W> Writer<W> where W: Write {
    /// Creates a BED writer.
    pub fn new(inner: W) -> Self { Self { inner } }

    /// Writes a BED record.
    pub fn write_record<const N: u8>(&mut self, record: &BED<N>) -> io::Result<()>
    where
        Record<N>: fmt::Display,
    {
        write_record(&mut self.inner, record)
    }
}

fn write_record<W, const N: u8>(writer: &mut W, record: &Record<N>) -> io::Result<()>
where
    W: Write,
    Record<N>: fmt::Display,
{
    writeln!(writer, "{}", record)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_record() -> Result<(), Box<dyn std::error::Error>> {
        let mut buf = Vec::new();
        let record: Record<3> = "sq0\t8\t13".parse()?;
        write_record(&mut buf, &record)?;
        assert_eq!(buf, b"sq0\t8\t13\n");
        Ok(())
    }
}
*/