pub mod score;
pub mod strand;
pub mod io;
pub mod tree;

pub use self::{score::Score, strand::Strand};

use std::{
    fmt::{self, Write},
    num,
    ops::Deref,
    str::FromStr,
};

const DELIMITER: char = '\t';
const MISSING_ITEM : &str = ".";

/*
/// A minimal BED record with only 3 fields.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BED3(String, u64, u64);

impl BED3 {
    fn new<C>(chrom: C, chrom_start: u64, chrom_end: u64) -> Self
    where
        C: Into<String>,
    { Self(chrom.into(), chrom_start, chrom_end) }
}
*/

/// A standard BED record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BED<const N: u8> {
    chrom: String,
    chrom_start: u64,
    chrom_end: u64,
    name: Option<String>,
    score: Option<Score>,
    strand: Option<Strand>,
    optional_fields: OptionalFields,
}

impl<const N: u8> BED<N> {
    fn new<C>(chrom: C, chrom_start: u64, chrom_end: u64, name: Option<String>,
        score: Option<Score>, strand: Option<Strand>, optional_fields: OptionalFields) -> Self
    where
        C: Into<String>,
    { Self { chrom: chrom.into(), chrom_start, chrom_end, name, score, strand, optional_fields } }

    fn new_bed3<C>(chrom: C, chrom_start: u64, chrom_end: u64) -> Self
    where
        C: Into<String>,
    { Self { chrom: chrom.into(), chrom_start, chrom_end, name: None, score: None, strand: None, optional_fields: OptionalFields(Vec::new()) } }
}

/// Common BED fields
pub trait BEDLike {
    fn chrom(&self) -> &str;
    fn chrom_start(&self) -> u64;
    fn chrom_end(&self) -> u64;
    fn name(&self) -> Option<&str> { None }
    fn score(&self) -> Option<Score> { None }
    fn strand(&self) -> Option<Strand> { None }
}

/*
impl BEDLike for BED3 {
    fn chrom(&self) -> &str { &self.0 }
    fn chrom_start(&self) -> u64 { self.1 }
    fn chrom_end(&self) -> u64 { self.2 }
    fn name(&self) -> Option<&str> { None }
    fn score(&self) -> Option<Score> { None }
    fn strand(&self) -> Option<Strand> { None }
}
*/

impl<const N: u8> BEDLike for BED<N> {
    fn chrom(&self) -> &str { &self.chrom }
    fn chrom_start(&self) -> u64 { self.chrom_start }
    fn chrom_end(&self) -> u64 { self.chrom_end }
    fn name(&self) -> Option<&str> { self.name.as_deref() }
    fn score(&self) -> Option<Score> { self.score }
    fn strand(&self) -> Option<Strand> { self.strand }
}

// Display trait
impl<const N: u8> fmt::Display for BED<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{}{}{}{}",
            self.chrom(),
            DELIMITER,
            self.chrom_start(),
            DELIMITER,
            self.chrom_end()
        )?;
        if N > 3 {
            write!(f, "{}{}", DELIMITER, self.name().unwrap_or(MISSING_ITEM))?;
            if N > 4 {
                f.write_char(DELIMITER)?;
                if let Some(score) = self.score() {
                    write!(f, "{}", score)?;
                } else { f.write_str(MISSING_ITEM)?; }

                if N > 5 {
                    f.write_char(DELIMITER)?;
                    if let Some(strand) = self.strand() {
                        write!(f, "{}", strand)?;
                    } else { f.write_str(MISSING_ITEM)?; }
                }
            }
        }
        Ok(())
    }
}

impl<const N: u8> FromStr for BED<N> {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut fields = s.split(DELIMITER);
        let chrom = parse_chrom(&mut fields)?;
        let chrom_start = parse_chrom_start(&mut fields)?;
        let chrom_end = parse_chrom_end(&mut fields)?;
        let name = if N > 3 { parse_name(&mut fields)? } else { None };
        let score = if N > 4 { parse_score(&mut fields)? } else { None };
        let strand = if N > 5 { parse_strand(&mut fields)? } else { None };
        Ok(BED::new(chrom, chrom_start, chrom_end, name, score, strand, OptionalFields::default()))
    }
}

fn parse_chrom<'a, I>(fields: &mut I) -> Result<&'a str, ParseError>
where
    I: Iterator<Item = &'a str>,
{
    fields
        .next()
        .ok_or(ParseError::MissingReferenceSequenceName)
}

fn parse_chrom_start<'a, I>(fields: &mut I) -> Result<u64, ParseError>
where
    I: Iterator<Item = &'a str>,
{
    fields
        .next()
        .ok_or(ParseError::MissingStartPosition)
        .and_then(|s| s.parse().map_err(ParseError::InvalidStartPosition))
}

fn parse_chrom_end<'a, I>(fields: &mut I) -> Result<u64, ParseError>
where
    I: Iterator<Item = &'a str>,
{
    fields
        .next()
        .ok_or(ParseError::MissingEndPosition)
        .and_then(|s| s.parse().map_err(ParseError::InvalidEndPosition))
}

fn parse_name<'a, I>(fields: &mut I) -> Result<Option<String>, ParseError>
where
    I: Iterator<Item = &'a str>,
{
    fields
        .next()
        .ok_or(ParseError::MissingName)
        .map(|s| match s {
            MISSING_ITEM => None,
            _ => Some(s.into()),
        })
}

fn parse_score<'a, I>(fields: &mut I) -> Result<Option<Score>, ParseError>
where
    I: Iterator<Item = &'a str>,
{
    fields
        .next()
        .ok_or(ParseError::MissingName)
        .and_then(|s| match s {
            MISSING_ITEM => Ok(None),
            _ => s.parse().map(Some).map_err(ParseError::InvalidScore),
        })
}

fn parse_strand<'a, I>(fields: &mut I) -> Result<Option<Strand>, ParseError>
where
    I: Iterator<Item = &'a str>,
{
    fields
        .next()
        .ok_or(ParseError::MissingStrand)
        .and_then(|s| match s {
            MISSING_ITEM => Ok(None),
            _ => s.parse().map(Some).map_err(ParseError::InvalidStrand),
        })
}

/// An error returned when a raw BED record fails to parse.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ParseError {
    /// The reference sequence name is missing.
    MissingReferenceSequenceName,
    /// The start position is missing.
    MissingStartPosition,
    /// The start position is invalid.
    InvalidStartPosition(num::ParseIntError),
    /// The end position is missing.
    MissingEndPosition,
    /// The end position is invalid.
    InvalidEndPosition(num::ParseIntError),
    /// The name is missing.
    MissingName,
    /// The score is missing.
    MissingScore,
    /// The score is invalid.
    InvalidScore(score::ParseError),
    /// The strand is missing.
    MissingStrand,
    /// The strand is invalid.
    InvalidStrand(strand::ParseError),
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