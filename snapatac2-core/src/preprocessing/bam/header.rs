use std::{
    io::{self, BufRead, BufReader, Read},
    num::NonZeroUsize,
};
use bstr::BString;
use byteorder::{LittleEndian, ReadBytesExt};
use noodles::sam;
use noodles::sam::header::{
    record::value::{map::ReferenceSequence, Map},
    ReferenceSequences,
};

static MAGIC_NUMBER: &[u8] = b"BAM\x01";

// Read misformatted 10X bam headers
pub(super) fn read_10x_header<R>(reader: &mut R) -> io::Result<sam::Header>
where
    R: Read,
{
    read_magic(reader)?;

    let mut header = read_header_inner(reader)?;
    let reference_sequences = read_reference_sequences(reader)?;

    if header.reference_sequences().is_empty() {
        *header.reference_sequences_mut() = reference_sequences;
    } else if !reference_sequences_eq(header.reference_sequences(), &reference_sequences) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "SAM header and binary reference sequence dictionaries mismatch",
        ));
    }

    Ok(header)
}

// Read misformatted 10X bam headers
fn read_header_inner<R>(reader: &mut R) -> io::Result<sam::Header>
where
    R: Read,
{
    let l_text = reader.read_u32::<LittleEndian>().map(u64::from)?;

    let mut parser = sam::header::Parser::default();

    let mut header_reader = BufReader::new(reader.take(l_text));
    let mut buf = Vec::new();

    if read_line(&mut header_reader, &mut buf)? != 0 {
        let first_line = std::str::from_utf8(&buf).unwrap();
        if first_line.starts_with("@HD") {
            let mut fields: Vec<_> = first_line.split('\t').collect();
            if fields.len() == 1 || !fields[1].starts_with("VN") {
                fields.insert(1, "VN:1.0");
            } else {
                // Replace VN field if it exists
                fields[1] = "VN:1.0";
            }
            parser.parse_partial(fields.join("\t").as_bytes())
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        } else {
            parser.parse_partial(&buf)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        }

        while read_line(&mut header_reader, &mut buf)? != 0 {
            parser
                .parse_partial(&buf)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        }
    }

    discard_padding(&mut header_reader)?;

    Ok(parser.finish())
}

fn read_magic<R>(reader: &mut R) -> io::Result<()>
where
    R: Read,
{
    let mut magic = [0; 4];
    reader.read_exact(&mut magic)?;

    if magic == MAGIC_NUMBER {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid BAM header",
        ))
    }
}

fn read_line<R>(reader: &mut R, dst: &mut Vec<u8>) -> io::Result<usize>
where
    R: BufRead,
{
    const NUL: u8 = 0x00;
    const LINE_FEED: u8 = b'\n';
    const CARRIAGE_RETURN: u8 = b'\r';

    dst.clear();

    let src = reader.fill_buf()?;

    if src.is_empty() || src[0] == NUL {
        return Ok(0);
    }

    match reader.read_until(LINE_FEED, dst)? {
        0 => Ok(0),
        n => {
            if dst.ends_with(&[LINE_FEED]) {
                dst.pop();

                if dst.ends_with(&[CARRIAGE_RETURN]) {
                    dst.pop();
                }
            }

            Ok(n)
        }
    }
}

fn discard_padding<R>(reader: &mut R) -> io::Result<()>
where
    R: BufRead,
{
    loop {
        let src = reader.fill_buf()?;

        if src.is_empty() {
            return Ok(());
        }

        let len = src.len();
        reader.consume(len);
    }
}

fn read_reference_sequences<R>(reader: &mut R) -> io::Result<ReferenceSequences>
where
    R: Read,
{
    let n_ref = reader.read_u32::<LittleEndian>().and_then(|n| {
        usize::try_from(n).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    })?;

    let mut reference_sequences = ReferenceSequences::with_capacity(n_ref);

    for _ in 0..n_ref {
        let (name, reference_sequence) = read_reference_sequence(reader)?;
        reference_sequences.insert(name, reference_sequence);
    }

    Ok(reference_sequences)
}

fn read_reference_sequence<R>(reader: &mut R) -> io::Result<(BString, Map<ReferenceSequence>)>
where
    R: Read,
{
    let l_name = reader.read_u32::<LittleEndian>().and_then(|n| {
        usize::try_from(n).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    })?;

    let mut c_name = vec![0; l_name];
    reader.read_exact(&mut c_name)?;

    let name = bytes_with_nul_to_bstring(&c_name)?;

    let l_ref = reader.read_u32::<LittleEndian>().and_then(|len| {
        usize::try_from(len)
            .and_then(NonZeroUsize::try_from)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    })?;

    let reference_sequence = Map::<ReferenceSequence>::new(l_ref);

    Ok((name, reference_sequence))
}

pub(crate) fn bytes_with_nul_to_bstring(buf: &[u8]) -> io::Result<BString> {
    std::ffi::CStr::from_bytes_with_nul(buf)
        .map(|c_str| c_str.to_bytes().into())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

pub(crate) fn reference_sequences_eq(
    header_reference_sequences: &ReferenceSequences,
    binary_reference_sequences: &ReferenceSequences,
) -> bool {
    header_reference_sequences.len() == binary_reference_sequences.len()
        && header_reference_sequences
            .iter()
            .zip(binary_reference_sequences)
            .all(|((h_name, h_map), (b_name, b_map))| {
                h_name == b_name && h_map.length() == b_map.length()
            })
}