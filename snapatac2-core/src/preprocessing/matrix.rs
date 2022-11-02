use anndata_rs::{anndata::{AnnData, AnnDataSet}, iterator::CsrIterator, element::ElemCollection};
use anndata_rs::anndata_trait::DataType;
use anndata_rs::anndata_trait::DataPartialIO;
use polars::prelude::{NamedFrom, DataFrame, Series};
use anyhow::{Result, bail, anyhow};
use rayon::iter::{ParallelIterator, IntoParallelIterator};
use indicatif::{ProgressIterator, style::ProgressStyle};
use num::integer::div_ceil;
use nalgebra_sparse::CsrMatrix;
use noodles::{core::Position, gff::{Reader, record::Strand}};
use std::{fmt::Debug, str::FromStr, io::BufRead, collections::{BTreeMap, HashSet, HashMap}};
use std::marker::PhantomData;
use indexmap::map::IndexMap;
use bed_utils::bed::{GenomicRange, BEDLike, BedGraph, tree::{GenomeRegions, SparseCoverage, SparseBinnedCoverage}};
use itertools::Itertools;
use num::traits::{ToPrimitive, NumCast};
use hdf5::types::TypeDescriptor::*;
use hdf5::types::IntSize::*;

/// A structure that stores the feature counts.
pub trait FeatureCounter {
    type Value;

    /// Reset the counter.
    fn reset(&mut self);

    /// Update counter according to the region and the assocated count.
    fn insert<B: BEDLike, N: ToPrimitive + Copy>(&mut self, tag: &B, count: N);

    fn inserts<B, N>(&mut self, data: B)
    where
        B: Into<ChromValues<N>>,
        N: ToPrimitive + Copy,
    {
        data.into().into_iter().for_each(|x| self.insert(&x, x.value));
    }

    /// Retrieve feature ids.
    fn get_feature_ids(&self) -> Vec<String>;

    /// Retrieve feature names.
    fn get_feature_names(&self) -> Option<Vec<String>> { None }

    /// Retrieve stored counts.
    fn get_counts(&self) -> Vec<(usize, Self::Value)>;
}

impl<D: BEDLike + Clone> FeatureCounter for SparseBinnedCoverage<'_, D, u32> {
    type Value = u32;

    fn reset(&mut self) { self.reset(); }

    fn insert<B: BEDLike, N: ToPrimitive + Copy>(&mut self, tag: &B, count: N) {
        self.insert(tag, <u32 as NumCast>::from(count).unwrap());
    }

    fn get_feature_ids(&self) -> Vec<String> {
        self.get_regions().flatten().map(|x| x.to_genomic_range().pretty_show()).collect()
    }

    fn get_counts(&self) -> Vec<(usize, Self::Value)> {
        self.get_coverage().iter().map(|(k, v)| (*k, *v)).collect()
    }
}

impl<D: BEDLike> FeatureCounter for SparseCoverage<'_, D, u32> {
    type Value = u32;

    fn reset(&mut self) { self.reset(); }

    fn insert<B: BEDLike, N: ToPrimitive + Copy>(&mut self, tag: &B, count: N) {
        self.insert(tag, <u32 as NumCast>::from(count).unwrap());
    }

    fn get_feature_ids(&self) -> Vec<String> {
        self.get_regions().map(|x| x.to_genomic_range().pretty_show()).collect()
    }

    fn get_counts(&self) -> Vec<(usize, Self::Value)> {
        self.get_coverage().iter().map(|(k, v)| (*k, *v)).collect()
    }
}

/// Position is 0-based.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Transcript {
    pub transcript_name: Option<String>,
    pub transcript_id: String,
    pub gene_name: String,
    pub gene_id: String,
    pub is_coding: Option<bool>,
    pub chrom: String,
    pub left: Position,
    pub right: Position,
    pub strand: Strand,
}

impl Transcript {
    pub fn get_tss(&self) -> Option<usize> {
        match self.strand {
            Strand::Forward => Some(<Position as TryInto<usize>>::try_into(self.left).unwrap() - 1),
            Strand::Reverse => Some(<Position as TryInto<usize>>::try_into(self.right).unwrap() - 1),
            _ => None,
        }
    }
}

pub fn read_transcripts<R>(input: R) -> Vec<Transcript>
where
    R: BufRead, 
{
    Reader::new(input).records().flat_map(|r| {
        let record = r.unwrap();
        if record.ty() == "transcript" {
            let err_msg = |x: &str| -> String {
                format!("failed to find '{}' in record: {}", x, record)
            };
            let left = record.start();
            let right = record.end();
            let attributes: HashMap<&str, &str> = record.attributes().iter()
                .map(|x| (x.key(), x.value())).collect();
            Some(Transcript {
                transcript_name: attributes.get("transcript_name")
                        .map(|x| x.to_string()),
                transcript_id: attributes.get("transcript_id")
                    .expect(&err_msg("transcript_id"))
                    .to_string(),
                gene_name: attributes.get("gene_name")
                    .expect(&err_msg("gene_name"))
                    .to_string(),
                gene_id: attributes.get("gene_id")
                    .expect(&err_msg("gene_id"))
                    .to_string(),
                is_coding: attributes.get("transcript_type")
                    .map(|x| *x == "protein_coding"),
                chrom: record.reference_sequence_name().to_string(),
                left, right, strand: record.strand(),
            })
        } else {
            None
        }
    }).collect()
}

pub struct Promoters {
    pub regions: GenomeRegions<GenomicRange>,
    pub transcripts: Vec<Transcript>,
}

impl Promoters {
    pub fn new(
        transcripts: Vec<Transcript>,
        upstream: u64,
        downstream: u64,
        include_gene_body: bool,
    ) -> Self
    {
        let regions = transcripts.iter().map(|transcript| {
            let left = (<Position as TryInto<usize>>::try_into(transcript.left).unwrap() - 1) as u64;
            let right = (<Position as TryInto<usize>>::try_into(transcript.right).unwrap() - 1) as u64;
            let (start, end) = match transcript.strand {
                Strand::Forward => (
                    left.saturating_sub(upstream),
                    downstream + (if include_gene_body { right } else { left })
                ),
                Strand::Reverse => (
                    (if include_gene_body { left } else { right }).saturating_sub(downstream),
                    right + upstream
                ),
                _ => panic!("Miss strand information for {}", transcript.transcript_id),
            };
            GenomicRange::new(transcript.chrom.clone(), start, end)
        }).collect();
        Promoters { regions, transcripts }
    }
}

#[derive(Clone)]
pub struct TranscriptCount<'a> {
    counter: SparseCoverage<'a, GenomicRange, u32>,
    promoters: &'a Promoters,
}

impl<'a> TranscriptCount<'a> {
    pub fn new(promoters: &'a Promoters) -> Self {
        Self {
            counter: SparseCoverage::new(&promoters.regions),
            promoters,
        }
    }
    
    pub fn gene_names(&self) -> Vec<String> {
        self.promoters.transcripts.iter().map(|x| x.gene_name.clone()).collect()
    }
}

impl FeatureCounter for TranscriptCount<'_> {
    type Value = u32;

    fn reset(&mut self) { self.counter.reset(); }

    fn insert<B: BEDLike, N: ToPrimitive + Copy>(&mut self, tag: &B, count: N) {
        self.counter.insert(tag, <u32 as NumCast>::from(count).unwrap());
    }

    fn get_feature_ids(&self) -> Vec<String> {
        self.promoters.transcripts.iter().map(|x| x.transcript_id.clone()).collect()
    }

    fn get_counts(&self) -> Vec<(usize, Self::Value)> {
        self.counter.get_counts()
    }
}

#[derive(Clone)]
pub struct GeneCount<'a> {
    counter: TranscriptCount<'a>,
    gene_name_to_idx: IndexMap<&'a str, usize>,
}

impl<'a> GeneCount<'a> {
    pub fn new(counter: TranscriptCount<'a>) -> Self {
        let gene_name_to_idx: IndexMap<_, _> = counter.promoters.transcripts.iter()
            .map(|x| x.gene_name.as_str()).collect::<HashSet<_>>().into_iter()
            .enumerate().map(|(a,b)| (b,a)).collect();
        Self { counter, gene_name_to_idx }
    }
}

impl FeatureCounter for GeneCount<'_> {
    type Value = u32;

    fn reset(&mut self) { self.counter.reset(); }

    fn insert<B: BEDLike, N: ToPrimitive + Copy>(&mut self, tag: &B, count: N) {
        self.counter.insert(tag, <u32 as NumCast>::from(count).unwrap());
    }

    fn get_feature_ids(&self) -> Vec<String> {
        self.gene_name_to_idx.keys().map(|x| x.to_string()).collect()
    }

    fn get_counts(&self) -> Vec<(usize, Self::Value)> {
        let mut counts = BTreeMap::new();
        self.counter.get_counts().into_iter().for_each(|(k, v)| {
            let idx = *self.gene_name_to_idx.get(
                self.counter.promoters.transcripts[k].gene_name.as_str()
            ).unwrap();
            let current_v = counts.entry(idx).or_insert(v);
            if *current_v < v { *current_v = v }
        });
        counts.into_iter().collect()
    }
}


/// Genomic interval associating with integer values
pub type ChromValues<N> = Vec<BedGraph<N>>;

/// GenomeIndex stores genomic loci in a compact way. It maps
/// integers to genomic intervals.
pub trait GenomeIndex {
    fn lookup_region(&self, i: usize) -> GenomicRange;
}

/// Base-resolution compact representation of genomic, stored as a vector of
/// chromosome names and their accumulated lengths.
/// The accumulated lengths start from 0.
pub struct GBaseIndex(Vec<(String, u64)>);

impl GBaseIndex {
    pub fn read_from_anndata(elems: &mut ElemCollection) -> Result<Self> {
        let (chrs, chr_sizes): (Vec<_>, Vec<_>) = get_reference_seq_info_(elems)?.into_iter().unzip();
        let chrom_index = chrs.into_iter().zip(
            std::iter::once(0).chain(chr_sizes.into_iter().scan(0, |state, x| {
                *state = *state + x;
                Some(*state)
            }))
        ).collect();
        Ok(Self(chrom_index))
    }

    pub(crate) fn index_downsampled(&self, ori_idx: usize, sample_size: usize) -> usize {
        if sample_size <= 1 {
            ori_idx
        } else { 
            match self.0.binary_search_by_key(&ori_idx, |s| s.1.try_into().unwrap()) {
                Ok(_) => ori_idx,
                Err(j) => {
                    let p: usize = self.0[j - 1].1.try_into().unwrap();
                    (ori_idx - p) / sample_size * sample_size + p
                },
            }
        }
    }
}

impl GenomeIndex for GBaseIndex {
    fn lookup_region(&self, i: usize) -> GenomicRange {
        match self.0.binary_search_by_key(&i, |s| s.1.try_into().unwrap()) {
            Ok(j) => GenomicRange::new(self.0[j].0.clone(), 0, 1),
            Err(j) => {
                let (chr, p) = self.0[j - 1].clone();
                GenomicRange::new(chr, i as u64 - p, i as u64 - p + 1)
            },
        }
    }
}

/// A set of genomic loci.
pub struct GIntervalIndex(pub Vec<GenomicRange>);

impl GenomeIndex for GIntervalIndex {
    fn lookup_region(&self, i: usize) -> GenomicRange { self.0[i].clone() }
}

pub struct ChromValueIter<I, G, N> {
    iter: I,
    genome_index: G,
    length: usize,
    phantom: PhantomData<N>,
}

impl<I, G, N> Iterator for ChromValueIter<I, G, N>
where
    I: Iterator<Item = Box<dyn DataPartialIO>>,
    G: GenomeIndex,
    N: NumCast + Copy,
{
    type Item = Vec<ChromValues<N>>;

    fn next(&mut self) -> Option<Self::Item> {
        macro_rules! convert {
            ($x:expr, $ty:tt) => {
                $x.into_any().downcast::<CsrMatrix<$ty>>().unwrap().row_iter().map(|row|
                    row.col_indices().iter().zip(row.values()).map(|(i, v)|
                        BedGraph::from_bed(&self.genome_index.lookup_region(*i), N::from(*v).unwrap())).collect()
                ).collect()
            }
        }
        
        self.iter.next().map(|x| match x.get_dtype() {
            DataType::CsrMatrix(ty) => match ty {
                Integer(U1) => convert!(x, i8),
                Integer(U2) => convert!(x, i16),
                Integer(U4) => convert!(x, i32),
                Integer(U8) => convert!(x, i64),
                Unsigned(U1) => convert!(x, u8),
                Unsigned(U2) => convert!(x, u16),
                Unsigned(U4) => convert!(x, u32),
                Unsigned(U8) => convert!(x, u64),
                _ => todo!(),
            },
            _ => todo!(),
        })
    }
}

impl<I, G, N> ExactSizeIterator for ChromValueIter<I, G, N>
where
    I: Iterator<Item = Box<dyn DataPartialIO>>,
    G: GenomeIndex,
    N: NumCast + Copy,
{
    fn len(&self) -> usize { self.length }
}

pub type BaseCountIterator<N> = ChromValueIter<Box<dyn Iterator<Item = Box<dyn DataPartialIO>>>, GBaseIndex, N>;
pub type ChromValueIterator<N> = ChromValueIter<Box<dyn Iterator<Item = Box<dyn DataPartialIO>>>, GIntervalIndex, N>;

/// Read genomic region and its associated account
pub trait ChromValuesReader {
    /// Return values in .obsm['insertion']
    fn raw_count_iter(&self, chunk_size: usize) -> Result<BaseCountIterator<u8>>;

    /// Return values in .X
    fn read_chrom_values<N: NumCast + Copy>(&self) -> Result<ChromValueIterator<N>>;

    /// Return chromosome names and sizes.
    fn get_reference_seq_info(&self) -> Result<Vec<(String, u64)>>;
}


impl ChromValuesReader for AnnData {
    fn raw_count_iter(&self, chunk_size: usize) -> Result<BaseCountIterator<u8>> {
       Ok(ChromValueIter {
            iter: Box::new(self.get_obsm().inner().get("insertion")
                .expect("cannot find 'insertion' in .obsm")
                .chunked(chunk_size)
            ),
            genome_index: GBaseIndex::read_from_anndata(&mut self.get_uns().inner())?,
            length: div_ceil(self.n_obs(), chunk_size),
            phantom: PhantomData,
        })
    }

    fn read_chrom_values<N: NumCast + Copy>(&self) -> Result<ChromValueIterator<N>>
    {
        let chunk_size = 500;
        let genome_index = GIntervalIndex(
            self.var_names()?.into_iter().map(|x| GenomicRange::from_str(x.as_str()).unwrap()).collect()
        );
        Ok(ChromValueIter {
            genome_index,
            iter: Box::new(self.get_x().chunked(chunk_size)),
            length: div_ceil(self.n_obs(), chunk_size),
            phantom: PhantomData,
        })
    }

    fn get_reference_seq_info(&self) -> Result<Vec<(String, u64)>> {
        get_reference_seq_info_(&mut self.get_uns().inner())
    }
}

impl ChromValuesReader for AnnDataSet {
    fn raw_count_iter(&self, chunk_size: usize) -> Result<BaseCountIterator<u8>> {
        let n = self.n_obs();
        let inner = self.anndatas.inner();
        let ref_seq_same = inner.iter().map(|(_, adata)|
            get_reference_seq_info_(&mut adata.get_uns().inner()).unwrap()
        ).all_equal();
        if !ref_seq_same {
            return Err(anyhow!("reference genome information mismatch"));
        }
        let genome_index = GBaseIndex::read_from_anndata(
            &mut inner.iter().next().unwrap().1.get_uns().inner()
        )?;

        Ok(ChromValueIter {
            iter: Box::new(inner.obsm.data.get("insertion").unwrap().chunked(chunk_size)),
            genome_index,
            length: div_ceil(n, chunk_size),
            phantom: PhantomData,
        })
    }

    fn read_chrom_values<N: NumCast + Copy>(&self) -> Result<ChromValueIterator<N>>
    {
        let n = self.n_obs();
        let chunk_size = 500;
        let genome_index = GIntervalIndex(
            self.var_names()?.into_iter().map(|x| GenomicRange::from_str(x.as_str()).unwrap()).collect()
        );
        Ok(ChromValueIter {
            genome_index,
            iter: Box::new(self.anndatas.inner().x.chunked(chunk_size)),
            length: div_ceil(n, chunk_size),
            phantom: PhantomData,
        })
    }

    fn get_reference_seq_info(&self) -> Result<Vec<(String, u64)>> {
        get_reference_seq_info_(&mut self.anndatas.inner().iter().next().unwrap()
            .1.get_uns().inner())
    }
}

fn get_reference_seq_info_(elems: &mut ElemCollection) -> Result<Vec<(String, u64)>> {
    match elems.get_mut("reference_sequences") {
        None => bail!("Cannot find key 'reference_sequences' in: {}", elems),
        Some(ref_seq) => {
            let df: Box<DataFrame> = ref_seq.read()?.into_any().downcast().unwrap();
            let chrs = df.column("reference_seq_name").unwrap().utf8()?;
            let chr_sizes = df.column("reference_seq_length").unwrap().u64()?;
            Ok(chrs.into_iter().flatten().map(|x| x.to_string()).zip(
                chr_sizes.into_iter().flatten()
            ).collect())
        },
    }
}

/// Create cell by feature matrix, and compute qc matrix.
/// 
/// # Arguments
/// 
/// * `file` - 
/// * `fragments` -
/// * `promoter` -
/// * `region` -
/// * `bin_size` -
/// * `min_num_fragment` -
/// * `min_tsse` -
pub fn create_feat_matrix<C, I, N>(
    anndata: &AnnData,
    insertions: I,
    feature_counter: C,
    ) -> Result<()>
where
    C: FeatureCounter<Value = u32> + Clone + Sync,
    I: Iterator<Item = Vec<ChromValues<N>>> + ExactSizeIterator,
    N: ToPrimitive + Copy + Send,
{
    let features = feature_counter.get_feature_ids();
    let style = ProgressStyle::with_template(
        "[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} (eta: {eta})"
    ).unwrap();
    anndata.set_x_from_row_iter(CsrIterator {
        iterator: insertions.progress_with_style(style).map(|chunk|
            chunk.into_par_iter().map(|ins| {
                let mut counter = feature_counter.clone();
                counter.inserts(ins);
                counter.get_counts()
            }).collect::<Vec<_>>()
        ),
        num_cols: features.len(),
    })?;

    anndata.set_var(Some(
        &DataFrame::new(vec![Series::new("Feature_ID", features)]).unwrap()
    ))?;

    Ok(())
}

/// Create cell by bin matrix, and compute qc matrix.
/// 
/// # Arguments
/// 
/// * `file` - 
/// * `fragments` -
/// * `promoter` -
/// * `region` -
/// * `bin_size` -
/// * `min_num_fragment` -
/// * `min_tsse` -
pub fn create_tile_matrix(
    anndata: &AnnData,
    bin_size: u64,
    chunk_size: usize,
    ) -> Result<()>
where
{
    let df: Box<DataFrame> = {
        anndata.get_uns().inner().get_mut("reference_sequences")
            .expect("No reference sequence information is available in the anndata object")
            .read()?.into_any().downcast().unwrap()
    };
    let regions = df.column("reference_seq_length")
        .unwrap().u64().unwrap().into_iter()
        .zip(df.column("reference_seq_name").unwrap().utf8().unwrap())
        .map(|(s, chr)| GenomicRange::new(chr.unwrap(), 0, s.unwrap())).collect();
    let feature_counter: SparseBinnedCoverage<'_, _, u32> =
        SparseBinnedCoverage::new(&regions, bin_size);
    create_feat_matrix(
        anndata,
        anndata.raw_count_iter(chunk_size)?,
        feature_counter,
    )?;
    Ok(())
}

pub fn create_peak_matrix<I, N>(
    output: &str,
    fragments: I,
    peaks: &GenomeRegions<GenomicRange>,
    ) -> Result<AnnData>
where
    I: Iterator<Item = Vec<ChromValues<N>>> + ExactSizeIterator,
    N: ToPrimitive + Copy + Send,
{
    let mut anndata = AnnData::new(output, 0, 0)?;
    let feature_counter: SparseCoverage<'_, _, u32> = SparseCoverage::new(&peaks);
    create_feat_matrix(&mut anndata, fragments, feature_counter)?;
    Ok(anndata)
}

pub fn create_gene_matrix<I, N>(
    output: &str,
    fragments: I,
    transcripts: Vec<Transcript>,
    id_type: &str, 
    ) -> Result<AnnData>
where
    I: Iterator<Item = Vec<ChromValues<N>>> + ExactSizeIterator,
    N: ToPrimitive + Copy + Send,
{
    let mut anndata = AnnData::new(output, 0, 0)?;
    let promoters = Promoters::new(transcripts, 2000, 0, true);

    match id_type {
        "transcript" => {
            let feature_counter: TranscriptCount<'_> = TranscriptCount::new(&promoters);
            let gene_names: Vec<String> = feature_counter.gene_names().iter()
                .map(|x| x.clone()).collect();
            create_feat_matrix(&mut anndata, fragments, feature_counter)?;
            let mut var = anndata.get_var().read()?;
            var.insert_at_idx(1, Series::new("gene_name", gene_names))?;
            anndata.set_var(Some(&var))?;
        },
        "gene" => {
            let feature_counter: GeneCount<'_> = GeneCount::new(
                TranscriptCount::new(&promoters)
            );
            create_feat_matrix(&mut anndata, fragments, feature_counter)?;
        },
        _ => panic!("id_type must be 'transcript' or 'gene'"),
    }

    Ok(anndata)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genome_index() {
        let gindex = GBaseIndex(vec![("1".to_owned(), 0), ("2".to_owned(), 13), ("3".to_owned(), 84)]);

        [
            (0, ("1", 0)),
            (12, ("1", 12)),
            (13, ("2", 0)),
            (100, ("3", 16)),
        ].into_iter().for_each(|(i, (chr, s))|
            assert_eq!(gindex.lookup_region(i), GenomicRange::new(chr, s, s+1))
        );

        [
            (0, 2, 0),
            (1, 2, 0),
            (2, 2, 2),
            (3, 2, 2),
            (10, 2, 10),
            (11, 2, 10),
            (12, 2, 12),
            (13, 2, 13),
            (14, 2, 13),
            (15, 2, 15),
            (16, 2, 15),
            (84, 2, 84),
            (85, 2, 84),
            (86, 2, 86),
            (87, 2, 86),
            (85, 1, 85),
        ].into_iter().for_each(|(i, s, i_)|
            assert_eq!(gindex.index_downsampled(i, s), i_)
        );
    }

    #[test]
    fn test_read_transcript() {
        let input = "chr1\tHAVANA\tgene\t11869\t14409\t.\t+\t.\tgene_id=ENSG00000223972.5;gene_type=transcribed_unprocessed_pseudogene;gene_name=DDX11L1;level=2;hgnc_id=HGNC:37102;havana_gene=OTTHUMG00000000961.2\n\
                     chr1\tHAVANA\ttranscript\t11869\t14409\t.\t+\t.\tgene_id=ENSG00000223972.5;transcript_id=ENST00000456328.2;gene_type=transcribed_unprocessed_pseudogene;gene_name=DDX11L1;transcript_type=processed_transcript;transcript_name=DDX11L1-202;level=2;transcript_support_level=1\n\
                     chr1\tHAVANA\texon\t11869\t12227\t.\t+\t.\tgene_id=ENSG00000223972.5;transcript_id=ENST00000456328.2;gene_type=transcribed_unprocessed_pseudogene;gene_name=DDX11L1;transcript_type=processed_transcript;transcript_name=DDX11L1-202;exon_number=1\n\
                     chr1\tHAVANA\texon\t12613\t12721\t.\t+\t.\tgene_id=ENSG00000223972.5;transcript_id=ENST00000456328.2;gene_type=transcribed_unprocessed_pseudogene;gene_name=DDX11L1;transcript_type=processed_transcript;transcript_name=DDX11L1-202;exon_number=2\n\
                     chr1\tHAVANA\texon\t13221\t14409\t.\t+\t.\tgene_id=ENSG00000223972.5;transcript_id=ENST00000456328.2;gene_type=transcribed_unprocessed_pseudogene;gene_name=DDX11L1;transcript_type=processed_transcript;transcript_name=DDX11L1-202;exon_number=3";
        let expected = Transcript {
            transcript_name: Some("DDX11L1-202".to_string()),
            transcript_id: "ENST00000456328.2".to_string(),
            gene_name: "DDX11L1".to_string(),
            gene_id: "ENSG00000223972.5".to_string(),
            is_coding: Some(false),
            chrom: "chr1".to_string(),
            left: Position::try_from(11869).unwrap(),
            right: Position::try_from(14409).unwrap(),
            strand: Strand::Forward,
        };
        assert_eq!(read_transcripts(input.as_bytes())[0], expected)
    }
}