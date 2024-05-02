from __future__ import annotations
from collections.abc import Callable

from snapatac2.datasets import datasets
from pathlib import Path
from pooch import Decompress

class Genome:
    """
    A class that encapsulates information about a genome, including its FASTA sequence,
    its annotation, and chromosome sizes.

    Attributes
    ----------
    fasta
        The path to the FASTA file.
    annotation
        The path to the annotation file.
    chrom_sizes
        A dictionary containing chromosome names and sizes.

    Raises
    ------
    ValueError
        If `fasta` or `annotation` are not a Path, a string, or a callable.
    """

    def __init__(
        self,
        *,
        fasta: Path | Callable[[], Path], 
        annotation: Path | Callable[[], Path],
        chrom_sizes: dict[str, int] | None = None,
    ):
        """
        Initializes the Genome object with paths or callables for FASTA and annotation files,
        and optionally, chromosome sizes.

        Parameters
        ----------
        fasta
            A Path or callable that returns a Path to the FASTA file.
        annotation
            A Path or callable that returns a Path to the annotation file.
        chrom_sizes
            Optional chromosome sizes. If not provided, chromosome sizes will
            be inferred from the FASTA file.
        """
        if callable(fasta):
            self._fetch_fasta = fasta
            self._fasta = None
        elif isinstance(fasta, Path) or isinstance(fasta, str):
            self._fasta = Path(fasta)
            self._fetch_fasta = None
        else:
            raise ValueError("fasta must be a Path or Callable")

        if callable(annotation):
            self._fetch_annotation = annotation
            self._annotation = None
        elif isinstance(annotation, Path) or isinstance(annotation, str):
            self._annotation = Path(annotation)
            self._fetch_annotation = None
        else:
            raise ValueError("annotation must be a Path or Callable")

        self._chrom_sizes = chrom_sizes

    @property
    def fasta(self):
        """
        The Path to the FASTA file. 

        Returns
        -------
        Path
            The path to the FASTA file.
        """
        if self._fasta is None:
            self._fasta = Path(self._fetch_fasta())
        return self._fasta

    @property
    def annotation(self):
        """
        The Path to the annotation file.

        Returns
        -------
        Path
            The path to the annotation file.
        """
        if self._annotation is None:
            self._annotation = Path(self._fetch_annotation())
        return self._annotation

    @property
    def chrom_sizes(self):
        """
        A dictionary with chromosome names as keys and their lengths as values.

        Returns
        -------
        dict[str, int]
            A dictionary of chromosome sizes.
        """
        if self._chrom_sizes is None:
            from pyfaidx import Fasta
            fasta = Fasta(self.fasta)
            self._chrom_sizes = {chr: len(fasta[chr]) for chr in fasta.keys()}
        return self._chrom_sizes
        
GRCh37 = Genome(
    fasta=lambda : datasets().fetch(
        "gencode_v41_GRCh37.fa.gz", processor=Decompress(method = "gzip"), progressbar=True),
    annotation=lambda : datasets().fetch(
        "gencode_v41_GRCh37.gff3.gz", progressbar=True),
    )
hg19 = GRCh37

GRCh38 = Genome(
    fasta=lambda :  datasets().fetch(
        "gencode_v41_GRCh38.fa.gz", processor=Decompress(method = "gzip"), progressbar=True),
    annotation=lambda : datasets().fetch(
        "gencode_v41_GRCh38.gff3.gz", progressbar=True),
    chrom_sizes= {"chr1": 248956422, "chr2": 242193529, "chr3": 198295559,
                  "chr4": 190214555, "chr5": 181538259, "chr6": 170805979,
                  "chr7": 159345973, "chr8": 145138636, "chr9": 138394717,
                  "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
                  "chr13": 114364328, "chr14": 107043718, "chr15": 101991189,
                  "chr16": 90338345, "chr17": 83257441, "chr18": 80373285,
                  "chr19": 58617616, "chr20": 64444167, "chr21": 46709983,
                  "chr22": 50818468, "chrX": 156040895, "chrY": 57227415,
                  "chrM": 16569 },
    )
hg38 = GRCh38

GRCm39 = Genome(
    fasta=lambda : datasets().fetch(
        "gencode_vM30_GRCm39.fa.gz", processor=Decompress(method = "gzip"), progressbar=True),
    annotation=lambda : datasets().fetch(
        "gencode_vM30_GRCm39.gff3.gz", progressbar=True),
    chrom_sizes={
        "chr1": 195154279,
        "chr2": 181755017,
        "chr3": 159745316,
        "chr4": 156860686,
        "chr5": 151758149,
        "chr6": 149588044,
        "chr7": 144995196,
        "chr8": 130127694,
        "chr9": 124359700,
        "chr10": 130530862,
        "chr11": 121973369,
        "chr12": 120092757,
        "chr13": 120883175,
        "chr14": 125139656,
        "chr15": 104073951,
        "chr16": 98008968,
        "chr17": 95294699,
        "chr18": 90720763,
        "chr19": 61420004,
        "chrX": 169476592,
        "chrY": 91455967,
        "chrM": 16299,
    },
    )
mm39 = GRCm39

GRCm38 = Genome(
    fasta=lambda : datasets().fetch(
        "gencode_vM25_GRCm38.fa.gz", processor=Decompress(method = "gzip"), progressbar=True),
    annotation=lambda : datasets().fetch(
        "gencode_vM25_GRCm38.gff3.gz", progressbar=True),
    chrom_sizes={
        "chr1": 195471971,
        "chr2": 182113224,
        "chr3": 160039680,
        "chr4": 156508116,
        "chr5": 151834684,
        "chr6": 149736546,
        "chr7": 145441459,
        "chr8": 129401213,
        "chr9": 124595110,
        "chr10": 130694993,
        "chr11": 122082543,
        "chr12": 120129022,
        "chr13": 120421639,
        "chr14": 124902244,
        "chr15": 104043685,
        "chr16": 98207768,
        "chr17": 94987271,
        "chr18": 90702639,
        "chr19": 61431566,
        "chrX": 171031299,
        "chrY": 91744698,
        "chrM": 16299,
    },
    )
mm10 = GRCm38