from __future__ import annotations

from pathlib import Path
import pooch

from snapatac2._snapatac2 import read_motifs, PyDNAMotif

def pbmc5k() -> Path:
    """
    5k PBMCs from 10x Genomics.
    """
    return Path(_datasets.fetch("atac_pbmc_5k.tsv.gz"))

def colon() -> list[tuple[str, Path]]:
    """
    5 colon transverse samples from Zhang et al., 2021.
    """
    files = _datasets.fetch("colon_transverse.tar", processor = pooch.Untar())
    return [(fl.split("/")[-1].split("_rep1_fragments")[0], Path(fl)) for fl in files]

def cre_HEA() -> Path:
    """
    cis-regulatory elements from Zhang et al., 2021.
    """
    return Path(_datasets.fetch("HEA_cCRE.bed.gz"))

def cis_bp(unique: bool = False) -> list[PyDNAMotif]:
    """
    Motifs from CIS-BP database.
    """
    motifs = read_motifs(_datasets.fetch("cisBP_human.meme"))
    if unique:
        unique_motifs = {}
        for motif in motifs:
            name = motif.name.split('+')[0]
            if (
                    name not in unique_motifs or 
                    unique_motifs[name].info_content() < motif.info_content()
               ):
               unique_motifs[name] = motif
        motifs = list(unique_motifs.values())
    return motifs

_datasets = pooch.create(
    path=pooch.os_cache("snapatac2"),
    base_url="http://renlab.sdsc.edu/kai/public_datasets/",
    # The registry specifies the files that can be fetched
    registry={
        "atac_pbmc_5k.tsv.gz": "sha256:5fe44c0f8f76ce1534c1ae418cf0707ca5ef712004eee77c3d98d2d4b35ceaec",
        "colon_transverse.tar": "sha256:18c56bf405ec0ef8e0e2ea31c63bf2299f21bcb82c67f46e8f70f8d71c65ae0e",
        "HEA_cCRE.bed.gz": "sha256:d69ae94649201cd46ffdc634852acfccc317196637c1786aba82068618001408",
        "cisBP_human.meme": "sha256:bd9eda5000879ab8bc179e4a4c1562bf6e69af34fd16be797c1b665b558e1914",

        # Genome files
        "gencode_v41_GRCh37.gff3.gz": "sha256:df96d3f0845127127cc87c729747ae39bc1f4c98de6180b112e71dda13592673",
        "gencode_v41_GRCh37.fa.gz": "sha256:94330d402e53cf39a1fef6c132e2500121909c2dfdce95cc31d541404c0ed39e",
        "gencode_v41_GRCh38.gff3.gz": "sha256:b82a655bdb736ca0e463a8f5d00242bedf10fa88ce9d651a017f135c7c4e9285",
        "gencode_v41_GRCh38.fa.gz": "sha256:4fac949d7021cbe11117ddab8ec1960004df423d672446cadfbc8cca8007e228",
        "gencode_vM25_GRCm38.gff3.gz": "sha256:e8ed48bef6a44fdf0db7c10a551d4398aa341318d00fbd9efd69530593106846",
        "gencode_vM25_GRCm38.fa.gz": "sha256:617b10dc7ef90354c3b6af986e45d6d9621242b64ed3a94c9abeac3e45f18c17",
        "gencode_vM30_GRCm39.gff3.gz": "sha256:6f433e2676e26569a678ce78b37e94a64ddd50a09479e433ad6f75e37dc82e48",
        "gencode_vM30_GRCm39.fa.gz": "sha256:3b923c06a0d291fe646af6bf7beaed7492bf0f6dd5309d4f5904623cab41b0aa",
    },
    urls={
        "atac_pbmc_5k.tsv.gz": "http://renlab.sdsc.edu/kai/public_datasets/single_cell_atac/atac_pbmc_5k_nextgem_fragments.tsv.gz",
        "colon_transverse.tar": "http://renlab.sdsc.edu/kai/public_datasets/single_cell_atac/colon_transverse.tar",
        "HEA_cCRE.bed.gz": "http://renlab.sdsc.edu/kai/public_datasets/single_cell_atac/HEA_cCRE.bed.gz",
        "cisBP_human.meme": "http://renlab.sdsc.edu/kai/public_datasets/cisBP_human.meme",
        "gencode_v41_GRCh37.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/GRCh37_mapping/gencode.v41lift37.basic.annotation.gff3.gz",
        "gencode_v41_GRCh37.fa.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/GRCh37_mapping/GRCh37.primary_assembly.genome.fa.gz",
        "gencode_v41_GRCh38.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/gencode.v41.basic.annotation.gff3.gz",
        "gencode_v41_GRCh38.fa.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/GRCh38.primary_assembly.genome.fa.gz",
        "gencode_vM25_GRCm38.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.basic.annotation.gff3.gz",
        "gencode_vM25_GRCm38.fa.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/GRCm38.primary_assembly.genome.fa.gz",
        "gencode_vM30_GRCm39.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M30/gencode.vM30.basic.annotation.gff3.gz",
        "gencode_vM30_GRCm39.fa.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M30/GRCm39.primary_assembly.genome.fa.gz",
    },
)
