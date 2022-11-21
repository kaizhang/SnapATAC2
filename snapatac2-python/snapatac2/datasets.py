from __future__ import annotations
from typing_extensions import Literal

from pathlib import Path
import pooch

from snapatac2._snapatac2 import read_motifs, PyDNAMotif

# This is a global variable used to store all datasets. It is initialized only once
# when the data is requested.
_datasets = None

def datasets():
    global _datasets
    if _datasets is None:
        _datasets = pooch.create(
            path=pooch.os_cache("snapatac2"),
            base_url="http://renlab.sdsc.edu/kai/public_datasets/",
            env="SNAP_DATA_DIR",  # The user can overwrite the storage path by setting this environment variable.
            # The registry specifies the files that can be fetched
            registry={
                "atac_pbmc_500.tsv.gz": "sha256:196c5d7ee0169957417e9f4d5502abf1667ef99453328f8d290d4a7f3b205c6c",
                "atac_pbmc_500_downsample.tsv.gz": "sha256:6053cf4578a140bfd8ce34964602769dc5f5ec6b25ba4f2db23cdbd4681b0e2f",
                "atac_pbmc_5k.tsv.gz": "sha256:5fe44c0f8f76ce1534c1ae418cf0707ca5ef712004eee77c3d98d2d4b35ceaec",
                "atac_pbmc_5k.h5ad": "sha256:dcaca8ca4ac28674ec2172b4a975f75fba2ede1fc86571f7c452ba00f5cd4b94",
                "atac_pbmc_5k_annotated.h5ad": "sha256:3d5f147ce13a01cd2bdc3d9d2e8cf7897ee98e44255ff12f868517dd78427a87",
                "atac_pbmc_5k_gene.h5ad": "sha256:333f08af090c3306c681d26cce93614a71fee2a12d268b54ef1fce29fda8f078",

                "colon_transverse.tar": "sha256:18c56bf405ec0ef8e0e2ea31c63bf2299f21bcb82c67f46e8f70f8d71c65ae0e",
                "HEA_cCRE.bed.gz": "sha256:d69ae94649201cd46ffdc634852acfccc317196637c1786aba82068618001408",
                "cisBP_human.meme": "sha256:8bf995450258e61cb1c535d5bf9656d580eb68ba68893fa36b77d17ee0730579",

                "10x-Multiome-Pbmc10k-ATAC.h5ad": "sha256:24d030fb7f90453a0303b71a1e3e4e7551857d1e70072752d7fff9c918f77217",
                "10x-Multiome-Pbmc10k-RNA.h5ad": "sha256:a25327acff48b20b295c12221a84fd00f8f3f486ff3e7bd090fdef241b996a22",

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
                "atac_pbmc_500.tsv.gz": "https://cf.10xgenomics.com/samples/cell-atac/2.0.0/atac_pbmc_500_nextgem/atac_pbmc_500_nextgem_fragments.tsv.gz",
                "atac_pbmc_500_downsample.tsv.gz": "http://renlab.sdsc.edu/kai/public_datasets/single_cell_atac/atac_pbmc_500_downsample.tsv.gz",

                "atac_pbmc_5k.tsv.gz": "http://renlab.sdsc.edu/kai/public_datasets/single_cell_atac/atac_pbmc_5k_nextgem_fragments.tsv.gz",
                "atac_pbmc_5k.h5ad": "http://renlab.sdsc.edu/kai/public_datasets/single_cell_atac/atac_pbmc_5k.h5ad",
                "atac_pbmc_5k_gene.h5ad": "http://renlab.sdsc.edu/kai/public_datasets/single_cell_atac/atac_pbmc_5k_gene.h5ad",
                "atac_pbmc_5k_annotated.h5ad": "http://renlab.sdsc.edu/kai/public_datasets/single_cell_atac/atac_pbmc_5k_annotated.h5ad",


                "10x-Multiome-Pbmc10k-ATAC.h5ad": "http://renlab.sdsc.edu/kai/public_datasets/single_cell_multiome/10x-Multiome-Pbmc10k-ATAC.h5ad",
                "10x-Multiome-Pbmc10k-RNA.h5ad": "http://renlab.sdsc.edu/kai/public_datasets/single_cell_multiome/10x-Multiome-Pbmc10k-RNA.h5ad",

                "colon_transverse.tar": "http://renlab.sdsc.edu/kai/public_datasets/single_cell_atac/colon_transverse.tar",
                "HEA_cCRE.bed.gz": "http://renlab.sdsc.edu/kai/public_datasets/single_cell_atac/HEA_cCRE.bed.gz",
                "cisBP_human.meme": "http://renlab.sdsc.edu/kai/public_datasets/cisBP_human_rev_1.meme",

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
    return _datasets

def pbmc500(downsampled: bool = False) -> Path:
    """500 PBMCs from 10x Genomics.

    Parameters
    ----------
    downsampled
        Whether to return downsampled dataset.

    Returns
    -------
    Path
        Path to the fragment file.
    """
    if downsampled:
        return Path(datasets().fetch("atac_pbmc_500_downsample.tsv.gz"))
    else:
        return Path(datasets().fetch("atac_pbmc_500.tsv.gz"))

def pbmc5k(type: Literal["fragment, h5ad, gene, annotated_h5ad"] = "fragment") -> Path:
    """5k PBMCs from 10x Genomics.

    Parameters
    ----------
    type
        One of the following:
            - "fragment": the fragment file.
            - "h5ad": preprocessed h5ad file.
            - "gene": gene activity matrix.
            - "annotated_h5ad": annotated h5ad file.

    Returns
    -------
    Path
        path to the file.
    """
    if type == "fragment":
        return Path(datasets().fetch("atac_pbmc_5k.tsv.gz"))
    elif type == "h5ad":
        return Path(datasets().fetch("atac_pbmc_5k.h5ad"))
    elif type == "annotated_h5ad":
        return Path(datasets().fetch("atac_pbmc_5k_annotated.h5ad"))
    elif type == "gene":
        return Path(datasets().fetch("atac_pbmc_5k_gene.h5ad"))
    else:
        raise NameError("type '{}' is not available.".format(type))
    

def pbmc_multiome(modality: Literal["ATAC", "RNA"] = "RNA") -> Path:
    """10k PBMCs from 10x Genomics.
    """
    if modality == "RNA":
        return Path(datasets().fetch("10x-Multiome-Pbmc10k-RNA.h5ad"))
    elif modality == "ATAC":
        return Path(datasets().fetch("10x-Multiome-Pbmc10k-ATAC.h5ad"))
    else:
        raise NameError("modality '{}' is not available.".format(modality))

def colon() -> list[tuple[str, Path]]:
    """5 colon transverse samples from Zhang et al., 2021.
    """
    files = datasets().fetch("colon_transverse.tar", processor = pooch.Untar())
    return [(fl.split("/")[-1].split("_rep1_fragments")[0], Path(fl)) for fl in files]

def cre_HEA() -> Path:
    """cis-regulatory elements from Zhang et al., 2021.
    """
    return Path(datasets().fetch("HEA_cCRE.bed.gz"))

def cis_bp(unique: bool = True) -> list[PyDNAMotif]:
    """Motifs from CIS-BP database.

    Parameters
    ----------
    unique
        A transcription factor may have multiple motifs. If `unique=True`, 
        only the motifs with the highest information content will be selected.

    Returns
    -------
    list[PyDNAMotif]
        A list of motifs.
    """
    motifs = read_motifs(datasets().fetch("cisBP_human.meme"))
    for motif in motifs:
        motif.name = motif.id.split('+')[0]
    if unique:
        unique_motifs = {}
        for motif in motifs:
            name = motif.name
            if (
                    name not in unique_motifs or 
                    unique_motifs[name].info_content() < motif.info_content()
               ):
               unique_motifs[name] = motif
        motifs = list(unique_motifs.values())
    return motifs