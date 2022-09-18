from __future__ import annotations

from pathlib import Path
import pooch

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

_datasets = pooch.create(
    path=pooch.os_cache("snapatac2"),
    base_url="http://renlab.sdsc.edu/kai/public_datasets/",
    # The registry specifies the files that can be fetched
    registry={
        "atac_pbmc_5k.tsv.gz": "sha256:5fe44c0f8f76ce1534c1ae418cf0707ca5ef712004eee77c3d98d2d4b35ceaec",
        "colon_transverse.tar": "sha256:18c56bf405ec0ef8e0e2ea31c63bf2299f21bcb82c67f46e8f70f8d71c65ae0e",
        "HEA_cCRE.bed.gz": "sha256:d69ae94649201cd46ffdc634852acfccc317196637c1786aba82068618001408",

        # Genome files
        "gencode_v19_GRCh37.gff3.gz": "sha256:bb292f6df072e78116c74e3545d8fd8ec7adb4bed2f22fb2715e794400b5c460",
        "gencode_v41_GRCh38.gff3.gz": "sha256:b82a655bdb736ca0e463a8f5d00242bedf10fa88ce9d651a017f135c7c4e9285",
        "gencode_vM25_GRCm38.gff3.gz": "sha256:e8ed48bef6a44fdf0db7c10a551d4398aa341318d00fbd9efd69530593106846",
        "gencode_vM30_GRCm39.gff3.gz": "sha256:6f433e2676e26569a678ce78b37e94a64ddd50a09479e433ad6f75e37dc82e48",
    },
    urls={
        "atac_pbmc_5k.tsv.gz": "http://renlab.sdsc.edu/kai/public_datasets/single_cell_atac/atac_pbmc_5k_nextgem_fragments.tsv.gz",
        "colon_transverse.tar": "http://renlab.sdsc.edu/kai/public_datasets/single_cell_atac/colon_transverse.tar",
        "HEA_cCRE.bed.gz": "http://renlab.sdsc.edu/kai/public_datasets/single_cell_atac/HEA_cCRE.bed.gz",
        "gencode_v19_GRCh37.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/gencode.v19.annotation.gff3.gz",
        "gencode_v41_GRCh38.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/gencode.v41.basic.annotation.gff3.gz",
        "gencode_vM25_GRCm38.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.basic.annotation.gff3.gz",
        "gencode_vM30_GRCm39.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M30/gencode.vM30.basic.annotation.gff3.gz",
    },
)