import pooch

def pbmc5k() -> str:
    return _datasets.fetch("atac_pbmc_5k.tsv.gz")

_datasets = pooch.create(
    path=pooch.os_cache("snapatac2"),
    base_url="http://renlab.sdsc.edu/kai/public_datasets/",
    # The registry specifies the files that can be fetched
    registry={
        "atac_pbmc_5k.tsv.gz": "sha256:5fe44c0f8f76ce1534c1ae418cf0707ca5ef712004eee77c3d98d2d4b35ceaec",

        # Genome files
        "gencode_v19_GRCh37.gff3.gz": "sha256:bb292f6df072e78116c74e3545d8fd8ec7adb4bed2f22fb2715e794400b5c460",
        "gencode_v41_GRCh38.gff3.gz": "sha256:b82a655bdb736ca0e463a8f5d00242bedf10fa88ce9d651a017f135c7c4e9285",
        "gencode_vM25_GRCm38.gff3.gz": "sha256:e8ed48bef6a44fdf0db7c10a551d4398aa341318d00fbd9efd69530593106846",
        "gencode_vM30_GRCm39.gff3.gz": "sha256:6f433e2676e26569a678ce78b37e94a64ddd50a09479e433ad6f75e37dc82e48",
    },
    urls={
        "atac_pbmc_5k.tsv.gz": "http://renlab.sdsc.edu/kai/public_datasets/single_cell_atac/atac_pbmc_5k_nextgem_fragments.tsv.gz",
        "gencode_v19_GRCh37.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/gencode.v19.annotation.gff3.gz",
        "gencode_v41_GRCh38.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/gencode.v41.basic.annotation.gff3.gz",
        "gencode_vM25_GRCm38.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.basic.annotation.gff3.gz",
        "gencode_vM30_GRCm39.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M30/gencode.vM30.basic.annotation.gff3.gz",
    },
)