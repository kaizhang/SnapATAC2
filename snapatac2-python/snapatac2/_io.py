from __future__ import annotations

from pathlib import Path
from snapatac2._snapatac2 import AnnData, read_mtx
from scipy.sparse import csr_matrix

def read_10x_mtx(
    path: str,
    storage: str,
    prefix: str = None,
) -> AnnData:
    """Read 10x-Genomics-formatted mtx directory.

    Parameters
    ----------
    path
        Path to directory for `.mtx` and `.tsv` files. The directory should contain
        three files:

        1. count matrix: "matrix.mtx" or "matrix.mtx.gz".
        2. features: "genes.tsv", or "genes.tsv.gz", or "features.tsv", or "features.tsv.gz".
        3. barcodes: "barcodes.tsv", or "barcodes.tsv.gz".
    storage
        File name of the output ".h5ad" file.
    prefix
        Any prefix before `matrix.mtx`, `genes.tsv` and `barcodes.tsv`. For instance,
        if the files are named `patientA_matrix.mtx`, `patientA_genes.tsv` and
        `patientA_barcodes.tsv`, then the prefix is `patientA_`.

    Returns
    -------
    AnnData
        An AnnData object.
    """
    import polars as pl

    def get_files(prefix, names):
        return list(filter(
            lambda x: x.is_file(),
            map(lambda x: Path(path + "/" + prefix + x), names)
        ))

    prefix = "" if prefix is None else prefix

    matrix_files = get_files(prefix, ["matrix.mtx", "matrix.mtx.gz"])
    if len(matrix_files) == 1:
        adata = read_mtx(str(matrix_files[0]), storage)
        mat = csr_matrix(adata.X[:].T)
        adata.X = None
        adata.X = mat
    else:
        raise ValueError("Expecting a single 'matrix.mtx' or 'matrix.mtx.gz' file")

    feature_files = get_files(
        prefix,
        ["genes.tsv", "genes.tsv.gz", "features.tsv", "features.tsv.gz"]
    )
    if len(feature_files) == 1:
        adata.var = pl.read_csv(str(feature_files[0]), sep = '\t', has_header = False)
        
    else:
        raise ValueError("Expecting a single feature file")

    barcode_files = get_files(prefix, ["barcodes.tsv", "barcodes.tsv.gz"])
    if len(barcode_files) == 1:
        adata.obs = pl.read_csv(str(barcode_files[0]), sep = '\t', has_header = False)
    else:
        raise ValueError("Expecting a single barcode file")

    return adata