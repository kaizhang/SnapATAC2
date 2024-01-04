from __future__ import annotations

from pathlib import Path
from anndata import AnnData
import snapatac2._snapatac2 as internal
from scipy.sparse import csr_matrix

def read_10x_mtx(
    path: Path,
    file: Path | None = None,
    prefix: str | None = None,
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
    file
        File name of the ".h5ad" file used to save the AnnData object. If `None`,
        an in-memory AnnData object is returned.
    prefix
        Any prefix before `matrix.mtx`, `genes.tsv` and `barcodes.tsv`. For instance,
        if the files are named `patientA_matrix.mtx`, `patientA_genes.tsv` and
        `patientA_barcodes.tsv`, then the prefix is `patientA_`.

    Returns
    -------
    AnnData
        An AnnData object.
    """
    import pandas as pd

    def get_files(prefix, names):
        return list(filter(
            lambda x: x.is_file(),
            map(lambda x: Path(path + "/" + prefix + x), names)
        ))

    prefix = "" if prefix is None else prefix

    matrix_files = get_files(prefix, ["matrix.mtx", "matrix.mtx.gz"])
    n = len(matrix_files)
    if n == 1:
        mat = csr_matrix(internal.read_mtx(str(matrix_files[0])).X[:].T)
        adata = AnnData(X=mat) if file is None else internal.AnnData(X=mat, filename=file)
    else:
        raise ValueError("Expecting a single 'matrix.mtx' or 'matrix.mtx.gz' file, but found {}.".format(n))

    feature_files = get_files(
        prefix,
        ["genes.tsv", "genes.tsv.gz", "features.tsv", "features.tsv.gz"]
    )
    n = len(feature_files)
    if n == 1:
        df = pd.read_csv(str(feature_files[0]), sep='\t', header=None, index_col=0)
        df.index.name = "index"
        adata.var_names = df.index
        adata.var = df
    else:
        raise ValueError("Expecting a single feature file, but found {}.".format(n))

    barcode_files = get_files(prefix, ["barcodes.tsv", "barcodes.tsv.gz"])
    n = len(barcode_files)
    if n == 1:
        df = pd.read_csv(str(barcode_files[0]), sep='\t', header=None, index_col=0)
        df.index.name = "index"
        adata.obs_names = df.index
        adata.obs = df
    else:
        raise ValueError("Expecting a single barcode file, but found {}.".format(n))

    return adata