from pathlib import Path
from typing import Union
import pandas as pd
from anndata import (
    AnnData,
    read_mtx,
)

def read_10x_mtx(
    path: Union[Path, str],
    prefix: str = None,
) -> AnnData:
    """
    Read 10x-Genomics-formatted mtx directory.

    Parameters
    ----------
    path
        Path to directory for `.mtx` and `.tsv` files,
    prefix
        Any prefix before `matrix.mtx`, `genes.tsv` and `barcodes.tsv`. For instance,
        if the files are named `patientA_matrix.mtx`, `patientA_genes.tsv` and
        `patientA_barcodes.tsv` the prefix is `patientA_`.
    Returns
    -------
    An :class:`~anndata.AnnData` object
    """

    def get_files(prefix, names):
        return list(filter(
            lambda x: x.is_file(),
            map(lambda x: Path(path + "/" + prefix + x), names)
        ))

    prefix = "" if prefix is None else prefix

    matrix_files = get_files(prefix, ["matrix.mtx", "matrix.mtx.gz"])
    if len(matrix_files) == 1:
        adata = read_mtx(matrix_files[0]).T  # transpose the data
    else:
        raise ValueError("Expecting a single 'matrix.mtx' or 'matrix.mtx.gz' file")

    feature_files = get_files(
        prefix,
        ["genes.tsv", "genes.tsv.gz", "features.tsv", "features.tsv.gz"]
    )
    if len(feature_files) == 1:
        adata.var['regions'] = pd.read_csv(feature_files[0], header=None, sep='\t')[0].values
    else:
        raise ValueError("Expecting a single feature file")

    barcode_files = get_files(prefix, ["barcodes.tsv", "barcodes.tsv.gz"])
    if len(barcode_files) == 1:
        adata.obs_names = pd.read_csv(barcode_files[0], header=None)[0].values
    else:
        raise ValueError("Expecting a single barcode file")

    return adata