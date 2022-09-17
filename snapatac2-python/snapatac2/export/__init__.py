from pathlib import Path
import numpy as np
from typing import Optional, Sequence, Union, Set, Mapping

from snapatac2._snapatac2 import AnnData, AnnDataSet
import snapatac2._snapatac2 as internal

def export_bed(
    adata: Union[AnnData, AnnDataSet],
    groupby: Union[str, Sequence[str]],
    selections: Optional[Set[str]] = None,
    ids: Optional[Union[str, np.ndarray]] = None,
    out_dir: Path = "./",
    prefix: str = "",
    suffix: str = ".bed.gz",
) -> Mapping[str, str]:
    """
    Export base-resolution TN5 insertion sites as BED format file.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
    groupby
        Group the cells. If a `str`, groups are obtained from
        `.obs[groupby]`.
    selections
        Export only the selected groups.
    ids
        Cell ids add to the bed records, default to the `.obs_names`.
    out_dir
        Directory for saving the outputs.
    prefix
        Text that adds to the output file name.
    suffix
        Text that adds to the output file name.

    Returns
    -------
    A dictionary contains `(groupname, filename)` pairs. The file names are
    formatted as `{prefix}{groupname}{suffix}`.
    """
    if isinstance(groupby, str):
        groupby = adata.obs[groupby].astype("str")
    
    if ids is None:
        ids = adata.obs[:, 0].astype("str")
    elif isinstance(ids, str):
        ids = adata.obs[ids].astype("str")

    return internal.export_bed(
        adata, list(ids), list(groupby), selections, str(out_dir), prefix, suffix,
    )


def export_bigwig(
    adata: Union[AnnData, AnnDataSet],
    groupby: Union[str, Sequence[str]],
    selections: Optional[Set[str]] = None,
    resolution: int = 1,
    out_dir: Path = "./",
    prefix: str = "",
    suffix: str = ".bw",
) -> Mapping[str, str]:
    """
    Export TN5 insertions as BigWig format file.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
    groupby
        Group the cells. If a `str`, groups are obtained from
        `.obs[groupby]`.
    selections
        Export only the selected groups.
    resolution
        resolution.
    out_dir
        Directory for saving the outputs.
    prefix
        Text that adds to the output file name.
    suffix
        Text that adds to the output file name.

    Returns
    -------
    A dictionary contains `(groupname, filename)` pairs. The file names are
    formatted as `{prefix}{groupname}{suffix}`.
    """
    if isinstance(groupby, str):
        groupby = adata.obs[groupby].astype("str")
    
    return internal.export_bigwig(
        adata, list(groupby), selections, resolution, str(out_dir), prefix, suffix,
    )