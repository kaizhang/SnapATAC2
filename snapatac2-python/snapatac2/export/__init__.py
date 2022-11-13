from __future__ import annotations

from pathlib import Path
import numpy as np

from snapatac2._snapatac2 import AnnData, AnnDataSet
import snapatac2._snapatac2 as internal

def export_bed(
    adata: AnnData | AnnDataSet,
    groupby: str | list[str],
    selections: list[str] | None = None,
    ids: str | list[str] | None = None,
    out_dir: Path = "./",
    prefix: str = "",
    suffix: str = ".bed.gz",
) -> dict[str, str]:
    """Export base-resolution TN5 insertion sites as BED format file.

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
        Cell ids add to the bed records. If `None`, `.obs_names` is used.
    out_dir
        Directory for saving the outputs.
    prefix
        Text added to the output file name.
    suffix
        Text added to the output file name.

    Returns
    -------
    dict[str, str]
        A dictionary contains `(groupname, filename)` pairs. The file names are
        formatted as `{prefix}{groupname}{suffix}`.
    """
    if isinstance(groupby, str):
        groupby = adata.obs[groupby]
    if selections is not None:
        selections = set(selections)
    
    if ids is None:
        ids = adata.obs_names
    elif isinstance(ids, str):
        ids = adata.obs[ids]

    return internal.export_bed(
        adata, list(ids), list(groupby), selections, out_dir, prefix, suffix,
    )


def export_bigwig(
    adata: AnnData | AnnDataSet,
    groupby: str | list[str],
    selections: list[str] | None = None,
    resolution: int = 1,
    out_dir: Path = "./",
    prefix: str = "",
    suffix: str = ".bw",
) -> dict[str, str]:
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
        Text added to the output file name.
    suffix
        Text added to the output file name.

    Returns
    -------
    dict[str, str]
        A dictionary contains `(groupname, filename)` pairs. The file names are
        formatted as `{prefix}{groupname}{suffix}`.
    """
    if isinstance(groupby, str):
        groupby = adata.obs[groupby]
    if selections is not None:
        selections = set(selections)
    
    return internal.export_bigwig(
        adata, list(groupby), selections, resolution, out_dir, prefix, suffix,
    )