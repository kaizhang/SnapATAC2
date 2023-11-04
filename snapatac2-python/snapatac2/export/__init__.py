from __future__ import annotations
from typing_extensions import Literal

from pathlib import Path
import numpy as np

import snapatac2._snapatac2 as internal

def export_fragments(
    adata: internal.AnnData | internal.AnnDataSet,
    groupby: str | list[str],
    selections: list[str] | None = None,
    ids: str | list[str] | None = None,
    out_dir: Path = "./",
    prefix: str = "",
    suffix: str = ".bed.zst",
    compression: Literal["gzip", "zstandard"] | None = None,
    compression_level: int | None = None,
) -> dict[str, str]:
    """Export and save fragments in a BED format file.

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
    compression
        Compression type. If `None`, it is inferred from the suffix.
    compression_level
        Compression level. 1-9 for gzip, 1-22 for zstandard.
        If `None`, it is set to 6 for gzip and 3 for zstandard.

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

    if compression is None:
        if suffix.endswith(".gz"):
            compression = "gzip"
        elif suffix.endswith(".zst"):
            compression = "zstandard"

    return internal.export_fragments(
        adata, list(ids), list(groupby), out_dir, prefix, suffix, selections, compression, compression_level,
    )

def export_bedgraph(
    adata: internal.AnnData | internal.AnnDataSet,
    groupby: str | list[str],
    selections: list[str] | None = None,
    resolution: int = 1,
    smooth_length: int | None = None,
    blacklist: Path | None = None,
    normalization: Literal["RPKM", "CPM"] | None = "RPKM",
    ignore_for_norm: list[str] | None = None,
    min_frag_length: int | None = None,
    max_frag_length: int | None = 2000,
    out_dir: Path = "./",
    prefix: str = "",
    suffix: str = ".bedgraph.zst",
    compression: Literal["gzip", "zstandard"] | None = None,
    compression_level: int | None = None,
    tempdir: Path | None = None,
) -> dict[str, str]:
    """Export and save fragments in a BED format file.

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
    compression
        Compression type. If `None`, it is inferred from the suffix.
    compression_level
        Compression level. 1-9 for gzip, 1-22 for zstandard.
        If `None`, it is set to 6 for gzip and 3 for zstandard.

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
    
    if compression is None:
        if suffix.endswith(".gz"):
            compression = "gzip"
        elif suffix.endswith(".zst"):
            compression = "zstandard"
    
    return internal.export_bedgraph(
        adata, list(groupby), resolution, out_dir, prefix, suffix, selections,
        smooth_length, blacklist, normalization, ignore_for_norm, min_frag_length,
        max_frag_length, compression, compression_level, tempdir
    )

def export_bigwig(
    adata: internal.AnnData | internal.AnnDataSet,
    groupby: str | list[str],
    selections: list[str] | None = None,
    resolution: int = 1,
    out_dir: Path = "./",
    prefix: str = "",
    suffix: str = ".bw",
) -> dict[str, str]:
    """
    Export and create BigWig format files.

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