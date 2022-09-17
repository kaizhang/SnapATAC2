from __future__ import annotations
from typing_extensions import Literal

from pathlib import Path
import numpy as np
import polars as pl
import functools
from natsort import natsorted

from snapatac2._snapatac2 import AnnData, AnnDataSet

def aggregate_X(
    adata: AnnData | AnnDataSet,
    groupby: str | list[str] | None = None,
    normalize: Literal["RPM", "RPKM"] | None = None,
    file: Path | None = None,
) -> np.ndarray | dict[str, np.ndarray] | AnnData:
    """
    Aggregate values in adata.X in a row-wise fashion.

    Aggregate values in adata.X in a row-wise fashion. This is used to compute
    RPKM or RPM values stratified by user-provided groupings.

    Parameters
    ----------
    adata
        The AnnData or AnnDataSet object.
    groupby
        Group the cells into different groups. If a `str`, groups are obtained from
        `.obs[groupby]`.
    normalize
        normalization method: "RPM" or "RPKM".
    file
        if provided, the results will be saved to a new h5ad file.

    Returns
    -------
        If file=None, return the result as numpy array.
    """
    def norm(x):
        if normalize is None:
            return x
        elif normalize == "RPKM":
            size_factor = _get_sizes(adata.var_names) / 1000.0
            return _normalize(x, size_factor)
        elif normalize == "RPM":
            return _normalize(x)
        else:
            raise NameError("Normalization method must be 'RPKM' or 'RPM'")

    if groupby is None:
        row_sum = functools.reduce(
            lambda a, b: a + b,
            (np.ravel(chunk.sum(axis=0)) for chunk in adata.X.chunked(1000)),
        )
        row_sum = norm(row_sum)

        if file is None:
            return row_sum
        else:
            out_adata = AnnData(
                filename = file,
                X = np.array([row_sum]),
                var = None if adata.var is None else adata.var[:],
            )
            return out_adata
    else:
        groups = adata.obs[groupby] if isinstance(groupby, str) else np.array(groupby)
        if len(groups) != adata.n_obs:
            raise NameError("the length of `groupby` should equal to the number of obervations")

        cur_row = 0
        result = {}
        for chunk in adata.X.chunked(2000):
            n = chunk.shape[0]
            labels = groups[cur_row:cur_row+n]
            for key, mat in _groupby(chunk, labels).items():
                s = np.ravel(mat.sum(axis = 0))
                if key in result:
                    result[key] += s
                else:
                    result[key] = s
            cur_row = cur_row + n
        for k, v in result.items():
            result[k] = norm(v)

        result = natsorted(result.items())
        if file is None:
            return dict(result)
        else:
            keys, values = zip(*result)
            column_name = groupby if isinstance(groupby, str) else "_index"
            out_adata = AnnData(
                filename = file,
                X = np.array(values),
                obs = pl.DataFrame({ column_name: np.array(keys) }),
                var = None if adata.var is None else adata.var[:],
            )
            return out_adata

def _groupby(x, groups):
    idx = groups.argsort()
    groups = groups[idx]
    x = x[idx]
    u, indices = np.unique(groups, return_index=True)
    splits = np.split(np.arange(x.shape[0]), indices[1:])
    return dict((label, x[indices, :]) for (label, indices) in zip(u, splits))

def _normalize(x, size_factor = None):
    result = x / (x.sum() / 1000000.0)
    if size_factor is not None:
        result /= size_factor
    return result

def _get_sizes(regions):
    def size(x):
        x = x.split(':')[1].split("-")
        return int(x[1]) - int(x[0])
    return np.array(list(size(x) for x in regions))