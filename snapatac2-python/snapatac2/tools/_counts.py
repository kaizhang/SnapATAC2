from __future__ import annotations

import numpy as np
import functools

from snapatac2._snapatac2 import AnnData, AnnDataSet

def _group_by(x, groups):
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

def aggregate_X(
    adata: AnnData | AnnDataSet,
    group_by: str | list[str] | None = None,
    normalize: str | None = None,
    inplace: bool = True,
) -> np.ndarray | dict[str, np.ndarray] | None:
    """
    Aggregate values in adata.X in a row-wise fashion.

    Aggregate values in adata.X in a row-wise fashion. This is used to compute
    RPKM or RPM values stratified by user-provided groupings.

    Parameters
    ----------
    adata
        The AnnData or AnnDataSet object.
    group_by
        Group information, as adata.obs[group_by].
    normalize
        normalization method: "RPM" or "RPKM".
    inplace
        if True, results are added to adata.var.

    Returns
    -------
        If inplace=False, return the result as numpy array.
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

    if group_by is None:
        row_sum = functools.reduce(
            lambda a, b: a + b,
            (np.ravel(chunk.sum(axis=0)) for chunk in adata.X.chunked(1000)),
        )
        row_sum = norm(row_sum)

        if inplace:
            adata.var["aggregate_X"] = row_sum
        else:
            return row_sum
    else:
        if isinstance(group_by, list):
            groups = adata.obs[group_by]
            groups = [tuple(groups[i, :]) for i in range(groups.shape[0])]
            out = np.empty(len(groups), dtype=object)
            out[:] = groups
            groups = out
            '''
            groups = np.array(list(
                '+'.join(map(lambda x: str(x), list(groups[i, :]))) for i in range(groups.shape[0])
            ))
            '''
        else:
            groups = adata.obs[group_by]
        cur_row = 0
        result = {}
        for chunk in adata.X.chunked(2000):
            n = chunk.shape[0]
            labels = groups[cur_row:cur_row+n]
            for key, mat in _group_by(chunk, labels).items():
                s = np.ravel(mat.sum(axis = 0))
                if key in result:
                    result[key] += s
                else:
                    result[key] = s
            cur_row = cur_row + n
        for k, v in result.items():
            result[k] = norm(v)
        if inplace:
            var = adata.var[:]
            for k, v in result.items():
                var[k] = v
            adata.var = var
        else:
            return result