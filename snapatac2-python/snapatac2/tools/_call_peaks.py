from __future__ import annotations

from pathlib import Path
from snapatac2._snapatac2 import AnnData, AnnDataSet
import snapatac2._snapatac2 as _snapatac2

def call_peaks(
    adata: AnnData | AnnDataSet,
    groupby: str | list[str],
    selections: set[str] | None = None,
    q_value: float = 0.05,
    out_dir: Path | None = None,
    key_added: str = 'peaks',
    inplace: bool = True,
) -> 'polars.DataFrame' | None:
    """
    Call peaks using MACS2.

    Use the `callpeak` command in MACS2 to identify regions enriched with TN5
    insertions. The parameters passed to MACS2 are:
    "-shift -100 -extsize 200 -nomodel -callsummits -nolambda -keep-dup all"

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
    groupby
        Group the cells before peak calling. If a `str`, groups are obtained from
        `.obs[groupby]`.
    selections
        Call peaks for the selected groups only.
    q_value
        q_value cutoff used in MACS2.
    out_dir
        If provided, raw peak files from each group will be saved in the directory.
        Otherwise, they will be stored in a temporary directory which will be removed
        afterwards.
    key_added
        `.uns` key under which to add the peak information.
    inplace
        Whether to store the result inplace.

    Returns
    -------
    'polars.DataFrame' | None
        If `inplace=True` it stores the result in `adata.uns[`key_added`]`.
        Otherwise, it returns the result as a dataframe.
    """
    if isinstance(groupby, str):
        groupby = list(adata.obs[groupby])
    out_dir = out_dir if out_dir is None else str(out_dir)
    res = _snapatac2.call_peaks(adata, groupby, selections, q_value, out_dir)
    if inplace:
        if adata.isbacked:
            adata.uns[key_added] = res
        else:
            adata.uns[key_added] = res.to_pandas()
    else:
        return res