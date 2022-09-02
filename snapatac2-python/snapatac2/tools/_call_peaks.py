from __future__ import annotations

from pathlib import Path
from snapatac2._snapatac2 import AnnData, AnnDataSet
import snapatac2._snapatac2 as _snapatac2

def call_peaks(
    data: AnnData | AnnDataSet,
    group_by: str | list[str],
    selections: set[str] | None = None,
    q_value: float = 0.05,
    out_dir: Path | None = None,
    key_added: str = 'peaks',
    inplace: bool = True,
):
    """
    Call peaks using MACS2.

    Use the `callpeak` command in MACS2 to identify regions enriched with TN5
    insertions. The parameters passed to MACS2 are:
    "-shift -100 -extsize 200 -nomodel -callsummits -nolambda -keep-dup all"

    The results are stored in `.uns[key_added]`.

    Parameters
    ----------
    data
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
    group_by
        Group the cells before peak calling. If a `str`, groups are obtained from
        `.obs[group_by]`.
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
    """
    if isinstance(group_by, str):
        group_by = data.obs[group_by].astype("str")
    out_dir = out_dir if out_dir is None else str(out_dir)
    res = _snapatac2.call_peaks(data, group_by, selections, q_value, out_dir)
    if inplace:
        data.uns[key_added] = res
    else:
        return res