import numpy as np
from typing import Optional, Sequence, Union, Set

from snapatac2._snapatac2 import AnnData, AnnDataSet
import snapatac2._snapatac2 as internal

def export_bed(
    adata: Union[AnnData, AnnDataSet],
    group_by: Union[str, Sequence[str]],
    selections: Optional[Set[str]] = None,
    ids: Optional[Union[str, np.ndarray]] = None,
    out_dir: str = "./",
):
    if isinstance(group_by, str):
        group_by = adata.obs[group_by].astype("str")
    
    if ids is None:
        ids = adata.obs[:, 0].astype("str")
    elif isinstance(ids, str):
        ids = adata.obs[ids].astype("str")

    return internal.export_bed(adata, list(ids), list(group_by), selections, out_dir)