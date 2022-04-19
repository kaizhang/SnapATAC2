from pathlib import Path
from typing import Optional, Union, Sequence, Set
from snapatac2._snapatac2 import AnnData, AnnDataSet
import snapatac2._snapatac2 as _snapatac2

def call_peaks(
    data: Union[AnnData, AnnDataSet],
    group_by: Union[str, Sequence[str]],
    selections: Optional[Set[str]] = None,
    q_value: float = 0.05,
    n_jobs: int = 8,
):
    if isinstance(group_by, str):
        group_by = data.obs[group_by].astype("str")
    _snapatac2.call_peaks(data, group_by, selections, q_value)