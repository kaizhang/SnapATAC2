from __future__ import annotations
from typing_extensions import Literal

from pathlib import Path
import numpy as np
import anndata as ad

import snapatac2
from snapatac2._snapatac2 import AnnData, AnnDataSet
import snapatac2._snapatac2 as internal

def frip(
    adata: AnnData | list[AnnData],
    regions: dict[str, Path | list[str]],
    *,
    inplace: bool = True,
    n_jobs: int = 8,
) -> dict[str, list[float]] | list[dict[str, list[float]]] | None:
    """ Add fraction of reads in peaks (FRiP) to the AnnData object.

    :func:`~snapatac2.pp.import_data` must be ran first in order to use this function.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
        `adata` could also be a list of AnnData objects.
        In this case, the function will be applied to each AnnData object in parallel.
    regions
        A dictionary containing the peak sets to compute FRiP.
        The keys are peak set names and the values are either a bed file name or a list of
        strings representing genomic regions. For example,
        `{"promoter_frac": "promoter.bed", "enhancer_frac": ["chr1:100-200", "chr2:300-400"]}`.
    inplace
        Whether to add the results to `adata.obs` or return it as a dictionary.
    n_jobs
        Number of jobs to run in parallel when `adata` is a list.
        If `n_jobs=-1`, all CPUs will be used.

    Returns
    -------
    dict[str, list[float]] | list[dict[str, list[float]]] | None
        If `inplace = True`, directly adds the results to `adata.obs`.
        Otherwise return a dictionary containing the results.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> data = snap.read(snap.datasets.pbmc5k(type='h5ad'), backed=None)
    >>> snap.pp.add_frip(data, {"peaks_frac": snap.datasets.cre_HEA()})
    >>> print(data.obs['peaks_frac'].head())
    index
    AAACGAAAGACGTCAG-1    0.708841
    AAACGAAAGATTGACA-1    0.731711
    AAACGAAAGGGTCCCT-1    0.692434
    AAACGAACAATTGTGC-1    0.694849
    AAACGAACACTCGTGG-1    0.687787
    Name: peaks_frac, dtype: float64
    """

    for k in regions.keys():
        if isinstance(regions[k], str) or isinstance(regions[k], Path):
            regions[k] = internal.read_regions(Path(regions[k]))
        elif not isinstance(regions[k], list):
            regions[k] = list(iter(regions[k]))

    if isinstance(adata, list):
        return snapatac2._utils.anndata_par(
            adata,
            lambda x: frip(x, regions, inplace),
            n_jobs=n_jobs,
        )
    else:
        result = internal.add_frip(adata, regions)
        if inplace:
            for k, v in result.items():
                adata.obs[k] = v
        else:
            return result


def frag_size_distr(
    adata: AnnData | list[AnnData],
    *,
    max_recorded_size: int = 1000,
    add_key: str = "frag_size_distr",
    inplace: bool = True,
    n_jobs: int = 8,
) -> np.ndarray | list[np.ndarray] | None:
    """ Compute the fragment size distribution of the dataset. 

    This function computes the fragment size distribution of the dataset.
    Note that it does not operate at the single-cell level.
    The result is stored in a vector where each element represents the number of fragments
    and the index represents the fragment length. The first posision of the vector is
    reserved for fragments with size larger than the `max_recorded_size` parameter.
    :func:`~snapatac2.pp.import_data` must be ran first in order to use this function.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
        `adata` could also be a list of AnnData objects.
        In this case, the function will be applied to each AnnData object in parallel.
    max_recorded_size
        The maximum fragment size to record in the result.
        Fragments with length larger than `max_recorded_size` will be recorded in the first
        position of the result vector.
    add_key
        Key used to store the result in `adata.uns`.
    inplace
        Whether to add the results to `adata.uns` or return it.
    n_jobs
        Number of jobs to run in parallel when `adata` is a list.
        If `n_jobs=-1`, all CPUs will be used.

    Returns
    -------
    np.ndarray | list[np.ndarray] | None
        If `inplace = True`, directly adds the results to `adata.uns['`add_key`']`.
        Otherwise return the results.
    """
    if isinstance(adata, list):
        return snapatac2._utils.anndata_par(
            adata,
            lambda x: frag_size_distr(x, add_key=add_key, max_recorded_size=max_recorded_size, inplace=inplace),
            n_jobs=n_jobs,
        )
    else:
        result = internal.fragment_size_distribution(adata, max_recorded_size)
        if inplace:
            adata.uns[add_key] = result
        else:
            return result