from __future__ import annotations

from pathlib import Path
import numpy as np

import snapatac2
import snapatac2._snapatac2 as internal
from snapatac2.genome import Genome

def tsse(
    adata: internal.AnnData | list[internal.AnnData],
    gene_anno: Genome | Path,
    *,
    exclude_chroms: list[str] | str | None = ["chrM", "M"],
    inplace: bool = True,
    n_jobs: int = 8,
) -> np.ndarray | list[np.ndarray] | None:
    """ Compute the TSS enrichment score (TSSe) for each cell.

    :func:`~snapatac2.pp.import_data` must be ran first in order to use this function.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
        `adata` could also be a list of AnnData objects.
        In this case, the function will be applied to each AnnData object in parallel.
    gene_anno
        A :class:`~snapatac2.Genome` object or a GTF/GFF file containing the gene annotation.
    exclude_chroms
        A list of chromosomes to exclude.
    inplace
        Whether to add the results to `adata.obs` or return it as a dictionary.
    n_jobs
        Number of jobs to run in parallel when `adata` is a list.
        If `n_jobs=-1`, all CPUs will be used.

    Returns
    -------
    tuple[np.ndarray, tuple[float, float]] | list[tuple[np.ndarray, tuple[float, float]]] | None
        If `inplace = True`, cell-level TSSe scores are computed and stored in `adata.obs['tsse']`.
        Library-level TSSe scores are stored in `adata.uns['library_tsse']`.
        Fraction of fragments overlapping TSS are stored in `adata.uns['fraction_overlap_TSS']`.
        If `inplace = False`, return a tuple containing all these values.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> data = snap.pp.import_data(snap.datasets.pbmc500(downsample=True), chrom_sizes=snap.genome.hg38, sorted_by_barcode=False)
    >>> snap.metrics.tsse(data, snap.genome.hg38)
    >>> print(data.obs['tsse'].head())
    AAACTGCAGACTCGGA-1    32.129514
    AAAGATGCACCTATTT-1    22.052786
    AAAGATGCAGATACAA-1    27.109808
    AAAGGGCTCGCTCTAC-1    24.990329
    AAATGAGAGTCCCGCA-1    33.264463
    Name: tsse, dtype: float64
    """
    gene_anno = gene_anno.annotation if isinstance(gene_anno, Genome) else gene_anno
 
    if isinstance(adata, list):
        result = snapatac2._utils.anndata_par(
            adata,
            lambda x: tsse(x, gene_anno, exclude_chroms=exclude_chroms, inplace=inplace),
            n_jobs=n_jobs,
        )
    else:
        result = internal.tss_enrichment(adata, gene_anno, exclude_chroms)
        result['tsse'] = np.array(result['tsse'])
        result['TSS_profile'] = np.array(result['TSS_profile'])
        if inplace:
            adata.obs["tsse"] = result['tsse']
            adata.uns['library_tsse'] = result['library_tsse']
            adata.uns['fraction_overlap_TSS'] = result['fraction_overlap_TSS']
            adata.uns['TSS_profile'] = result['TSS_profile']
    if inplace:
        return None
    else:
        return result

def frip(
    adata: internal.AnnData | list[internal.AnnData],
    regions: dict[str, Path | list[str]],
    *,
    normalized: bool = True,
    count_as_insertion: bool = False,
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
    normalized
        Whether to normalize the counts by the total number of fragments.
        If False, the raw number of fragments in peaks will be returned.
    count_as_insertion
        Whether to count transposition events instead of fragments. Transposition
        events are located at both ends of fragments.
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
    >>> data = snap.pp.import_data(snap.datasets.pbmc500(downsample=True), chrom_sizes=snap.genome.hg38, sorted_by_barcode=False)
    >>> snap.metrics.frip(data, {"peaks_frac": snap.datasets.cre_HEA()})
    >>> print(data.obs['peaks_frac'].head())
    AAACTGCAGACTCGGA-1    0.715930
    AAAGATGCACCTATTT-1    0.697364
    AAAGATGCAGATACAA-1    0.713615
    AAAGGGCTCGCTCTAC-1    0.678428
    AAATGAGAGTCCCGCA-1    0.724910
    Name: peaks_frac, dtype: float64
    """

    for k in regions.keys():
        if isinstance(regions[k], str) or isinstance(regions[k], Path):
            regions[k] = internal.read_regions(Path(regions[k]))
        elif not isinstance(regions[k], list):
            regions[k] = list(iter(regions[k]))

    if isinstance(adata, list):
        result = snapatac2._utils.anndata_par(
            adata,
            lambda x: frip(x, regions, inplace=inplace),
            n_jobs=n_jobs,
        )
    else:
        result = internal.add_frip(adata, regions, normalized, count_as_insertion)
        if inplace:
            for k, v in result.items():
                adata.obs[k] = v
    if inplace:
        return None
    else:
        return result

def frag_size_distr(
    adata: internal.AnnData | list[internal.AnnData],
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
        result = np.array(internal.fragment_size_distribution(adata, max_recorded_size))
        if inplace:
            adata.uns[add_key] = result
        else:
            return result