from __future__ import annotations

from pathlib import Path
from snapatac2._snapatac2 import AnnData, AnnDataSet
import snapatac2._snapatac2 as _snapatac2
import logging

def call_peaks(
    adata: AnnData | AnnDataSet,
    groupby: str | list[str],
    selections: set[str] | None = None,
    q_value: float = 0.05,
    nolambda: bool = False,
    shift: int = -100,
    extsize: int = 200,
    blacklist: Path | None = None,
    out_dir: Path | None = None,
    key_added: str = 'peaks',
    keep_unmerged: bool = False,
    inplace: bool = True,
    n_jobs: int = 8,
) -> 'polars.DataFrame' | None:
    """
    Call peaks using MACS2.

    Use the `callpeak` command in MACS2 to identify regions enriched with TN5
    insertions. The default parameters passed to MACS2 are:
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
    nolambda
        Whether to use the `--nolambda` option in MACS2.
    shift
        The shift size in MACS2.
    extsize
        The extension size in MACS2.
    blacklist
        Path to the blacklist file in BED format. If provided, regions in the blacklist will be
        removed.
    out_dir
        If provided, raw peak files from each group will be saved in the directory.
        Otherwise, they will be stored in a temporary directory which will be removed
        afterwards.
    key_added
        `.uns` key under which to add the peak information.
    keep_unmerged
        Whether to keep the unmerged peaks in `.uns['unmerged_peaks']`.
    inplace
        Whether to store the result inplace.
    n_jobs
        Number of processes to use for peak calling.

    Returns
    -------
    'polars.DataFrame' | None
        If `inplace=True` it stores the result in `adata.uns[`key_added`]`.
        Otherwise, it returns the result as a dataframe.
    """
    from MACS3.Signal.PeakDetect import PeakDetect
    from collections import namedtuple
    from math import log
    from multiprocess import Pool
    from tqdm import tqdm

    if isinstance(groupby, str):
        groupby = list(adata.obs[groupby])
    out_dir = out_dir if out_dir is None else str(out_dir)
    fw_tracks = _snapatac2.create_fwtrack_obj(adata, groupby, selections)

    chroms = adata.uns['reference_sequences']['reference_seq_name']
    sizes = adata.uns['reference_sequences']['reference_seq_length']

    options = type('MACS3_OPT', (), {})()
    options.info = lambda _: None
    options.debug = lambda _: None
    options.warn = logging.warn
    options.log_qvalue = log(q_value, 10) * -1
    options.log_pvalue = None
    options.PE_MODE = False
    options.maxgap = 30
    options.minlen = 50
    options.shift = shift
    options.gsize = sizes.sum()
    options.nolambda = nolambda
    options.smalllocal = 1000
    options.largelocal = 10000
    options.store_bdg = False
    options.name = "MACS3"
    options.bdg_treat = 't'
    options.bdg_control = 'c'
    options.do_SPMR = False
    options.cutoff_analysis = False
    options.cutoff_analysis_file = 'a'
    options.trackline = False
    options.call_summits = True
    options.broad = False
    options.fecutoff = 1.0

    def _call_peaks(fwt):
        peakdetect = PeakDetect(treat=fwt, opt=options, d=1.0)
        peakdetect.call_peaks()
        peakdetect.peaks.filter_fc(fc_low = options.fecutoff)
        return peakdetect.peaks
    with Pool(n_jobs) as p:
        result = list(tqdm(p.imap(_call_peaks, list(fw_tracks.values())), total=len(fw_tracks)))
        fw_tracks = {k: v for k, v in zip(fw_tracks.keys(), result)}

    chrom_sizes = {k: v for k, v in zip(chroms, sizes)}
    (merged, unmerged) = _snapatac2.py_merge_peaks(fw_tracks, chrom_sizes, blacklist)

    if inplace:
        if keep_unmerged:
            adata.uns['unmerged_peaks'] = unmerged
        if adata.isbacked:
            adata.uns[key_added] = merged
        else:
            adata.uns[key_added] = merged.to_pandas()
    else:
        if keep_unmerged:
            return (merged, unmerged)
        else:
            return merged

def call_peaks2(
    adata: AnnData | AnnDataSet,
    groupby: str | list[str],
    selections: set[str] | None = None,
    q_value: float = 0.05,
    nolambda: bool = True,
    shift: int = -100,
    extsize: int = 200,
    blacklist: Path | None = None,
    out_dir: Path | None = None,
    key_added: str = 'peaks',
    inplace: bool = True,
) -> 'polars.DataFrame' | None:
    """
    Call peaks using MACS2.

    Use the `callpeak` command in MACS2 to identify regions enriched with TN5
    insertions. The default parameters passed to MACS2 are:
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
    nolambda
        Whether to use the `--nolambda` option in MACS2.
    shift
        The shift size in MACS2.
    extsize
        The extension size in MACS2.
    blacklist
        Path to the blacklist file in BED format. If provided, regions in the blacklist will be
        removed.
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
    res = _snapatac2.call_peaks(adata, groupby, q_value, nolambda,
        shift, extsize, selections, blacklist, out_dir)
    if inplace:
        if adata.isbacked:
            adata.uns[key_added] = res
        else:
            adata.uns[key_added] = res.to_pandas()
    else:
        return res