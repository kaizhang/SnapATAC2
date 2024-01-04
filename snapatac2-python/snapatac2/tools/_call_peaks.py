from __future__ import annotations

from pathlib import Path
from snapatac2._snapatac2 import AnnData, AnnDataSet
import snapatac2._snapatac2 as _snapatac2
import logging
from snapatac2.genome import Genome

def macs3(
    adata: AnnData | AnnDataSet,
    groupby: str | list[str],
    *,
    qvalue: float = 0.05,
    replicate: str | list[str] | None = None,
    replicate_qvalue: float | None = None,
    max_frag_size: int | None = None,
    selections: set[str] | None = None,
    nolambda: bool = False,
    shift: int = -100,
    extsize: int = 200,
    blacklist: Path | None = None,
    key_added: str = 'macs3',
    tempdir: Path | None = None,
    inplace: bool = True,
    n_jobs: int = 8,
) -> dict[str, 'polars.DataFrame'] | None:
    """ Call peaks using MACS3.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
    groupby
        Group the cells before peak calling. If a `str`, groups are obtained from
        `.obs[groupby]`.
    qvalue
        qvalue cutoff used in MACS3.
    replicate
        Replicate information. If provided, reproducible peaks will be called
        for each group.
    replicate_qvalue
        qvalue cutoff used in MACS3 for calling peaks in replicates.
        This parameter is only used when `replicate` is provided.
        Typically this parameter is used to call peaks in replicates with a more lenient cutoff.
        If not provided, `qvalue` will be used.
    max_frag_size
        Maximum fragment size. If provided, fragments with sizes larger than
        `max_frag_size` will be not be used in peak calling.
        This is used in ATAC-seq data to remove fragments that are not 
        from nucleosome-free regions.
        You can use :func:`~snapatac2.pl.frag_size_distr` to choose a proper value for
        this parameter.
    selections
        Call peaks for the selected groups only.
    nolambda
        Whether to use the `--nolambda` option in MACS.
    shift
        The shift size in MACS.
    extsize
        The extension size in MACS.
    blacklist
        Path to the blacklist file in BED format. If provided, regions in the blacklist will be
        removed.
    key_added
        `.uns` key under which to add the peak information.
    tempdir
        If provided, a temporary directory will be created in the directory.
        Otherwise, a temporary directory will be created in the system default temporary directory.
    inplace
        Whether to store the result inplace.
    n_jobs
        Number of processes to use for peak calling.

    Returns
    -------
    dict[str, 'polars.DataFrame'] | None
        If `inplace=True` it stores the result in `adata.uns[`key_added`]`.
        Otherwise, it returns the result as dataframes.

    See Also
    --------
    merge_peaks
    """
    from MACS3.Signal.PeakDetect import PeakDetect
    from math import log
    from multiprocess import get_context
    from tqdm import tqdm
    import tempfile

    if isinstance(groupby, str):
        groupby = list(adata.obs[groupby])
    if replicate is not None and isinstance(replicate, str):
        replicate = list(adata.obs[replicate])
    if tempdir is None:
        tempdir = Path(tempfile.mkdtemp())
    else:
        tempdir = Path(tempfile.mkdtemp(dir=tempdir))

    logging.info("Exporting fragments...")
    fragments = _snapatac2.export_tags(adata, tempdir, groupby, replicate, max_frag_size, selections)
    
    options = type('MACS3_OPT', (), {})()
    options.info = lambda _: None
    options.debug = lambda _: None
    options.warn = logging.warn
    options.log_pvalue = None
    options.PE_MODE = False
    options.maxgap = 30
    options.minlen = 50
    options.shift = shift
    options.gsize = adata.uns['reference_sequences']['reference_seq_length'].sum()
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
    options.d = extsize
    options.scanwindow = 2 * options.d

    def _call_peaks(tags):
        merged, reps = _snapatac2.create_fwtrack_obj(tags)
        options.log_qvalue = log(qvalue, 10) * -1
        logging.getLogger().setLevel(logging.CRITICAL + 1)
        peakdetect = PeakDetect(treat=merged, opt=options)
        peakdetect.call_peaks()
        peakdetect.peaks.filter_fc(fc_low = options.fecutoff)
        merged = peakdetect.peaks

        others = []
        if replicate_qvalue is not None:
            options.log_qvalue = log(replicate_qvalue, 10) * -1
        for x in reps:
            peakdetect = PeakDetect(treat=x, opt=options)
            peakdetect.call_peaks()
            peakdetect.peaks.filter_fc(fc_low = options.fecutoff)
            others.append(peakdetect.peaks)
        
        logging.getLogger().setLevel(logging.INFO)
        return _snapatac2.find_reproducible_peaks(merged, others, blacklist)

    logging.info("Calling peaks...")
    peaks = _par_map(_call_peaks, [(x,) for x in fragments.values()], n_jobs)
    peaks = {k: v for k, v in zip(fragments.keys(), peaks)}
    if inplace:
        if adata.isbacked:
            adata.uns[key_added] = peaks
        else:
            adata.uns[key_added] = {k: v.to_pandas() for k, v in peaks.items()}
    else:
        return peaks

def merge_peaks(
    peaks: dict[str, 'polars.DataFrame'],
    chrom_sizes: dict[str, int] | Genome,
    half_width: int = 250,
) -> 'polars.DataFrame':
    """Merge peaks from different groups.

    Merge peaks from different groups. It is typically used to merge
    results from :func:`~snapatac2.tools.macs3`.

    This function initially expands the summits of identified peaks by `half_width`
    on both sides. Following this expansion, it addresses the issue of overlapping
    peaks through an iterative process. The procedure begins by prioritizing the
    most significant peak, determined by the smallest p-value. This peak is retained,
    and any peak that overlaps with it is excluded. Subsequently, the same method
    is applied to the next most significant peak. This iteration continues until
    all peaks have been evaluated, resulting in a final list of non-overlapping
    peaks, each with a fixed width determined by the initial extension.

    Parameters
    ----------
    peaks
        Peak information from different groups.
    chrom_sizes
        Chromosome sizes. If a :class:`~snapatac2.genome.Genome` is provided,
        chromosome sizes will be obtained from the genome.
    half_width
        Half width of the merged peaks.

    Returns
    -------
    'polars.DataFrame'
        A dataframe with merged peaks.

    See Also
    --------
    macs3
    """
    chrom_sizes = chrom_sizes.chrom_sizes if isinstance(chrom_sizes, Genome) else chrom_sizes
    return _snapatac2.py_merge_peaks(peaks, chrom_sizes, half_width)

def _par_map(mapper, args, nprocs):
    import time
    from multiprocess import get_context
    from tqdm import tqdm

    with get_context("spawn").Pool(nprocs) as pool:
        procs = set(pool._pool)
        jobs = [(i, pool.apply_async(mapper, x)) for i, x in enumerate(args)]
        results = []
        with tqdm(total=len(jobs)) as pbar:
            while len(jobs) > 0:
                if any(map(lambda p: not p.is_alive(), procs)):
                    raise RuntimeError("Some worker process has died unexpectedly.")

                remaining = []
                for i, job in jobs:
                    if job.ready():
                        results.append((i, job.get()))
                        pbar.update(1)
                    else:
                        remaining.append((i, job))
                jobs = remaining
                time.sleep(0.5)
        return [x for _,x in sorted(results, key=lambda x: x[0])]