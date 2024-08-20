from pathlib import Path
import numpy as np

import snapatac2

def filter_kwargs(func, kwargs_dict):
    import inspect
    signature = inspect.signature(func)
    accepted_params = set(signature.parameters.keys())
    filtered_kwargs = {k: v for k, v in kwargs_dict.items() if k in accepted_params}
    return filtered_kwargs

def recipe_10x_metrics(
    bam_file: Path,
    output_fragment_file: Path,
    output_h5ad_file: Path,
    peaks: Path | list[str] | None = None,
    **kwargs,
) -> dict:
    """
    Preprocess raw Bam files and generate a set of QC metrics similar to the
    10x Genomics pipeline.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> metrics = snap.pp.recipe_10x_metrics(snap.datasets.pbmc500(type='bam'), 'fragments.tsv.gz', 'data.h5ad', barcode_tag='CB', source='10x', chrom_sizes=snap.genome.hg38, gene_anno=snap.genome.hg38)
    >>> print(metrics)
    """
    qc = {
        "Sequencing": {},
        "Cells": {},
        "Library Complexity": {},
        "Mapping": {},
        "Targeting": {},
    }

    bam_qc = snapatac2.pp.make_fragment_file(
        bam_file,
        output_fragment_file,
        **filter_kwargs(snapatac2.pp.make_fragment_file, kwargs)
    )
    qc["Sequencing"]["Sequenced_reads"] = bam_qc["Sequenced_reads"]
    qc["Sequencing"]["Sequenced_read_pairs"] = bam_qc["Sequenced_read_pairs"]
    qc["Sequencing"]["Fraction_valid_barcode"] = bam_qc["Fraction_valid_barcode"]
    qc["Mapping"]["Fraction_confidently_mapped"] = bam_qc["Fraction_confidently_mapped"]
    qc["Mapping"]["Fraction_unmapped"] = bam_qc["Fraction_unmapped"]
    qc["Mapping"]["Fraction_nonnuclear"] = bam_qc["Fraction_nonnuclear"]
    qc["Mapping"]["Fraction_fragment_in_nucleosome_free_region"] = bam_qc["Fraction_fragment_in_nucleosome_free_region"]
    qc["Mapping"]["Fraction_fragment_flanking_single_nucleosome"] = bam_qc["Fraction_fragment_flanking_single_nucleosome"]
    qc["Library Complexity"]["Fraction_duplicates"] = bam_qc["Fraction_duplicates"]

    adata = snapatac2.pp.import_data(
        output_fragment_file,
        min_num_fragments=0,
        file=output_h5ad_file,
        **filter_kwargs(snapatac2.pp.import_data, kwargs),
    )
    snapatac2.metrics.tsse(adata, **filter_kwargs(snapatac2.metrics.tsse, kwargs))
    qc["Targeting"]["TSS_enrichment_score"] = adata.uns['library_tsse']
    qc["Targeting"]["Fraction_of_high-quality_fragments_overlapping_TSS"] = adata.uns['fraction_overlap_TSS']

    if peaks is None:
        snapatac2.tl.macs3(adata, qvalue=0.001)
        peaks = [f"{row[0]}:{row[1]}-{row[2]}" for row in adata.uns['macs3_pseudobulk'].iter_rows()]
    else:
        if not isinstance(peaks, list):
            p = []
            with open(peaks, 'r') as f:
                for line in f:
                    if not line.startswith('#'):
                        items = line.strip().split()
                        p.append(f'{items[0]}:{items[1]}-{items[2]}')
            peaks = p
    qc["Targeting"]["Number_of_peaks"] = len(peaks)

    snapatac2.metrics.frip(adata, {"n_frag_overlap_peak": peaks}, normalized=False)
    qc["Targeting"]["Fraction_of_high-quality_fragments_overlapping_peaks"] = adata.obs['n_frag_overlap_peak'].sum() / adata.obs['n_fragment'].sum()

    cell_idx = snapatac2.pp.call_cells(adata, use_rep="n_frag_overlap_peak", inplace=False)
    n_cells = len(cell_idx)
    n_fragment = adata.obs['n_fragment'].to_numpy()
    qc["Cells"]["Number_of_cells"] = n_cells
    is_paired = bam_qc["Sequenced_read_pairs"] > 0
    if is_paired:
        qc["Cells"]["Mean_raw_read_pairs_per_cell"] = bam_qc["Sequenced_read_pairs"] / n_cells
    else:
        qc["Cells"]["Mean_raw_read_pairs_per_cell"] = bam_qc["Sequenced_reads"] / n_cells
    qc["Cells"]["Median_high-quality_fragments_per_cell"] = np.median(n_fragment[cell_idx])
    qc["Cells"]["Fraction of high-quality fragments in cells"] = n_fragment[cell_idx].sum() / n_fragment.sum()

    adata.subset(cell_idx)
    frip = snapatac2.metrics.frip(adata, {"overlap_peak": peaks}, normalized=False, count_as_insertion=True, inplace=False)
    n_fragment = adata.obs['n_fragment'].to_numpy()
    if is_paired:
        qc["Cells"]["Fraction_of_transposition_events_in_peaks_in_cells"] = np.sum(frip['overlap_peak']) / (n_fragment.sum() * 2)
    else:
        qc["Cells"]["Fraction_of_transposition_events_in_peaks_in_cells"] = np.sum(frip['overlap_peak']) / n_fragment.sum()

    adata.close()
    return qc