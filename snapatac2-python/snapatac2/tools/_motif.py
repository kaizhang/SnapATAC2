from __future__ import annotations
from typing_extensions import Literal

from pathlib import Path
import logging

from snapatac2._snapatac2 import PyDNAMotif
from snapatac2._utils import fetch_seq
from snapatac2.genome import Genome
from snapatac2.tools._diff import _p_adjust_bh

def motif_enrichment(
    motifs: list[PyDNAMotif],
    regions: dict[str, list[str]],
    genome_fasta: Path | Genome,
    background: list[str] | None = None,
    method: Literal['binomial', 'hypergeometric'] | None = None,
) -> dict[str, 'polars.DataFrame']:
    """
    Identify enriched transcription factor motifs.

    Parameters
    ----------
    motifs
        A list of transcription factor motifs.
    regions
        Groups of regions. Each group will be tested independently against the background.
    genome_fasta
        A fasta file containing the genome sequences or a Genome object.
    background
        A list of regions to be used as the background. If None, the union of elements
        in `regions` will be used as the background.
    method
        Statistical testing method: "binomial" or "hypergeometric".
        To use "hypergeometric", the testing regions must be a subset of
        background regions.

    Returns
    -------
    dict[str, pl.DataFrame]:
        Dataframes containing the enrichment analysis results for different groups.
    """
    from pyfaidx import Fasta
    from tqdm import tqdm
    from scipy.stats import binom, hypergeom
    from math import log2
    import polars as pl

    def count_occurrence(query, idx_map, bound):
        return sum(bound[idx_map[q]] for q in query)

    if method is None:
        method = "hypergeometric" if background is None else "binomial"

    all_regions = set(p for ps in regions.values() for p in ps)
    if background is not None:
        for p in background:
            all_regions.add(p)
    all_regions = list(all_regions)
    region_to_idx = dict(map(lambda x: (x[1], x[0]), enumerate(all_regions)))

    logging.info("Fetching {} sequences ...".format(len(all_regions)))
    genome = genome_fasta.fetch_fasta() if isinstance(genome_fasta, Genome) else str(genome_fasta)
    genome = Fasta(genome, one_based_attributes=False)
    sequences = [fetch_seq(genome, region) for region in all_regions]

    motif_name = []
    group_name = []
    fold_change = []
    n_fg = []
    N_fg = []
    n_bg = []
    N_bg = []
    logging.info("Computing enrichment ...")
    for motif in tqdm(motifs):
        bound = motif.with_nucl_prob().exists(sequences)
        if background is None:
            total_bg = len(bound)
            bound_bg = sum(bound)
        else:
            total_bg = len(background)
            bound_bg = count_occurrence(background, region_to_idx, bound)
        
        for key, val in regions.items():
            total_fg = len(val)
            bound_fg = count_occurrence(val, region_to_idx, bound)

            if bound_fg == 0:
                log_fc = 0 if bound_bg == 0 else float('-inf')
            else:
                log_fc = log2((bound_fg / total_fg) / (bound_bg / total_bg))

            motif_name.append(motif.id)
            group_name.append(key)
            fold_change.append(log_fc)
            n_fg.append(bound_fg)
            N_fg.append(total_fg)
            n_bg.append(bound_bg)
            N_bg.append(total_bg)
          
    if method == "bionomial":
        pval = binom.cdf(n_fg, N_fg, np.array(n_bg) / np.array(N_bg))
    elif method == "hypergeometric":
        pval = hypergeom.cdf(n_fg, N_bg, n_bg, N_fg)
    else:
        raise NameError("'method' needs to be 'binomial' or 'hypergeometric'")

    result = dict(
        (key, {'motif name': [], 'log2(fold change)': [], 'p-value': []}) for key in regions.keys()
    )
    for i, key in enumerate(group_name):
        log_fc = fold_change[i]
        p = (1 - pval[i]) if log_fc >= 0 else pval[i]
        result[key]['motif name'].append(motif_name[i])
        result[key]['log2(fold change)'].append(log_fc)
        result[key]['p-value'].append(float(p))

    for key in result.keys():
        result[key]['adjusted p-value'] = _p_adjust_bh(result[key]['p-value'])
        result[key] = pl.DataFrame(result[key])
    return result
