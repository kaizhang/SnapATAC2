from __future__ import annotations
from typing_extensions import Literal

import logging
from typing import Callable
from pathlib import Path
import numpy as np
import retworkx
import scipy.sparse as sp

from snapatac2.genome import Genome
from snapatac2._utils import fetch_seq
from snapatac2._snapatac2 import (
    AnnData, AnnDataSet, link_region_to_gene, NodeData, LinkData, PyDNAMotif,
)

def init_network_from_annotation(
    regions: list[str],
    anno_file: Path | Genome,
    upstream: int = 250000,
    downstream: int = 250000,
    id_type: Literal["gene_name", "gene_id", "transcript_id"] = "gene_name",
    coding_gene_only: bool = True,
) -> retworkx.PyDiGraph:
    """
    Build CRE-gene network from gene annotations.

    Link CREs to genes if they are close to genes' promoter regions.

    Parameters
    ----------
    regions:
        A list of peaks/regions, e.g., `["chr1:100-1000", "chr2:55-222"]`.
    anno_file
        The GFF file containing the transcript level annotations.
    upstream
        Upstream extension to the transcription start site.
    downstream
        Downstream extension to the transcription start site.
    id_type
        "gene_name", "gene_id" or "transcript_id".
    coding_gene_only
        Retain only coding genes in the network.

    Returns
    -------
    A network where peaks/regions point towards genes if they are within genes'
    regulatory domains.
    """
    if isinstance(anno_file, Genome):
        anno_file = anno_file.fetch_annotations()
        
    region_added = {}
    graph = retworkx.PyDiGraph()
    links = link_region_to_gene(
        regions,
        str(anno_file),
        upstream,
        downstream,
        id_type,
        coding_gene_only,
    )
    for gene, regions in links.items():
        to = graph.add_node(gene)
        for region, data in regions:
            if region in region_added:
                graph.add_edge(region_added[region], to, data)
            else:
                region_added[region] = graph.add_parent(to, region, data)
    return graph

def add_cor_scores(
    network: retworkx.PyDiGraph,
    *,
    gene_mat: AnnData | AnnDataSet,
    peak_mat: AnnData | AnnDataSet | None = None,
    select: list[str] | None = None,
):
    """
    Compute correlation scores between genes and CREs.

    Parameters
    ----------
    network
        network
    gene_mat
        AnnData or AnnDataSet object storing the cell by gene count matrix,
        where the `.var_names` contains genes.
    peak_mat
        AnnData or AnnDataSet object storing the cell by peak count matrix,
        where the `.var_names` contains peaks.
    select
        Run this for selected genes only.
    """
    from tqdm import tqdm
    from scipy.stats import spearmanr

    if peak_mat is not None and peak_mat.obs_names != gene_mat.obs_names:
        raise NameError("gene matrix and peak matrix should have the same obs_names")
    if select is not None:
        select = set(select)

    gene_set = set(gene_mat.var_names)
    prune_network(
        network, 
        node_filter = lambda x: x.id in gene_set or x.type != "gene",
    )

    if network.num_edges() > 0:
        if peak_mat is None:
            for (nd_X, X), (nd_y, y) in tqdm(_get_data_iter(network, gene_mat, gene_mat, select)):
                if sp.issparse(X):
                    X = X.todense()
                if sp.issparse(y):
                    y = y.todense()
                scores = np.apply_along_axis(lambda x: spearmanr(y, x)[0], 0, X)
                for nd, sc in zip(nd_X, scores):
                    network.get_edge_data(nd, nd_y).gg_cor_score = sc
        else:
            for (nd_X, X), (nd_y, y) in tqdm(_get_data_iter(network, peak_mat, gene_mat, select)):
                if sp.issparse(X):
                    X = X.todense()
                if sp.issparse(y):
                    y = y.todense()
                scores = np.apply_along_axis(lambda x: spearmanr(y, x)[0], 0, X)
                for nd, sc in zip(nd_X, scores):
                    network.get_edge_data(nd, nd_y).cor_score = sc

def add_regr_scores(
    network: retworkx.PyDiGraph,
    peak_mat: AnnData | AnnDataSet,
    gene_mat: AnnData | AnnDataSet,
    select: list[str] | None = None,
    use_gpu: bool = False,
):
    """
    Perform regression analysis between genes and CREs.

    Parameters
    ----------
    network
        network
    peak_mat
        AnnData or AnnDataSet object storing the cell by peak count matrix,
        where the `.var_names` contains peaks.
    gene_mat
        AnnData or AnnDataSet object storing the cell by gene count matrix,
        where the `.var_names` contains genes.
    select
        Run this for selected genes only.
    use_gpu
        Whether to use gpu
    """
    from tqdm import tqdm

    if peak_mat.obs_names != gene_mat.obs_names:
        raise NameError("gene matrix and peak matrix should have the same obs_names")
    if select is not None:
        select = set(select)

    gene_set = set(gene_mat.var_names)
    prune_network(
        network, 
        node_filter = lambda x: x.id in gene_set or x.type == "region",
    )
    if network.num_edges() == 0:
        return network

    tree_method = "gpu_hist" if use_gpu else "hist"

    for (nd_X, X), (nd_y, y) in tqdm(_get_data_iter(network, peak_mat, gene_mat, select)):
        y = y.todense() if sp.issparse(y) else y
        scores = gbTree(X, y, tree_method=tree_method)
        for nd, sc in zip(nd_X, scores):
            network.get_edge_data(nd, nd_y).regr_score = sc

def add_tf_binding(
    network: retworkx.PyDiGraph,
    motifs: list[PyDNAMotif],
    genome_fasta: Path | Genome,
):
    """
    Add TF motif binding information.

    Parameters
    ----------
    network
        Network
    motifs
        TF motifs
    genome_fasta
        A fasta file containing the genome sequences or a Genome object.
    """
    from pyfaidx import Fasta
    from tqdm import tqdm
    import itertools

    regions = [(i, network[i].id) for i in network.node_indices() if network[i].type == "region"]
    logging.info("Fetching {} sequences ...".format(len(regions)))
    genome = genome_fasta.fetch_fasta() if isinstance(genome_fasta, Genome) else str(genome_fasta)
    genome = Fasta(genome, one_based_attributes=False)
    sequences = [fetch_seq(genome, region) for _, region in regions]

    logging.info("Finding motif sites ...")
    for motif in tqdm(motifs):
        bound = motif.with_nucl_prob().exists(sequences)
        if any(bound):
            nid = network.add_node(NodeData(motif.name, "motif"))
            network.add_edges_from(
                [(nid, i, LinkData()) for i, _ in itertools.compress(regions, bound)]
            )

def to_gene_network(
    network: retworkx.PyDiGraph,
    motif_name_modifier = None,
) -> list[str]:
    """
    Make the network contains genes only.

    Parameters
    ----------

    Returns 
    -------
    """
    genes = dict((network[nd].id, nd) for nd in network.node_indices() if network[nd].type == "gene")
    not_found = []
    for nd in network.node_indices():
        if network[nd].type == "motif":
            name = network[nd].id 
            if motif_name_modifier is not None:
                name = motif_name_modifier(name)
            if name in genes:
                fr = genes[name]
                for region in network.successor_indices(nd):
                    for gene in network.successor_indices(region):
                        link = network.get_edge_data(region, gene)
                        network.add_edge(
                            fr, gene,
                            LinkData(regr_score=link.regr_score, cor_score=link.cor_score),
                        )
            else:
                not_found.append(name)
    
    # Clean up.
    # Remove edges first, for performance reason.
    for nd in network.node_indices():
        if network[nd].type != "gene":
            for fr, to, _ in network.out_edges(nd):
                network.remove_edge(fr, to)
    # Remove nodes
    network.remove_nodes_from(
        [nd for nd in network.node_indices() if network[nd].type != "gene"]
    )
    return not_found

def prune_network(
    network: retworkx.PyDiGraph,
    node_filter: Callable[[NodeData], bool] | None = None,
    edge_filter: Callable[[LinkData], bool] | None = None,
    remove_isolates: bool = False,
):
    """
    Prune the network.

    Parameters
    ----------
    network
        network
    node_filter
        Node filter function.
    edge_filter
        Edge filter function.
    """
    if edge_filter is not None:
        for eid in network.edge_indices():
            if not edge_filter(network.get_edge_data_by_index(eid)):
                network.remove_edge_from_index(eid)

    if node_filter is not None:
        for nid in network.node_indices():
            if not node_filter(network.get_node_data(nid)):
                network.remove_node(nid)

    if remove_isolates:
        for nid in network.node_indices():
            if network.in_degree(nid) + network.out_degree(nid) == 0:
                network.remove_node(nid)

class RegionGenePairIter:
    def __init__(
        self,
        network,
        peak_mat,
        gene_mat,
        region_idx_map,
        gene_ids
    ) -> None:
        self.network = network
        self.gene_ids = gene_ids
        self.peak_mat = peak_mat
        self.gene_mat = gene_mat
        self.region_idx_map = region_idx_map
        self.index = 0

    def __len__(self):
        return self.gene_mat.shape[1]

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.__len__():
            raise StopIteration

        nd_y = self.gene_ids[self.index]
        nd_X = self.network.predecessor_indices(nd_y)
        y = self.gene_mat[:, self.index]
        X = self.peak_mat[:, [self.region_idx_map[nd] for nd in nd_X]]

        self.index += 1
        return (nd_X, X), (nd_y, y)

def _get_data_iter(
    network: retworkx.PyDiGraph,
    peak_mat: AnnData | AnnDataSet,
    gene_mat: AnnData | AnnDataSet,
    select: set[str] | None = None,
) -> RegionGenePairIter:
    """
    """
    genes = []
    regions = set()
    for nd in network.node_indices():
        if select is None or network[nd].id in select:
            parents = network.predecessor_indices(nd)
            if len(parents) > 0:
                genes.append(nd)
                for x in parents:
                    regions.add(x)
    regions = list(regions)

    gene_idx = gene_mat.var_ix([network[x].id for x in genes])
    gene_mat = gene_mat.X[:, gene_idx]
    region_idx = peak_mat.var_ix([network[x].id for x in regions])
    peak_mat = peak_mat.X[:, region_idx]
    region_idx_map = dict(zip(regions, range(len(regions))))

    return RegionGenePairIter(
        network,
        peak_mat,
        gene_mat,
        region_idx_map,
        genes,
    )

def elastic_net(X, y):
    from sklearn.linear_model import ElasticNet
    regr = ElasticNet(random_state=0).fit(np.asarray(X.todense()), np.asarray(y.todense()))
    return regr.coef_

def gbTree(X, y, tree_method = "hist"):
    import xgboost as xgb
    model = xgb.XGBRegressor(tree_method = tree_method)
    return model.fit(X, y).feature_importances_