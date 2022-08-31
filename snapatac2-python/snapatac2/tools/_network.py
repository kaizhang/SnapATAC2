from __future__ import annotations

from pathlib import Path
from pickle import UnpicklingError
from xml.etree.ElementTree import register_namespace
import scipy as sp
import numpy as np
import retworkx

from snapatac2._snapatac2 import AnnData, AnnDataSet, link_region_to_gene

def aggregate_cells():
    """
    Aggregate cells.
    """
    pass

def init_network_from_annotation(
    regions: list[str],
    anno_file: Path,
    upstream: int = 250000,
    downstream: int = 250000,
    id_type: str = "gene_name",
    coding_gene_only: bool = True,
) -> retworkx.PyDiGraph:
    """
    Build CRE-gene network from gene annotations.

    Link CREs to genes if they are within genes' promoter regions.

    id_type
        "gene_name", "gene_id" or "transcript_id".
    """
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
        for region in regions:
            if region in region_added:
                graph.add_edge(region_added[region], to, None)
            else:
                region_added[region] = graph.add_parent(to, region, None)
    return graph

def link_peak_to_gene(
    network: retworkx.PyDiGraph,
    peak_mat: AnnData | AnnDataSet,
    gene_mat: AnnData | AnnDataSet,
):
    """
    Link peaks to target genes.

    Parameters
    ----------
    peak_mat
        AnnData or AnnDataSet object storing the cell by peak count matrix,
        where the `.var_names` contains peaks.
    gene_mat
        AnnData or AnnDataSet object storing the cell by gene count matrix,
        where the `.var_names` contains genes.
    """
    from tqdm import tqdm

    if peak_mat.obs_names != gene_mat.obs_names:
        raise NameError("gene matrix and peak matrix should have the same obs_names")

    network = prune_network(network, gene_mat.var_names)
    if network.num_edges() == 0:
        return network

    genes = []
    regions = set()
    for nd in network.node_indices():
        parents = network.predecessor_indices(nd)
        if len(parents) > 0:
            genes.append(nd)
            for x in parents:
                regions.add(x)
    regions = list(regions)

    gene_idx = gene_mat.var_ix([network[x] for x in genes])
    gene_mat = gene_mat.X[:, gene_idx]
    region_idx = peak_mat.var_ix([network[x] for x in regions])
    peak_mat = peak_mat.X[:, region_idx]
    region_idx_map = dict(zip(regions, range(len(regions))))

    for i in tqdm(range(len(genes))):
        nd_y = genes[i]
        nd_X = network.predecessor_indices(nd_y)
        y = gene_mat[:, i]
        X = peak_mat[:, [region_idx_map[nd] for nd in nd_X]]
        scores = gbTree(X, y, tree_method='gpu_hist')

        for nd, sc in zip(nd_X, scores):
            network.update_edge(nd, nd_y, sc)

    return network

def prune_network(
    network: retworkx.PyDiGraph,
    selected_nodes: list[str],
    keep_neighbor: bool = True,
) -> retworkx.PyDiGraph:
    """
    Prune the network.

    Parameters
    ----------
    selected_nodes
        The nodes to retain. 
    keep_neighbors
        Also keep neighbors of the selected nodes.
    """
    nodes = []
    selected_nodes = set(selected_nodes)
    for nd in network.node_indices():
        if network[nd] in selected_nodes:
            nodes.append(nd)
            if keep_neighbor:
                nodes.extend(network.predecessor_indices(nd))
                nodes.extend(network.successor_indices(nd))
    return network.subgraph(nodes)

def elastic_net(X, y):
    from sklearn.linear_model import ElasticNet
    regr = ElasticNet(random_state=0).fit(np.asarray(X.todense()), np.asarray(y.todense()))
    return regr.coef_

def gbTree(X, y, tree_method = "hist"):
    import xgboost as xgb
    model = xgb.XGBRegressor(tree_method = tree_method)
    return model.fit(X, y.todense()).feature_importances_

