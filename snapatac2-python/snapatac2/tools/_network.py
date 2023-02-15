from __future__ import annotations
from typing_extensions import Literal

import logging
from typing import Callable
from pathlib import Path
import numpy as np
import rustworkx as rx
import scipy.sparse as sp

from snapatac2.genome import Genome
from snapatac2._utils import fetch_seq
from snapatac2._snapatac2 import (
    AnnData, AnnDataSet, link_region_to_gene, PyDNAMotif, spearman
)

__all__ = ['NodeData', 'LinkData',
           'init_network_from_annotation', 'add_cor_scores', 'add_regr_scores',
           'add_tf_binding', 'link_tf_to_gene', 'prune_network', 'pagerank']

class NodeData:
    def __init__(self, id: str = "", type: str = "") -> None:
        self.id = id
        self.type = type
        self.regr_fitness = None

    def __repr__(self):
        return str(self.__dict__)

class LinkData:
    def __init__(
        self,
        distance: int =0,
        label: str | None = None,
    ) -> None:
        self.distance = distance
        self.label = label
        self.regr_score = None
        self.cor_score = None
    
    def __repr__(self):
        return str(self.__dict__)
   
def init_network_from_annotation(
    regions: list[str],
    anno_file: Path | Genome,
    upstream: int = 250000,
    downstream: int = 250000,
    id_type: Literal["gene_name", "gene_id", "transcript_id"] = "gene_name",
    coding_gene_only: bool = True,
) -> rx.PyDiGraph:
    """
    Build CRE-gene network from gene annotations.

    Link CREs to genes if they are close to genes' promoter regions.

    Parameters
    ----------
    regions
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
    rx.PyDiGraph:
        A network where peaks/regions point towards genes if they are within genes'
        regulatory domains.
    """
    if isinstance(anno_file, Genome):
        anno_file = anno_file.fetch_annotations()
        
    region_added = {}
    graph = rx.PyDiGraph()
    links = link_region_to_gene(
        regions,
        str(anno_file),
        upstream,
        downstream,
        id_type,
        coding_gene_only,
    )
    for (id, type), regions in links.items():
        to = graph.add_node(NodeData(id.upper(), type))
        for i, t, distance in regions:
            key = (i, t)
            if key in region_added:
                graph.add_edge(region_added[key], to, LinkData(distance))
            else:
                region_added[key] = graph.add_parent(to, NodeData(i, t), LinkData(distance))
    return graph

def add_cor_scores(
    network: rx.PyDiGraph,
    *,
    gene_mat: AnnData | AnnDataSet | None = None,
    peak_mat: AnnData | AnnDataSet | None = None,
    select: list[str] | None = None,
    overwrite: bool = False,
):
    """
    Compute correlation scores for any two connected nodes in the network.

    This function can be used to compute correlation scores for any type of
    associations. There are typically three types of edges in the network:

    1. Region -> gene: CREs regulate target genes.
    2. Gene -> gene: genes regulate other genes.
    3. Gene -> region: TFs bind to CREs.

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
    overwrite
        Whether to overwrite existing records.
    """
    from tqdm import tqdm

    key = "cor_score"
    if list(peak_mat.obs_names) != list(gene_mat.obs_names):
        raise NameError("gene matrix and peak matrix should have the same obs_names")
    if select is not None:
        select = set(select)
    without_overwrite = None if overwrite else key 

    if network.num_edges() > 0:
        data = _get_data_iter(network, peak_mat, gene_mat, select, without_overwrite)
        for (nd_X, X), (nd_y, y) in tqdm(data):
            if sp.issparse(X):
                X = X.todense()
            if sp.issparse(y):
                y = y.todense()
            scores = np.ravel(spearman(X.T, y.reshape((1, -1))))
            for nd, sc in zip(nd_X, scores):
                setattr(network.get_edge_data(nd, nd_y), key, sc)

def add_regr_scores(
    network: rx.PyDiGraph,
    *,
    peak_mat: AnnData | AnnDataSet | None = None,
    gene_mat: AnnData | AnnDataSet | None = None,
    select: list[str] | None = None,
    method: Literal["gb_tree", "elastic_net"] = "elastic_net",
    scale_X: bool = False,
    scale_Y: bool = False,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    use_gpu: bool = False,
    overwrite: bool = False,
):
    """
    Perform regression analysis for nodes and their parents in the network.

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
    method
        Regresson model.
    scale_X
        Whether to scale the features.
    scale_Y
        Whether to scale the response variable.
    alpha
        Constant that multiplies the penalty terms in 'elastic_net'.
    l1_ratio
        Used in 'elastic_net'. The ElasticNet mixing parameter,
        with `0 <= l1_ratio <= 1`. For `l1_ratio = 0` the penalty is an L2 penalty.
        For `l1_ratio = 1` it is an L1 penalty. For `0 < l1_ratio < 1`,
        the penalty is a combination of L1 and L2.
    use_gpu
        Whether to use gpu
    overwrite
        Whether to overwrite existing records.
    """
    from tqdm import tqdm

    key = "regr_score"
    if list(peak_mat.obs_names) != list(gene_mat.obs_names):
        raise NameError("gene matrix and peak matrix should have the same obs_names")
    if select is not None:
        select = set(select)
    without_overwrite = None if overwrite else key 
    tree_method = "gpu_hist" if use_gpu else "hist"
    
    if network.num_edges() == 0:
        return network

    for (nd_X, X), (nd_y, y) in tqdm(_get_data_iter(network, peak_mat, gene_mat, select, without_overwrite, scale_X, scale_Y)):
        y = np.ravel(y.todense()) if sp.issparse(y) else y
        if method == "gb_tree":
            scores, fitness = _gbTree(X, y, tree_method=tree_method)
        elif method == "elastic_net":
            scores, fitness = _elastic_net(X, y, alpha, l1_ratio)
        elif method == "logistic_regression":
            scores, fitness = _logistic_regression(X, y)
        else:
            raise NameError("Unknown method")
        network[nd_y].regr_fitness = fitness
        for nd, sc in zip(nd_X, scores):
            setattr(network.get_edge_data(nd, nd_y), key, sc)

def add_tf_binding(
    network: rx.PyDiGraph,
    *,
    motifs: list[PyDNAMotif],
    genome_fasta: Path | Genome,
    pvalue: float = 1e-5,
):
    """Add TF motif binding information.

    Parameters
    ----------
    network
        Network
    motifs
        TF motifs
    genome_fasta
        A fasta file containing the genome sequences or a Genome object.
    pvalue
        P-value threshold for motif binding.
    """
    from pyfaidx import Fasta
    from tqdm import tqdm
    import itertools

    regions = [(i, network[i].id) for i in network.node_indices() if network[i].type == "region"]
    logging.info("Fetching {} sequences ...".format(len(regions)))
    genome = genome_fasta.fetch_fasta() if isinstance(genome_fasta, Genome) else str(genome_fasta)
    genome = Fasta(genome, one_based_attributes=False)
    sequences = [fetch_seq(genome, region) for _, region in regions]

    logging.info("Searching for the binding sites of {} motifs ...".format(len(motifs)))
    for motif in tqdm(motifs):
        bound = motif.with_nucl_prob().exists(sequences, pvalue=pvalue)
        if any(bound):
            name = motif.id if motif.name is None else motif.name
            nid = network.add_node(NodeData(name.upper(), "motif"))
            network.add_edges_from(
                [(nid, i, LinkData()) for i, _ in itertools.compress(regions, bound)]
            )

def link_tf_to_gene(network: rx.PyDiGraph) -> rx.PyDiGraph:
    """Contruct a genetic network by linking TFs to target genes.
    
    Convert the network to a genetic network.
    :func:`~snapatac2.tl.add_tf_binding` must be ran first in order to use this function.

    Parameters
    ----------

    Returns 
    -------
    rx.PyDiGraph
    """
    def aggregate(edge_data):
        best = 0
        for a, b in edge_data:
            sc = min(abs(a.cor_score), abs(b.cor_score))
            if sc >= best:
                e1 = a
                e2 = b
        return (e1, e2)

    graph = rx.PyDiGraph()

    genes = {}
    for node in network.nodes():
        if node.type == "gene":
            genes[node.id] = graph.add_node(node)

    edges = []
    for nid in network.node_indices():
        node = network[nid]
        if node.type == "motif" and node.id in genes:
            fr = genes[node.id]
            targets = {}
            for succ_id in network.successor_indices(nid):
                target = network[succ_id]
                if target.type == "region":
                    edge_data = network.get_edge_data(nid, succ_id)
                    for gene_id in network.successor_indices(succ_id):
                        region_gene = network.get_edge_data(succ_id, gene_id)
                        to = genes[network[gene_id].id]
                        if to in targets :
                            targets[to].append((edge_data, region_gene))
                        else:
                            targets[to] = [(edge_data, region_gene)]

            for k, v in targets.items():
                    #e1, e2 = aggregate(v)
                    #label = ''.join(['+' if x.cor_score > 0 else '-' for x in [e1, e2]])
                    #edges.append((fr, k, LinkData(label=label)))
                    edges.append((fr, k, LinkData()))

    graph.add_edges_from(edges)

    remove = []
    for nid in graph.node_indices():
        if graph.in_degree(nid) + graph.out_degree(nid) == 0:
            remove.append(nid)
    if len(remove) > 0:
        graph.remove_nodes_from(remove)
    
    return graph

def prune_network(
    network: rx.PyDiGraph,
    node_filter: Callable[[NodeData], bool] | None = None,
    edge_filter: Callable[[int, int, LinkData], bool] | None = None,
    remove_isolates: bool = True,
) -> rx.PyDiGraph:
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
    remove_isolates
        Whether to remove isolated nodes.

    Returns
    -------
    rx.PyDiGraph
    """
    graph = rx.PyDiGraph()
    
    node_retained = [nid for nid in network.node_indices()
                     if node_filter is None or node_filter(network[nid])]              
    node_indices = graph.add_nodes_from([network[nid] for nid in node_retained])
    node_index_map = dict(zip(node_retained, node_indices))
   
    edge_retained = [(node_index_map[fr], node_index_map[to], data)
                     for fr, to, data in network.edge_index_map().values()
                     if fr in node_index_map and to in node_index_map and
                        (edge_filter is None or edge_filter(fr, to, data))]

    graph.add_edges_from(edge_retained)

    if remove_isolates:
        remove = []
        for nid in graph.node_indices():
            if graph.in_degree(nid) + graph.out_degree(nid) == 0:
                remove.append(nid)
        if len(remove) > 0:
            graph.remove_nodes_from(remove)
            logging.info("Removed {} isolated nodes.".format(len(remove)))

    return graph

def pagerank(
    network,
    node_weights: str | list[float] | None = None,
    edge_weights: str | list[float] | None = None,
) -> list[tuple[str, float]]:
    tfs = {network[nid].id for nid in network.node_indices() if network.out_degree(nid) > 0}
    g = _to_igraph(network, node_weights, edge_weights, True)
    pagerank_scores = g.personalized_pagerank(
        reset=None if node_weights is None else 'weight',
        weights=None if edge_weights is None else 'weight',
    )
    return [(i['name'], s) for i, s in zip(g.vs, pagerank_scores) if i['name'] in tfs]

def _to_igraph(
    graph,
    node_weights: str | list[float] | None = None,
    edge_weights: str | list[float] | None = None,
    reverse_edge: bool = False,
):
    import igraph as ig
    g = ig.Graph()

    nodes = [x.id for x in graph.nodes()]
    node_attributes = None
    if node_weights is not None:
        if isinstance(node_weights, str):
            node_attributes = {"weight": [getattr(x, node_weights) for x in graph.nodes()]}
        else:
            node_attributes = {"weight": node_weights}
    g.add_vertices(nodes, attributes=node_attributes)
    
    edges = []
    if edge_weights is not None and isinstance(edge_weights, list):
        weights = edge_weights
    else:
        weights = []
    for fr, to, edge in graph.edge_index_map().values():
        if reverse_edge:
            edges.append((graph[to].id, graph[fr].id))
        else:
            edges.append((graph[fr].id, graph[to].id))
        if edge_weights is not None and isinstance(edge_weights, str):
            weights.append(getattr(edge, edge_weights))
    if len(weights) > 0:
        edge_attributes = {"weight": weights}
    else:
        edge_attributes = None
    g.add_edges(edges, attributes=edge_attributes)

    return g

def _network_stats(network: rx.PyDiGraph):
    from collections import defaultdict

    nNodes = network.num_nodes()
    nEdges = network.num_edges()
    region_stat = defaultdict(lambda: {'nParents': defaultdict(lambda: 0), 'nChildren': defaultdict(lambda: 0)})
    motif_stat = defaultdict(lambda: {'nParents': defaultdict(lambda: 0), 'nChildren': defaultdict(lambda: 0)})
    gene_stat = defaultdict(lambda: {'nParents': defaultdict(lambda: 0), 'nChildren': defaultdict(lambda: 0)})

    for fr, to, data in network.edge_index_map().values():
        fr_type = network[fr].type
        to_type = network[to].type
 

class _DataPairIter:
    """
    Interator generating X and y pairs.

    ...

    Attributes
    ----------
    regulator_mat
        Regulator data.
    regulatee_mat
        Regulatee data.
    regulator_idx_map
        Node id to regulator matrix index map.
    regulator_ids
        Node ids of regulators.
    regulatee_ids
        Node ids of regulatees.
    """
    def __init__(
        self,
        mat_X,
        mat_Y,
        idx_map_X,
        id_XY,
    ) -> None:
        self.mat_X = mat_X
        self.mat_Y = mat_Y
        self.idx_map_X = idx_map_X
        self.id_XY = id_XY
        self.index = 0

    def __len__(self):
        return self.mat_Y.shape[1]

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.__len__():
            raise StopIteration

        nd_X, nd_y = self.id_XY[self.index]
        y = self.mat_Y[:, self.index]
        X = self.mat_X[:, [self.idx_map_X[nd] for nd in nd_X]]

        self.index += 1
        return (nd_X, X), (nd_y, y)

def _get_data_iter(
    network: rx.PyDiGraph,
    peak_mat: AnnData | AnnDataSet | None,
    gene_mat: AnnData | AnnDataSet | None,
    select: set[str] | None = None,
    without_overwrite: str | None = None,
    scale_X: bool = False,
    scale_Y: bool = False,
) -> _DataPairIter:
    """
    """
    from scipy.stats import zscore

    def get_mat(nids, node_getter, gene_mat, peak_mat):
        genes = []
        peaks = []
        mats = []

        for x in nids:
            nd = node_getter(x) 
            if nd.type == "gene" or nd.type == "motif":
                genes.append(x)
            elif nd.type == "region":
                peaks.append(x)
            else:
                raise NameError("unknown type: {}".format(nd.type))

        if len(genes) != 0:
            if len(genes) == gene_mat.n_vars:
                mats.append(gene_mat.X[:])
            else:
                idx_map = {x.upper(): i for i, x in enumerate(gene_mat.var_names)}
                ix = [idx_map[node_getter(x).id] for x in genes]
                mats.append(gene_mat.X[:, ix])

        if len(peaks) != 0:
            if len(peaks) == peak_mat.n_vars:
                mats.append(peak_mat.X[:])
            else:
                if peak_mat.isbacked:
                    ix = peak_mat.var_ix([node_getter(x).id for x in peaks])
                else:
                    ix = [peak_mat.var_names.get_loc(node_getter(x).id) for x in peaks]
                mats.append(peak_mat.X[:, ix])
        
        if all([sp.issparse(x) for x in mats]):
            mat = sp.hstack(mats, format="csc")
        else:
            mat = np.hstack(mats)
        return (genes + peaks, mat)

    all_genes = set([x.upper() for x in gene_mat.var_names])
    select = all_genes if select is None else select
    id_XY = []

    for nid in network.node_indices():
        if network[nid].type == "region" or network[nid].id in select:
            parents = [pid for pid, _, edge_data in network.in_edges(nid)
                       if (without_overwrite is None or
                           getattr(edge_data, without_overwrite) is None) and
                          (network[pid].type == "region" or
                           network[pid].id in all_genes)]
            if len(parents) > 0:
                id_XY.append((parents, nid))
    unique_X = list({y for x, _ in id_XY for y in x})

    id_XY, mat_Y = get_mat(id_XY, lambda x: network[x[1]], gene_mat, peak_mat)

    unique_X, mat_X = get_mat(unique_X, lambda x: network[x], gene_mat, peak_mat)

    if scale_X:
        if sp.issparse(mat_X):
            logging.warning("Try to scale a sparse matrix")
        mat_X = zscore(mat_X, axis=0)
    if scale_Y:
        if sp.issparse(mat_Y):
            logging.warning("Try to scale a sparse matrix")
        mat_Y = zscore(mat_Y, axis=0)
        
    return _DataPairIter(
        mat_X,
        mat_Y,
        {v: i for i, v in enumerate(unique_X)},
        id_XY,
    )

def _logistic_regression(X, y):
    from sklearn.linear_model import LogisticRegression 

    y = y != 0
    regr = LogisticRegression(max_iter=1000, random_state=0).fit(X, y)
    return np.ravel(regr.coef_), regr.score(X, y)

def _elastic_net(X, y, alpha=1, l1_ratio=0.5, positive=False):
    from sklearn.linear_model import ElasticNet

    X = np.asarray(X)
    y = np.asarray(y)

    regr = ElasticNet(
        alpha=alpha, l1_ratio=l1_ratio, positive=positive,
        random_state=0, copy_X=False, max_iter=10000,
    ).fit(X, y)
    return regr.coef_, regr.score(X, y)

def _gbTree(X, y, tree_method = "hist"):
    import xgboost as xgb
    regr = xgb.XGBRegressor(tree_method = tree_method).fit(X, y)
    return regr.feature_importances_, regr.score(X, y)