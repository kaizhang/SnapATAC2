from warnings import warn

import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np
import pandas as pd


def sdi(data):
    """
    Calculate the Shannon Diversity Index (SDI) for a categorical data series.

    Higher SDI indicates higher diversity.
    """
    if len(data) == 0:
        return 0
    counts = pd.Series(data).value_counts()
    ratio = counts / counts.sum()
    score = -(np.log(ratio) * ratio).sum()
    return score


def calculate_direct_confusion(*args, **kwargs):
    warn("This function is deprecated. Call calculate_overlap_score instead", DeprecationWarning)
    return calculate_overlap_score(*args, **kwargs)


def _get_overlap_score(left_values, right_values):
    score = np.min([left_values, right_values], axis=0).sum()
    return score


def calculate_overlap_score(left_part, right_part):
    """
    Calculate the overlap score between intra-dataset clusters using co-cluster information.

    Input are 2 dataframes for left/source and right/target dataset,
    Each dataframe only contain 2 columns, first is original cluster, second is co-cluster.
    The returned confusion matrix will be the form of source-cluster by target-cluster.

    Parameters
    ----------
    left_part
        Dataframe for left/source dataset.
    right_part
        Dataframe for right/target dataset.
    """
    left_part = left_part.astype(str)
    original_left_name = left_part.columns[0]
    left_part.columns = ["cluster", "co_cluster"]

    right_part = right_part.astype(str)
    original_right_name = right_part.columns[0]
    if original_right_name == original_left_name:
        original_right_name += "_1"
    right_part.columns = ["cluster", "co_cluster"]

    left_confusion = left_part.groupby("cluster")["co_cluster"].value_counts().unstack()
    right_confusion = right_part.groupby("cluster")["co_cluster"].value_counts().unstack()

    left_confusion_portion = left_confusion.divide(left_confusion.sum(axis=1), axis=0).fillna(0)
    right_confusion_portion = right_confusion.divide(right_confusion.sum(axis=1), axis=0).fillna(0)

    union_index = left_confusion_portion.columns.intersection(right_confusion_portion.columns)
    left_confusion_portion = left_confusion_portion.reindex(columns=union_index).fillna(0)
    right_confusion_portion = right_confusion_portion.reindex(columns=union_index).fillna(0)

    records = []
    for left_cluster, left_row in left_confusion_portion.iterrows():
        for right_cluster, right_row in right_confusion_portion.iterrows():
            overlap_value = _get_overlap_score(left_row.values, right_row.values)
            records.append([left_cluster, right_cluster, overlap_value])

    flat_confusion_matrix = pd.DataFrame(records, columns=[original_left_name, original_right_name, "overlap_value"])

    confusion_matrix = flat_confusion_matrix.pivot(
        index=original_left_name, columns=original_right_name, values="overlap_value"
    )
    return confusion_matrix


def calculate_diagonal_score(confusion_matrix, col_group, row_group):
    """
    Given a confusion matrix, evaluate the overall integration performance with the diagonal score.

    Parameters
    ----------
    confusion_matrix :
        A confusion matrix.
    col_group :
        Integration group for the columns.
    row_group :
        Integration group for the rows.

    Returns
    -------
    float
    """
    # group the confusion matrix by the col_group and row_group,
    # then for each block, calculate the mean overlap score
    group_overlap_score_mean = confusion_matrix.groupby(col_group, axis=1).apply(
        lambda sub_df: sub_df.groupby(row_group, axis=0).mean().mean(axis=1)
    )

    # diagonal score is the ratio of the diagonal overlap score sum to the non-diagonal overlap score sum
    diag_sum = np.diag(group_overlap_score_mean, 0).sum()
    non_diag_sum = group_overlap_score_mean.values.sum() - diag_sum

    ratio = diag_sum / non_diag_sum
    return ratio


def confusion_matrix_clustering(
    confusion_matrix, min_value=0.1, max_value=0.9, partition_type=None, resolution=1, seed=0
):
    """
    Given a confusion matrix, bi-clustering the matrix using Leiden Algorithm.

    Parameters
    ----------
    confusion_matrix :
        A confusion matrix. Row is query, column is reference.
    min_value :
        minimum value to be used as an edge weight.
    max_value :
        maximum value to be used as an edge weight. Larger value will be capped to this value.
    partition_type :
        The type of partition to be used. See leidenalg documentation for details.
        If None, use the :class:`~leidenalg.RBConfigurationVertexPartition` partition type.
    resolution :
        The resolution parameter to be used. See leidenalg documentation for details.
    seed :
        random seed for Leiden Algorithm.

    Returns
    -------
    query_group:
        A series of query cluster integration group.
    reference_group:
        A series of reference cluster integration group.
    confusion_matrix：
        A confusion matrix ordered by the integration group.
    g:
        The Graph object used for leiden clustering and determine integration groups.
    modularity_score:
        The modularity score of the graph partition by integration groups.
    """
    # make sure confusion matrix index and columns are unique
    try:
        assert confusion_matrix.index.duplicated().sum() == 0
        assert confusion_matrix.columns.duplicated().sum() == 0
    except AssertionError:
        raise ValueError("Confusion matrix index and columns should be unique")

    confusion_matrix = confusion_matrix.copy()
    # map id to int
    # row is query, column is reference
    query_idx_map = {c: i for i, c in enumerate(confusion_matrix.index)}
    ref_idx_map = {c: i + len(query_idx_map) for i, c in enumerate(confusion_matrix.columns)}
    confusion_matrix.index = confusion_matrix.index.map(query_idx_map)
    confusion_matrix.columns = confusion_matrix.columns.map(ref_idx_map)

    # build a weighted graph from sig scores
    edges = confusion_matrix.unstack()
    edges = edges[edges > min_value].copy().reset_index()
    edges.columns = ["ref", "query", "weight"]
    g = nx.Graph()
    for _, (ref, query, weight) in edges.iterrows():
        weight = min(max_value, weight)
        g.add_edge(ref, query, weight=weight)
        # convert to igraph to run leiden
    h = ig.Graph.from_networkx(g)

    if partition_type is None:
        partition_type = la.RBConfigurationVertexPartition

    # leiden clustering
    partition = la.find_partition(h, partition_type, weights="weight", seed=seed, resolution_parameter=resolution)

    # store output into adata.obs
    groups = np.array(partition.membership)
    idx_to_group = {int(node): g for node, g in zip(g.nodes, groups)}

    # map cluster back to original labels
    # -1 means a ref or query cluster is not co-clustered with the other dataset,
    # so the OS is very small and not included in the graph
    query_group = pd.Series(query_idx_map).map(idx_to_group).fillna(-1).astype(int)
    ref_group = pd.Series(ref_idx_map).map(idx_to_group).fillna(-1).astype(int)

    # also make an ordered confusion matrix
    confusion_matrix.index = confusion_matrix.index.map({v: k for k, v in query_idx_map.items()})
    confusion_matrix.columns = confusion_matrix.columns.map({v: k for k, v in ref_idx_map.items()})
    confusion_matrix = confusion_matrix.loc[query_group.sort_values().index, ref_group.sort_values().index].copy()

    # calculate the graph modularity after partition
    from collections import defaultdict

    from networkx.algorithms.community import modularity

    group_nodes = defaultdict(set)
    for group, node in zip(groups, g.nodes):
        group_nodes[group].add(node)
    modularity_score = modularity(g, group_nodes.values())
    return query_group, ref_group, confusion_matrix, g, modularity_score
