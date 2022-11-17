from __future__ import annotations
from typing_extensions import Literal

import logging
from pathlib import Path
import numpy as np
import functools

from snapatac2._snapatac2 import AnnData, AnnDataSet
from snapatac2._utils import is_anndata 
from snapatac2.tools import leiden
from snapatac2.preprocessing import knn

__all__ = ['aggregate_X', 'aggregate_cells']

def aggregate_X(
    adata: AnnData | AnnDataSet,
    groupby: str | list[str] | None = None,
    normalize: Literal["RPM", "RPKM"] | None = None,
    file: Path | None = None,
) -> np.ndarray | AnnData:
    """
    Aggregate values in adata.X in a row-wise fashion.

    Aggregate values in adata.X in a row-wise fashion. This is used to compute
    RPKM or RPM values stratified by user-provided groupings.

    Parameters
    ----------
    adata
        The AnnData or AnnDataSet object.
    groupby
        Group the cells into different groups. If a `str`, groups are obtained
        from `.obs[groupby]`.
    normalize
        normalization method: "RPM" or "RPKM".
    file
        if provided, the results will be saved to a new h5ad file.

    Returns
    -------
    np.ndarray | AnnData
        If `grouby` is `None`, return a 1d array. Otherwise, return an AnnData
        object.
    """
    import polars as pl
    from natsort import natsorted
    import anndata as ad

    def norm(x):
        if normalize is None:
            return x
        elif normalize == "RPKM":
            size_factor = _get_sizes(adata.var_names) / 1000.0
            return _normalize(x, size_factor)
        elif normalize == "RPM":
            return _normalize(x)
        else:
            raise NameError("Normalization method must be 'RPKM' or 'RPM'")

    if groupby is None:
        row_sum = functools.reduce(
            lambda a, b: a + b,
            (np.ravel(chunk.sum(axis=0)) for chunk, _, _ in adata.chunked_X(1000)),
        )
        row_sum = norm(row_sum)
        return row_sum
    else:
        groups = adata.obs[groupby].to_numpy() if isinstance(groupby, str) else np.array(groupby)
        if groups.size != adata.n_obs:
            raise NameError("the length of `groupby` should equal to the number of obervations")

        result = {x: np.zeros(adata.n_vars) for x in natsorted(np.unique(groups))}
        for chunk, start, stop in adata.chunked_X(2000):
            for i in range(start, stop):
                result[groups[i]] += chunk[i-start, :]
        for k in result.keys():
            result[k] = norm(np.ravel(result[k]))

        keys, values = zip(*result.items())
        if file is None:
            out_adata = ad.AnnData(X=np.array(values))
        else:
            out_adata = AnnData(filename=file, X=np.array(values))
        out_adata.obs_names = list(keys)
        out_adata.var_names = adata.var_names
        return out_adata

def aggregate_cells(
    adata: AnnData | AnnDataSet | np.ndarray,
    use_rep: str = 'X_spectral',
    target_num_cells: int | None = None,
    min_cluster_size: int = 50,
    random_state: int = 0,
    key_added: str = 'pseudo_cell',
    inplace: bool = True,
) -> np.ndarray | None:
    """Aggregate cells into pseudo-cells.

    Aggregate cells into pseudo-cells by iterative clustering.

    Parameters
    ----------
    adata
        AnnData or AnnDataSet object or matrix.
    use_rep
        `adata.obs` key for retrieving the input matrix.
    target_num_cells
        If None, `target_num_cells = num_cells / min_cluster_size`.
    min_cluster_size
        The minimum size of clusters.
    random_state
        Change the initialization of the optimization.
    key_added
        `adata.obs` key under which to add the cluster labels.
    inplace
        Whether to store the result in the anndata object.

    Returns
    -------
    np.ndarray | None
        If `inplace=False`, return the result as a numpy array.
        Otherwise, store the result in `adata.obs[`key_added`]`.
    """
    def clustering(data):
        return leiden(knn(data), resolution=1, objective_function='modularity',
            min_cluster_size=min_cluster_size, random_state=random_state)

    if is_anndata(adata):
        X = adata.obsm[use_rep]
    else:
        inplace = False
        X = adata

    if target_num_cells is None:
        target_num_cells = X.shape[0] // min_cluster_size

    logging.info("Perform initial clustering ...")
    membership = clustering(X).astype('object')
    cluster_ids = [x for x in np.unique(membership) if x != "-1"]
    ids_next = cluster_ids
    n_clusters = len(cluster_ids)
    depth = 0
    while n_clusters < target_num_cells and len(ids_next) > 0:
        depth += 1
        logging.info("Iterative clustering: {}, number of clusters: {}".format(depth, n_clusters))
        ids = set()
        for cid in ids_next:
            mask = membership == cid
            sub_clusters = clustering(X[mask, :])
            n_sub_clusters = np.count_nonzero(np.unique(sub_clusters) != "-1")
            if n_sub_clusters > 1 and np.count_nonzero(sub_clusters != "-1") / sub_clusters.shape[0] > 0.9:
                n_clusters += n_sub_clusters - 1
                for i, i_ in enumerate(np.where(mask)[0]):
                    lab = sub_clusters[i]
                    if lab == "-1":
                        membership[i_] = lab
                    else:
                        new_lab = membership[i_] + "." + lab
                        membership[i_] = new_lab
                        ids.add(new_lab)
            if n_clusters >= target_num_cells:
                break
        ids_next = ids
    logging.info("Asked for {} pseudo-cells; Got: {}.".format(target_num_cells, n_clusters))

    if inplace:
        import polars
        adata.obs[key_added] = polars.Series(
            [str(x) for x in membership],
            dtype=polars.datatypes.Categorical,
        )
    else:
        return membership
 
def marker_enrichment(
    gene_matrix: AnnData,
    groupby: str | list[str],
    markers: dict[str, list[str]],
    min_num_markers: int = 1,
    hierarchical: bool = True,
):
    """
    Parameters
    ----------
    gene_matrix
        The cell by gene activity matrix.
    groupby
        Group the cells into different groups. If a `str`, groups are obtained from
        `.obs[groupby]`.
    """
    from scipy.stats import zscore
    import polars as pl

    gene_names = dict((x.upper(), i) for i, x in enumerate(gene_matrix.var_names))
    retained = []
    removed = []
    for key in markers.keys():
        genes = []
        for name in markers[key]:
            name = name.upper()
            if name in gene_names:
                genes.append(gene_names[name])
        if len(genes) >= min_num_markers:
            retained.append((key, genes))
        else:
            removed.append(key)
    if len(removed) > 0:
        logging.warn("The following cell types are not annotated because they have less than {} marker genes: {}", min_num_markers, removed)

    aggr_counts = aggregate_X(gene_matrix, groupby=groupby, normalize="RPM")
    zscores = zscore(
        np.log2(np.vstack(list(aggr_counts.values())) + 1),
        axis = 0,
    )

    if hierarchical:
        return _hierarchical_enrichment(dict(retained), zscores)
    else:
        df = pl.DataFrame(
            np.vstack([zscores[:, genes].mean(axis = 1) for _, genes in retained]),
            columns = list(aggr_counts.keys()),
        )
        df.insert_at_idx(0, pl.Series("Cell type", [cell_type for cell_type, _ in retained]))
        return df

def _hierarchical_enrichment(
    marker_genes,
    zscores,
):
    from scipy.cluster.hierarchy import linkage, to_tree
    from collections import Counter
    
    def jaccard_distances(x):
        def jaccard(a, b):
            a = set(a)
            b = set(b)
            return 1 - len(a.intersection(b)) / len(a.union(b))

        result = []
        n = len(x)
        for i in range(n):
            for j in range(i+1, n):
                result.append(jaccard(x[i], x[j]))
        return result

    def make_tree(Z, genes, labels):
        def get_genes_weighted(node, node2 = None):
            leaves = node.pre_order(lambda x: x.id)
            if node2 is not None:
                leaves = leaves + node2.pre_order(lambda x: x.id)
            n = len(leaves)
            count = Counter(g for i in leaves for g in genes[i])
            for key in count.keys():
                count[key] /= n
            return count
        
        def normalize_weights(a, b):
            a_ = []
            for k, v in a.items():
                if k in b:
                    v = v - b[k]
                if v > 0:
                    a_.append((k, v))
            return a_
        
        def process(pid, x, score):
            scores.append(score)
            parents.append(pid)
            ids.append(x.id)
            if x.id < len(labels):
                labels_.append(labels[x.id])
            else:
                labels_.append("")
            go(x)     

        def norm(b, x):
            return np.sqrt(np.exp(b) * np.exp(x))

        def go(tr):
            def sc_fn(gene_w):
                if len(gene_w) > 0:
                    idx, ws = zip(*gene_w)
                    return np.average(zscores[:, list(idx)], axis = 1, weights=list(ws))
                else:
                    return np.zeros(zscores.shape[0])

            left = tr.left
            right = tr.right
            if left is not None and right is not None:
                genes_left = get_genes_weighted(left)
                genes_right = get_genes_weighted(right)
                base = sc_fn(list(get_genes_weighted(left, right).items()))
                sc_left = sc_fn(normalize_weights(genes_left, genes_right))
                sc_right = sc_fn(normalize_weights(genes_right, genes_left))
                process(tr.id, left, norm(base, sc_left))
                process(tr.id, right, norm(base, sc_right))
                
        root = to_tree(Z)
        ids = [root.id]
        parents = [""]
        labels_ = [""]
        scores = [np.zeros(zscores.shape[0])]
        go(root)
        return (ids, parents, labels_, np.vstack(scores).T)

    jm = jaccard_distances([v for v in marker_genes.values()])
    Z = linkage(jm, method='average')
    return make_tree(
        Z, list(marker_genes.values()), list(marker_genes.keys()),
    )


def _groupby(x, groups):
    idx = groups.argsort()
    groups = groups[idx]
    x = x[idx]
    u, indices = np.unique(groups, return_index=True)
    splits = np.split(np.arange(x.shape[0]), indices[1:])
    return dict((label, x[indices, :]) for (label, indices) in zip(u, splits))

def _normalize(x, size_factor = None):
    result = x / (x.sum() / 1000000.0)
    if size_factor is not None:
        result /= size_factor
    return result

def _get_sizes(regions):
    def size(x):
        x = x.split(':')[1].split("-")
        return int(x[1]) - int(x[0])
    return np.array(list(size(x) for x in regions))