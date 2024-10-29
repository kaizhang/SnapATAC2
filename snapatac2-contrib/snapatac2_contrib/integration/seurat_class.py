import pathlib
from collections import OrderedDict

import anndata
import joblib
import numpy as np
import pandas as pd
import pynndescent
from scipy.cluster.hierarchy import linkage
from scipy.sparse import issparse
from scipy.stats import zscore
import scanpy as sc
import anndata as ad
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, normalize

from .cca import cca, lsi_cca

def scanpy_PCA_plus(
        a: ad.AnnData, n_comps: int,
        weight_by_var: bool = True, **kwargs) -> None:
    """
    Ref:
    https://github.com/satijalab/seurat/blob/1549dcb3075eaeac01c925c4b4bb73c73450fc50/R/dimensional_reduction.R#L897
    """
    sc.pp.pca(a, n_comps=n_comps, **kwargs)
    key_obsm = "X_pca"
    key_uns = "pca"
    if not weight_by_var:
        print("Normalize PCA by diving the singluar values.")
        sdev = np.sqrt(a.uns[key_uns]['variance'])
        singular_values = sdev * np.sqrt(a.shape[0] - 1)
        old_X_pca = a.obsm[key_obsm]
        new_X_pca = old_X_pca / singular_values
        a.obsm[key_obsm] = new_X_pca
        a.obsm[f"scanpy_{key_obsm}"] = old_X_pca
        a.uns[key_uns]['singular_values'] = singular_values
    else:
        print("Keep the scanpy default PCA, i.e., weighted by variance of PCs.")
    return None

def find_neighbor(cc1, cc2, k, random_state=0, n_jobs=-1):
    """
    Find all four way of neighbors for two datasets.

    Parameters
    ----------
    cc1
        cc for dataset 1
    cc2
        cc for dataset 2
    k
        number of neighbors
    random_state
        random seed
    n_jobs
        number of jobs to run in parallel

    Returns
    -------
    11, 12, 21, 22 neighbor matrix in shape (n_cell, k)
    """
    index = pynndescent.NNDescent(
        cc1,
        metric="euclidean",
        n_neighbors=k + 1,
        random_state=random_state,
        parallel_batch_queries=True,
        n_jobs=n_jobs,
    )
    G11 = index.neighbor_graph[0][:, 1: k + 1]
    G21 = index.query(cc2, k=k)[0]
    index = pynndescent.NNDescent(
        cc2,
        metric="euclidean",
        n_neighbors=k + 1,
        random_state=random_state,
        parallel_batch_queries=True,
        n_jobs=n_jobs,
    )
    G22 = index.neighbor_graph[0][:, 1: k + 1]
    G12 = index.query(cc1, k=k)[0]
    return G11, G12, G21, G22


def find_mnn(G12, G21, kanchor):
    """Calculate mutual nearest neighbor for two datasets."""
    anchor = [
        [i, G12[i, j]]
        for i in range(G12.shape[0])
        for j in range(kanchor)
        if (i in G21[G12[i, j], :kanchor])
    ]
    return np.array(anchor)


def min_max(tmp, q_left=1, q_right=90):
    """Normalize to q_left, q_right quantile to 0, 1, and cap extreme values."""
    tmin, tmax = np.percentile(tmp, [q_left, q_right])
    tmp = (tmp - tmin) / (tmax - tmin)
    tmp[tmp > 1] = 1
    tmp[tmp < 0] = 0
    return tmp


def filter_anchor(
    anchor,
    adata_ref=None,
    adata_qry=None,
    scale_ref=False,
    scale_qry=False,
    high_dim_feature=None,
    k_filter=200,
    random_state=0,
    n_jobs=-1,
):
    """
    Check if an anchor is still an anchor when only using the high_dim_features to construct KNN graph.

    If not, remove the anchor.
    """
    if issparse(adata_ref.X):
        ref_data = adata_ref.X[:, high_dim_feature].toarray()
    else:
        ref_data = adata_ref.X[:, high_dim_feature].copy()
    if scale_ref:
        ref_data = zscore(ref_data, axis=0)
    ref_data = normalize(ref_data, axis=1)

    if issparse(adata_qry.X):
        qry_data = adata_qry.X[:, high_dim_feature].toarray()
    else:
        qry_data = adata_qry.X[:, high_dim_feature].copy()
    if scale_qry:
        qry_data = zscore(qry_data, axis=0)
    qry_data = normalize(qry_data, axis=1)

    index = pynndescent.NNDescent(
        ref_data,
        metric="euclidean",
        n_neighbors=k_filter,
        random_state=random_state,
        parallel_batch_queries=True,
        n_jobs=n_jobs,
    )
    G = index.query(qry_data, k=k_filter)[0]
    input_anchors = anchor.shape[0]
    anchor = np.array([xx for xx in anchor if (xx[0] in G[xx[1]])])
    print(
        f"Anchor selected with high CC feature graph: {anchor.shape[0]} / {input_anchors}"
    )
    return anchor


def score_anchor(
    anchor, G11, G12, G21, G22, k_score=30, Gp1=None, Gp2=None, k_local=50
):
    """
    Score the anchor by the number of shared neighbors.

    Parameters
    ----------
    anchor
        anchor in shape (n_anchor, 2)
    G11
        neighbor graph of dataset 1
    G12
        neighbor graph of dataset 1 to 2
    G21
        neighbor graph of dataset 2 to 1
    G22
        neighbor graph of dataset 2
    k_score
        number of neighbors to score the anchor
    Gp1
        Intra-dataset1 kNN graph
    Gp2
        Intra-dataset2 kNN graph
    k_local
        number of neighbors to calculate the local score

    Returns
    -------
    anchor with score in shape (n_anchor, 3): pd.DataFrame
    """
    tmp = [
        len(set(G11[x, :k_score]).intersection(G21[y, :k_score]))
        + len(set(G12[x, :k_score]).intersection(G22[y, :k_score]))
        for x, y in anchor
    ]
    anchor_df = pd.DataFrame(anchor, columns=["x1", "x2"])
    anchor_df["score"] = min_max(tmp)

    if k_local:
        # if k_local is not None, then use local KNN to adjust the score
        share_nn = np.array(
            [len(set(Gp1[i]).intersection(G11[i, :k_local])) for i in range(len(Gp1))]
        )
        tmp = [share_nn[xx] for xx in anchor_df["x1"].values]
        anchor_df["score_local1"] = min_max(tmp)

        share_nn = np.array(
            [len(set(Gp2[i]).intersection(G22[i, :k_local])) for i in range(len(Gp2))]
        )
        tmp = [share_nn[xx] for xx in anchor_df["x2"].values]
        anchor_df["score_local2"] = min_max(tmp)

        anchor_df["score"] = (
            anchor_df["score"] * anchor_df["score_local1"] * anchor_df["score_local2"]
        )
    return anchor_df


def find_order(dist, ncell):
    """Use dendrogram to find the order of dataset pairs."""
    D = linkage(1 / dist, method="average")
    node_dict = {i: [i] for i in range(len(ncell))}
    alignment = []
    for xx in D[:, :2].astype(int):
        if ncell[xx[0]] < ncell[xx[1]]:
            xx = xx[::-1]
        alignment.append([node_dict[xx[0]], node_dict[xx[1]]])
        node_dict[len(ncell)] = node_dict[xx[0]] + node_dict[xx[1]]
        ncell.append(ncell[xx[0]] + ncell[xx[1]])
    return alignment


class SeuratIntegration:
    """Main class for Seurat integration."""

    def __init__(self, n_jobs=-1, random_state=0):
        self.n_jobs = n_jobs

        # intra-dataset KNN graph
        self.k_local = None
        self.key_local = None
        self.local_knn = []

        self.adata_dict = OrderedDict()
        self.n_dataset = 0
        self.n_cells = []
        self.alignments = None
        self.all_pairs = np.array([])
        self._get_all_pairs()

        self.anchor = {}
        self.mutual_knn = {}
        self.raw_anchor = {}
        self.label_transfer_results = {}

        self.random_state = random_state

    def _calculate_local_knn(self):
        """
        Calculate local kNN graph for each dataset.

        If klocal is provided, we calculate the local knn graph to
        evaluate whether the anchor preserves local structure within the dataset.
        One can use a different obsm with key_local to compute knn for each dataset.
        """
        if self.k_local is not None:
            print("Find neighbors within datasets")
            for adata in self.adata_dict.values():
                index = pynndescent.NNDescent(
                    adata.obsm[self.key_local],
                    metric="euclidean",
                    n_neighbors=self.k_local + 1,
                    random_state=self.random_state,
                    parallel_batch_queries=True,
                    n_jobs=self.n_jobs,
                )
                self.local_knn.append(index.neighbor_graph[0][:, 1:])
        else:
            self.local_knn = [None for _ in self.adata_dict.values()]

    def _get_all_pairs(self):
        if self.alignments is not None:
            all_pairs = []
            for pair in self.alignments:
                for xx in pair[0]:
                    for yy in pair[1]:
                        if xx < yy:
                            all_pairs.append(f"{xx}-{yy}")
                        else:
                            all_pairs.append(f"{yy}-{xx}")
            self.all_pairs = np.unique(all_pairs)
        else:
            self.all_pairs = np.array([])

    def _prepare_matrix(self, i, j, key_anchor):
        adata_dict = self.adata_dict
        adata1 = adata_dict[i]
        adata2 = adata_dict[j]

        if key_anchor == "X":
            # in case the adata var is not in the same order
            # select and order the var to make sure it is matched
            if (adata1.shape[1] != adata2.shape[1]) or (
                (adata1.var.index == adata2.var.index).sum() < adata1.shape[1]
            ):
                sel_b = adata1.var.index & adata2.var.index
                U = adata1[:, sel_b].X.copy()
                V = adata2[:, sel_b].X.copy()
            else:
                U = adata1.X.copy()
                V = adata2.X.copy()
        else:
            U = adata1.obsm[key_anchor]
            V = adata2.obsm[key_anchor]

        return U, V

    def _calculate_mutual_knn_and_raw_anchors(self, i, j, U, V, k, k_anchor):
        """
        Calculate the mutual knn graph and raw anchors.

        The results are saved to self.mutual_knn and self.raw_anchor.
        """
        G11, G12, G21, G22 = find_neighbor(U, V, k=k, n_jobs=self.n_jobs)
        raw_anchors = find_mnn(G12, G21, k_anchor)
        self.mutual_knn[(i, j)] = (G11, G12, G21, G22)
        self.raw_anchor[(i, j)] = raw_anchors
        return G11, G12, G21, G22, raw_anchors

    def _pairwise_find_anchor(
        self,
        i,
        i_sel,
        j,
        j_sel,
        dim_red,
        key_anchor,
        svd_algorithm,
        scale1,
        scale2,
        k_anchor,
        k_local,
        k_score,
        ncc,
        max_cc_cell,
        k_filter,
        n_features,
        chunk_size,
        random_state,
        signorm,
    ):
        """Pairwise anchor between two datasets."""
        adata1 = self.adata_dict[i]
        adata2 = self.adata_dict[j]

        min_sample = min(adata1.shape[0], adata2.shape[0])

        if i_sel is not None:
            adata1 = adata1[i_sel, :]
        if j_sel is not None:
            adata2 = adata2[j_sel, :]
        if dim_red in ("cca", "pca", "lsi", "lsi-cca"):
            print(f"1. Prepare input matrix using {key_anchor}.")
            U, V = self._prepare_matrix(i, j, key_anchor=key_anchor)
            if dim_red in ("cca", "pca"):
                print("2. Run CCA")
                U, V, high_dim_feature = cca(
                    data1=U,
                    data2=V,
                    scale1=scale1,
                    scale2=scale2,
                    n_components=ncc,
                    max_cc_cell=max_cc_cell,
                    k_filter=k_filter,
                    n_features=n_features,
                    chunk_size=chunk_size,
                    svd_algorithm=svd_algorithm,
                    random_state=random_state,
                )
            else:
                print("2. Run LSI-CCA")
                U, V = lsi_cca(
                    data1=U,
                    data2=V,
                    scale_factor=100000,
                    n_components=ncc,
                    max_cc_cell=max_cc_cell,
                    chunk_size=chunk_size,
                    svd_algorithm=svd_algorithm,
                    min_cov_filter=5,
                    random_state=random_state,
                )
                high_dim_feature = None
            print("3. Normalize CCV per sample/row")
            U = normalize(U, axis=1)
            V = normalize(V, axis=1)
            print("4. find MNN of U and V to get anchors.")
            _k = max(_temp for _temp
                    in [k_anchor, k_local, k_score] if _temp is not None)
            _k = min(min_sample - 2, _k)
            print(f"Find Anchors using k={_k}")
            G11, G12, G21, G22, raw_anchors = (
                self._calculate_mutual_knn_and_raw_anchors(
                    i=i, j=j, U=U, V=V, k=_k, k_anchor=k_anchor
                )
            )

            print("5. filter anchors by high dimensional neighbors.")
            # compute ccv feature loading
            if k_filter is not None and high_dim_feature is not None:
                if self.n_cells[i] >= self.n_cells[j]:
                    raw_anchors = filter_anchor(
                        anchor=raw_anchors,
                        adata_ref=adata1,
                        adata_qry=adata2,
                        scale_ref=scale1,
                        scale_qry=scale2,
                        high_dim_feature=high_dim_feature,
                        k_filter=k_filter,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs,
                    )
                else:
                    raw_anchors = filter_anchor(
                        anchor=raw_anchors[:, ::-1],
                        adata_ref=adata2,
                        adata_qry=adata1,
                        scale_ref=scale2,
                        scale_qry=scale1,
                        high_dim_feature=high_dim_feature,
                        k_filter=k_filter,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs,
                    )[:, ::-1]
        elif dim_red in ("rpca", "rlsi"):
            print(f"Perform {dim_red}.")
            from .cca import LSI, SVD, downsample

            adata1, adata2 = adata1.X, adata2.X
            k = max(i for i in [k_anchor, k_local, k_score, 50] if i is not None)
            if dim_red == "rpca":
                print("Run rPCA")
                model = SVD(n_components=ncc, random_state=random_state)
            elif dim_red == "rlsi":
                print("Run rLSI")
                model = LSI(n_components=ncc, random_state=random_state)
            else:
                raise ValueError(
                    f"Dimension reduction method {dim_red} is not supported."
                )
            tf1, tf2, scaler1, scaler2 = downsample(
                adata1,
                adata2,
                todense=True if dim_red == "rpca" else False,
                scale1=scale1,
                scale2=scale2,
                max_cc_cell=max_cc_cell,
            )

            # project adata2 to adata1
            model.fit(tf1)
            U = model.transform(adata1, chunk_size=chunk_size, scaler=scaler1)
            V = model.transform(adata2, chunk_size=chunk_size, scaler=scaler2)
            if (dim_red == "pca") and signorm:
                U = U / model.model.singular_values_
                V = V / model.model.singular_values_
            index = pynndescent.NNDescent(
                U,
                metric="euclidean",
                n_neighbors=k + 1,
                random_state=random_state,
                n_jobs=-1,
            )
            G11 = index.neighbor_graph[0][:, 1: k + 1]
            G21 = index.query(V, k=k)[0]

            # project adata1 to adata2
            model.fit(tf2)
            U = model.transform(adata1, chunk_size=chunk_size, scaler=scaler1)
            V = model.transform(adata2, chunk_size=chunk_size, scaler=scaler2)
            if (dim_red == "pca") and signorm:
                U = U / model.model.singular_values_
                V = V / model.model.singular_values_
            index = pynndescent.NNDescent(
                V,
                metric="euclidean",
                n_neighbors=k + 1,
                random_state=random_state,
                n_jobs=-1,
            )
            G22 = index.neighbor_graph[0][:, 1: k + 1]
            G12 = index.query(U, k=k)[0]

            raw_anchors = find_mnn(G12, G21, k_anchor)
        else:
            raise ValueError(f"{dim_red} is not supported.")

        print("6. Score anchors with snn and local structure preservation.")
        anchor_df = score_anchor(
            anchor=raw_anchors,
            G11=G11,
            G12=G12,
            G21=G21,
            G22=G22,
            k_score=k_score,
            k_local=k_local,
            Gp1=self.local_knn[i],
            Gp2=self.local_knn[j])
        return anchor_df

    def find_anchor(
        self,
        adata_list,
        adata_names=None,
        k_local=None,
        key_local="X_pca",
        key_anchor="X",
        dim_red="pca",
        svd_algorithm="randomized",
        scale1=True,
        scale2=True,
        scale_list=None,
        k_filter=None,
        n_features=200,
        n_components=None,
        max_cc_cells=50000,
        chunk_size=50000,
        k_anchor=5,
        k_score=30,
        alignments=None,
        random_state=0,
        signorm=True,
        key_match=None,
    ):
        """Find anchors for each dataset pair.

        Parameters
        ----------
        key_local
            str, used for KNN construction in each dataset
        key_anchor
            str, used for anchor construction
        dim_red
            str from pca, cca, lsi, lsi-cca, rpca, rlsi
            Based on current implementation,
            - choosing pca or cca, will perform CCA
            - choosing lsi or lsi-cca, will perform LSI-CCA
            - rpca is only used when key_match is None at least
        n_components
            int, dim used for CCA and other methods, default None will set it as 50.
        adata_names
            list of int.
        """
        valid_dim_red_name = ["pca", "cca", "lsi", "lsi-cca", "rpca", "rlsi"]
        if dim_red not in valid_dim_red_name:
            raise ValueError(f"Dimension reduction method {dim_red} is not supported.")

        if adata_names is None:
            adata_names = list(range(len(adata_list)))
        try:
            assert len(adata_names) == len(adata_list)
        except AssertionError:
            print("length of adata_names does not match length of adata_list.")

        self.adata_dict = {k: v for k, v in zip(adata_names, adata_list)}
        self.n_dataset = len(adata_list)
        self.n_cells = [adata.shape[0] for adata in adata_list]

        # intra-dataset KNN for scoring the anchors
        self.k_local = k_local
        self.key_local = key_local
        self._calculate_local_knn()

        # alignments and all_pairs
        self.alignments = alignments
        self._get_all_pairs()

        print("Find anchors across datasets.")
        for i in range(self.n_dataset - 1):
            for j in range(i + 1, self.n_dataset):
                if scale_list is not None:
                    scale1 = scale_list[i]
                    scale2 = scale_list[j]
                    print("Get scale1 and scale2 from scale_list")
                    print(f"dataset {i} scale: {scale1}")
                    print(f"dataset {j} scale: {scale2}")

                if key_match is None:
                    anchor_df = self._pairwise_find_anchor(
                        i=i,
                        i_sel=None,
                        j=j,
                        j_sel=None,
                        dim_red=dim_red,
                        key_anchor=key_anchor,
                        svd_algorithm=svd_algorithm,
                        scale1=scale1,
                        scale2=scale2,
                        k_anchor=k_anchor,
                        k_local=k_local,
                        k_score=k_score,
                        ncc=n_components,
                        max_cc_cell=max_cc_cells,
                        k_filter=k_filter,
                        n_features=n_features,
                        chunk_size=chunk_size,
                        random_state=random_state,
                        signorm=signorm,
                    )
                else:
                    tissue = [xx.obs[key_match].unique() for xx in adata_list]
                    sharet = list(set(tissue[i]).intersection(tissue[j]))
                    if len(sharet) > 0:
                        anchor_df_list = []
                        for t in sharet:
                            print(t)
                            adata1 = adata_list[i].copy()
                            adata2 = adata_list[j].copy()

                            idx1 = np.where(adata1.obs[key_match] == t)[0]
                            idx2 = np.where(adata2.obs[key_match] == t)[0]
                            tmp = self._pairwiser_find_anchor(
                                i=i,
                                i_sel=idx1,
                                j=j,
                                j_sel=idx2,
                                dim_red=dim_red,
                                key_anchor=key_anchor,
                                svd_algorithm=svd_algorithm,
                                scale1=scale1,
                                scale2=scale2,
                                k_anchor=k_anchor,
                                k_local=k_local,
                                k_score=k_score,
                                ncc=n_components,
                                max_cc_cell=max_cc_cells,
                                k_filter=k_filter,
                                n_features=n_features,
                                chunk_size=chunk_size,
                                random_state=random_state,
                                signorm=signorm,
                            )
                            tmp["x1"] = idx1[tmp["x1"].values]
                            tmp["x2"] = idx2[tmp["x2"].values]
                            anchor_df_list.append(tmp)
                        anchor_df = pd.concat(anchor_df_list, axis=0)
                    else:
                        anchor_df = self._pairwise_find_anchor(
                            i=i,
                            i_sel=None,
                            j=j,
                            j_sel=None,
                            dim_red="rpca",
                            key_anchor=key_anchor,
                            svd_algorithm=svd_algorithm,
                            scale1=scale1,
                            scale2=scale2,
                            k_anchor=k_anchor,
                            k_local=k_local,
                            k_score=k_score,
                            ncc=n_components,
                            max_cc_cell=max_cc_cells,
                            k_filter=k_filter,
                            n_features=n_features,
                            chunk_size=chunk_size,
                            random_state=random_state,
                            signorm=signorm,
                        )

                # save anchors
                self.anchor[(i, j)] = anchor_df.copy()
                print(
                    f"Identified {len(self.anchor[i, j])} anchors between datasets {i} and {j}."
                )
        return

    def find_nearest_anchor(
        self,
        data,
        data_qry,
        ref,
        qry,
        key_correct="X_pca",
        npc=30,
        k_weight=100,
        sd=1,
        random_state=0,
    ):
        """Find the nearest anchors for each cell in data."""
        print("Initialize")
        cum_ref, cum_qry = [0], [0]
        for xx in ref:
            cum_ref.append(cum_ref[-1] + data[xx].shape[0])
        for xx in qry:
            cum_qry.append(cum_qry[-1] + data[xx].shape[0])

        anchor = []
        for i, xx in enumerate(ref):
            for j, yy in enumerate(qry):
                if xx < yy:
                    tmp = self.anchor[(xx, yy)].copy()
                else:
                    tmp = self.anchor[(yy, xx)].copy()
                    tmp[["x1", "x2"]] = tmp[["x2", "x1"]]
                tmp["x1"] += cum_ref[i]
                tmp["x2"] += cum_qry[j]
                anchor.append(tmp)
        anchor = pd.concat(anchor)
        score = anchor["score"].values
        anchor = anchor[["x1", "x2"]].values

        if key_correct == "X":
            model = PCA(
                n_components=npc, svd_solver="arpack", random_state=random_state
            )
            reduce_qry = model.fit_transform(data_qry)
        else:
            reduce_qry = data_qry[:, :npc]

        print("Find nearest anchors", end=". ")
        index = pynndescent.NNDescent(
            reduce_qry[anchor[:, 1]],
            metric="euclidean",
            n_neighbors=k_weight,
            random_state=random_state,
            parallel_batch_queries=True,
            n_jobs=self.n_jobs,
        )
        k_weight = min(k_weight, anchor.shape[0] - 5)
        k_weight = max(5, k_weight)
        print("k_weight: ", k_weight, end="\n")
        G, D = index.query(reduce_qry, k=k_weight)

        print("Normalize graph")
        cell_filter = D[:, -1] == 0
        D = (1 - D / D[:, -1][:, None]) * score[G]
        D[cell_filter] = score[G[cell_filter]]
        D = 1 - np.exp(-D * (sd**2) / 4)
        D = D / (np.sum(D, axis=1) + 1e-6)[:, None]
        return anchor, G, D, cum_qry

    def transform(
        self,
        data,
        ref,
        qry,
        key_correct,
        npc=30,
        k_weight=100,
        sd=1,
        chunk_size=50000,
        random_state=0,
        row_normalize=True,
    ):
        """Transform query data to reference space."""
        data_ref = np.concatenate(data[ref])
        data_qry = np.concatenate(data[qry])

        anchor, G, D, cum_qry = self.find_nearest_anchor(
            data=data,
            data_qry=data_qry,
            key_correct=key_correct,
            ref=ref,
            qry=qry,
            npc=npc,
            k_weight=k_weight,
            sd=sd,
            random_state=random_state,
        )

        print("Transform data")
        bias = data_ref[anchor[:, 0]] - data_qry[anchor[:, 1]]
        data_prj = np.zeros(data_qry.shape)

        for chunk_start in np.arange(0, data_prj.shape[0], chunk_size):
            data_prj[chunk_start: (chunk_start + chunk_size)] = data_qry[
                chunk_start: (chunk_start + chunk_size)
            ] + (
                D[chunk_start: (chunk_start + chunk_size), :, None]
                * bias[G[chunk_start: (chunk_start + chunk_size)]]
            ).sum(
                axis=1
            )
        for i, xx in enumerate(qry):
            _data = data_prj[cum_qry[i]: cum_qry[i + 1]]
            if row_normalize:
                _data = normalize(_data, axis=1)
            data[xx] = _data
        return data

    def integrate(
        self,
        key_correct,
        row_normalize=True,
        n_components=30,
        k_weight=100,
        sd=1,
        alignments=None,
    ):
        """\
        Map query data to reference space.

        Transform query matrices datasets by transform data matrices
        from query to reference data using the MNN information.
        Reference data will not be changed or perform L2-normalization
        per row if row_normalize = True.
        
        NOTE: From the implementation, it will go through each element in
        the alignments, but only return the last one. So we should
        consider the ref and quey pair for this function.
        
        Parameters
        ----------
        key_correct
            str, X or field of obsm, such as X_pca
        row_normalize
            boolean, if perform L2-normalized for key_correct
            default is True

        Returns
        -------
        List of numpy arrays, ordered by alignments (ref, qeury).
        """
        if alignments is not None:
            self.alignments = alignments

        # find order of pairwise dataset merging with hierarchical clustering
        if self.alignments is None:
            dist = []
            for i in range(self.n_dataset - 1):
                for j in range(i + 1, self.n_dataset):
                    dist.append(
                        len(self.anchor[(i, j)])
                        / min([self.n_cells[i], self.n_cells[j]])
                    )
            self.alignments = find_order(np.array(dist), self.n_cells)
            print(f"Alignments: {self.alignments}")

        print("Merge datasets")
        adata_list = list(self.adata_dict.values())

        # initialize corrected with original data
        if key_correct == "X":
            # correct the original feature matrix
            corrected = [adata_list[i].X.copy() for i in range(self.n_dataset)]
        else:
            # correct dimensionality reduced matrix only
            if row_normalize:
                corrected = [
                    normalize(adata_list[i].obsm[key_correct], axis=1)
                    for i in range(self.n_dataset)]
            else:
                corrected = [
                    adata_list[i].obsm[key_correct] for i in range(self.n_dataset)]

        for xx in self.alignments:
            print(xx)
            corrected = self.transform(
                data=np.array(corrected, dtype="object"),
                ref=xx[0],
                qry=xx[1],
                npc=n_components,
                k_weight=k_weight,
                sd=sd,
                random_state=self.random_state,
                row_normalize=row_normalize,
                key_correct=key_correct,
            )
        return corrected

    def label_transfer(
        self,
        ref,
        qry,
        categorical_key=None,
        continuous_key=None,
        key_dist="X_pca",
        k_weight=100,
        npc=30,
        sd=1,
        chunk_size=50000,
        random_state=0,
    ):
        """Transfer labels from query to reference space."""
        adata_list = list(self.adata_dict.values())

        data_qry = np.concatenate(
            [normalize(adata_list[i].obsm[key_dist], axis=1) for i in qry]
        )
        data_qry_index = np.concatenate([adata_list[i].obs_names for i in qry])

        anchor, G, D, cum_qry = self.find_nearest_anchor(
            data=adata_list,
            data_qry=data_qry,
            ref=ref,
            qry=qry,
            npc=npc,
            k_weight=k_weight,
            key_correct=key_dist,
            sd=sd,
            random_state=random_state,
        )
        print("Label transfer")
        label_ref = []
        columns = []
        cat_counts = []

        if categorical_key is None:
            categorical_key = []
        if continuous_key is None:
            continuous_key = []
        if len(categorical_key) == 0 and len(continuous_key) == 0:
            raise ValueError("No categorical or continuous key specified.")

        if len(categorical_key) > 0:
            tmp = pd.concat([adata_list[i].obs[categorical_key] for i in ref], axis=0)
            enc = OneHotEncoder()
            label_ref.append(
                enc.fit_transform(tmp[categorical_key].values.astype(np.str_)).toarray()
            )
            # add categorical key to make sure col is unique
            columns += enc.categories_
            # enc.categories_ are a list of arrays, each array are categories in that categorical_key
            cat_counts += [cats.size for cats in enc.categories_]

        if len(continuous_key) > 0:
            tmp = pd.concat([adata_list[i].obs[continuous_key] for i in ref], axis=0)
            label_ref.append(tmp[continuous_key].values)
            columns += [[xx] for xx in continuous_key]
            cat_counts += [1 for _ in continuous_key]

        label_ref = np.concatenate(label_ref, axis=1)
        label_qry = np.zeros((data_qry.shape[0], label_ref.shape[1]))

        bias = label_ref[anchor[:, 0]]
        for chunk_start in np.arange(0, label_qry.shape[0], chunk_size):
            label_qry[chunk_start: (chunk_start + chunk_size)] = (
                D[chunk_start: (chunk_start + chunk_size), :, None]
                * bias[G[chunk_start: (chunk_start + chunk_size)]]
            ).sum(axis=1)

        # these column names might be duplicated
        all_column_names = np.concatenate(columns)
        all_column_variables = np.repeat(categorical_key + continuous_key, cat_counts)
        label_qry = pd.DataFrame(
            label_qry, index=data_qry_index, columns=all_column_names
        )
        result = {}
        for key in categorical_key + continuous_key:
            result[key] = label_qry.iloc[:, all_column_variables == key]
        return result

    def save(
        self,
        output_path,
        save_local_knn=False,
        save_raw_anchor=False,
        save_mutual_knn=False,
        save_adata=False,
    ):
        """Save the model and results to disk."""
        # save each adata in a separate dir
        output_path = pathlib.Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)

        if save_adata:
            # save adata and clear the self.adata_dict
            adata_dir = output_path / "adata"
            adata_dir.mkdir(exist_ok=True)
            with open(f"{adata_dir}/order.txt", "w") as f:
                for k, v in self.adata_dict.items():
                    for col, val in v.obs.items():
                        if val.dtype == "O":
                            v.obs[col] = val.fillna("nan").astype(str)
                        elif val.dtype == "category":
                            v.obs[col] = val.fillna("nan").astype(str)
                        else:
                            pass
                    v.write_h5ad(f"{adata_dir}/{k}.h5ad")
                    f.write(f"{k}\n")

        # clear the adata in integrator
        self.adata_dict = {}

        if not save_local_knn:
            self.local_knn = []
        if not save_raw_anchor:
            self.raw_anchor = {}
        if not save_mutual_knn:
            self.mutual_knn = {}

        joblib.dump(self, f"{output_path}/model.lib")
        return

    @classmethod
    def load(cls, input_path):
        """Load integrator from file."""
        adata_dir = f"{input_path}/adata"
        model_path = f"{input_path}/model.lib"

        obj = joblib.load(model_path)

        orders = pd.read_csv(f"{adata_dir}/order.txt", header=None, index_col=0).index
        adata_dict = OrderedDict()
        for k in orders:
            adata_path = f"{adata_dir}/{k}.h5ad"
            if pathlib.Path(adata_path).exists():
                adata_dict[k] = anndata.read_h5ad(f"{adata_dir}/{k}.h5ad")
        obj.adata_dict = adata_dict
        return obj

    @classmethod
    def save_transfer_results_to_adata(
        cls, adata, transfer_results, new_label_suffix="_transfer"
    ):
        """Save transfer results to adata."""
        for key, df in transfer_results.items():
            adata.obs[key + new_label_suffix] = adata.obs[key].copy()
            adata.obs.loc[df.index, key + new_label_suffix] = df.idxmax(axis=1).values
        return
