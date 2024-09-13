import numpy as np
from scipy.sparse import issparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import safe_sparse_dot

def tf_idf(data, scale_factor=100000, idf=None):
    sparse_input = issparse(data)

    if idf is None:
        # add small value in case down sample creates empty feature
        _col_sum = data.sum(axis=0)
        if sparse_input:
            col_sum = _col_sum.A1.astype(np.float32) + 0.00001
        else:
            col_sum = _col_sum.ravel().astype(np.float32) + 0.00001
        idf = np.log(1 + data.shape[0] / col_sum).astype(np.float32)
    else:
        idf = idf.astype(np.float32)

    _row_sum = data.sum(axis=1)
    if sparse_input:
        row_sum = _row_sum.A1.astype(np.float32) + 0.00001
    else:
        row_sum = _row_sum.ravel().astype(np.float32) + 0.00001

    tf = data.astype(np.float32)

    if sparse_input:
        tf.data = tf.data / np.repeat(row_sum, tf.getnnz(axis=1))
        tf.data = np.log1p(np.multiply(tf.data, scale_factor, dtype="float32"))
        tf = tf.multiply(idf)
    else:
        tf = tf / row_sum[:, np.newaxis]
        tf = np.log1p(np.multiply(tf, scale_factor, dtype="float32"))
        tf = tf * idf
    return tf, idf


def top_features_idx(data, n_features):
    """
    Select top features with the highest importance in CCs.

    Parameters
    ----------
    data
        data.shape = (n_cc, total_features)
    n_features
        number of features to select

    Returns
    -------
    features_idx : np.array
    """
    # data.shape = (n_cc, total_features)
    n_cc = data.shape[0]
    n_features_per_dim = n_features * 10 // n_cc
    n_features_per_dim = min(n_features_per_dim, data.shape[1] - 1)

    sample_range = np.arange(n_cc)[:, None]

    # get idx of n_features_per_dim features with the highest absolute loadings
    data = np.abs(data)
    idx = np.argpartition(-data, n_features_per_dim, axis=1)[:, :n_features_per_dim]
    # idx.shape = (n_cc, n_features_per_dim)

    # make sure the order of first n_features_per_dim is ordered by loadings
    idx = idx[sample_range, np.argsort(-data[sample_range, idx], axis=1)]

    for i in range(n_features // n_cc + 1, n_features_per_dim):
        features_idx = np.unique(idx[:, :i].flatten())
        if len(features_idx) > n_features:
            return features_idx
    else:
        features_idx = np.unique(idx[:, :n_features_per_dim].flatten())
        return features_idx


def cca(
    data1,
    data2,
    scale1=True,
    scale2=True,
    n_components=50,
    max_cc_cell=20000,
    chunk_size=50000,
    random_state=0,
    svd_algorithm="randomized",
    k_filter=None,
    n_features=200,
):
    np.random.seed(random_state)
    tf_data1, tf_data2, scaler1, scaler2 = downsample(
        data1=data1,
        data2=data2,
        todense=True,
        scale1=scale1,
        scale2=scale2,
        max_cc_cell=max_cc_cell,
        random_state=random_state,
    )

    # CCA decomposition
    model = TruncatedSVD(n_components=n_components, algorithm=svd_algorithm, random_state=random_state)
    tf_data2_t = tf_data2.T.copy()
    ## TODO: this can be optimized to reduce memory.
    U = model.fit_transform(tf_data1.dot(tf_data2_t))

    # select dimensions with non-zero singular values
    sel_dim = model.singular_values_ != 0
    print("non zero dims", sel_dim.sum())

    V = model.components_[sel_dim].T
    U = U[:, sel_dim] / model.singular_values_[sel_dim]

    # compute ccv feature loading
    if k_filter:
        high_dim_feature = top_features_idx(
            np.concatenate([U, V], axis=0).T.dot(np.concatenate([tf_data1, tf_data2], axis=0)), n_features=n_features
        )
    else:
        high_dim_feature = None

    # transform CC
    if data2.shape[0] > max_cc_cell:
        V = []
        for chunk_start in np.arange(0, data2.shape[0], chunk_size):
            if issparse(data2):
                tmp = data2[chunk_start : (chunk_start + chunk_size)].toarray()
            else:
                tmp = data2[chunk_start : (chunk_start + chunk_size)]
            if scale2:
                tmp = scaler2.transform(tmp)
            V.append(np.dot(np.dot(U.T, tf_data1), tmp.T).T)
        V = np.concatenate(V, axis=0)
        V = V / model.singular_values_[sel_dim]

    if data1.shape[0] > max_cc_cell:
        U = []
        for chunk_start in np.arange(0, data1.shape[0], chunk_size):
            if issparse(data1):
                tmp = data1[chunk_start : (chunk_start + chunk_size)].toarray()
            else:
                tmp = data1[chunk_start : (chunk_start + chunk_size)]
            if scale1:
                tmp = scaler1.transform(tmp)
            U.append(np.dot(tmp, np.dot(model.components_[sel_dim], tf_data2).T))
        U = np.concatenate(U, axis=0)
        U = U / model.singular_values_[sel_dim]

    return U, V, high_dim_feature


def adata_cca(adata, group_col, separate_scale=True, n_components=50, random_state=42):
    groups = adata.obs[group_col].unique()
    if len(groups) != 2:
        raise ValueError(f"CCA only handle 2 groups, " f"adata.obs[{group_col}] has {len(groups)} different groups.")
    group_a, group_b = groups
    a = adata[adata.obs[group_col] == group_a, :].X
    b = adata[adata.obs[group_col] == group_b, :].X

    pc, loading, _ = cca(
        data1=a,
        data2=b,
        scale1=separate_scale,
        scale2=separate_scale,
        n_components=n_components,
        random_state=random_state,
    )
    total_cc = np.concatenate([pc, loading], axis=0)
    adata.obsm["X_cca"] = total_cc
    return


# def incremental_cca(a, b, max_chunk_size=10000, random_state=0):
#     """
#     Perform Incremental CCA by chunk dot product and IncrementalPCA
#
#     Parameters
#     ----------
#     a
#         dask.Array of dataset a
#     b
#         dask.Array of dataset b
#     max_chunk_size
#         Chunk size for Incremental fit and transform, the larger the better as long as MEM is enough
#     random_state
#
#     Returns
#     -------
#     Top CCA components
#     """
#     raise NotImplementedError
#     # TODO PC is wrong
#     pca = dIPCA(n_components=50,
#                 whiten=False,
#                 copy=True,
#                 batch_size=None,
#                 svd_solver='auto',
#                 iterated_power=0,
#                 random_state=random_state)
#
#     # partial fit
#     n_sample = a.shape[0]
#     n_chunks = n_sample // max_chunk_size + 1
#     chunk_size = int(n_sample / n_chunks) + 1
#     for chunk_start in range(0, n_sample, chunk_size):
#         print(chunk_start)
#         X_chunk = a[chunk_start:chunk_start + chunk_size, :].dot(b.T)
#         pca.partial_fit(X_chunk)
#
#     # transform
#     pcs = []
#     for chunk_start in range(0, n_sample, chunk_size):
#         print(chunk_start)
#         X_chunk = a[chunk_start:chunk_start + chunk_size, :].dot(b.T)
#         pc_chunk = pca.transform(X_chunk).compute()
#         pcs.append(pc_chunk)
#     pcs = np.concatenate(pcs)
#
#     # concatenate CCA
#     total_cc = np.concatenate([pcs, pca.components_.T])
#     return total_cc


def lsi_cca(
    data1,
    data2,
    scale_factor=100000,
    n_components=50,
    max_cc_cell=20000,
    chunk_size=50000,
    svd_algorithm="randomized",
    min_cov_filter=5,
    random_state=0,
):
    np.random.seed(random_state)

    # down sample data1 and data2 to run tf_idf and CCA
    if max_cc_cell < data1.shape[0]:
        sel1 = np.sort(np.random.choice(np.arange(data1.shape[0]), max_cc_cell, False))
        tf_data1 = data1[sel1, :]
    else:
        tf_data1 = data1
    if max_cc_cell < data2.shape[0]:
        sel2 = np.sort(np.random.choice(np.arange(data2.shape[0]), max_cc_cell, False))
        tf_data2 = data2[sel2, :]
    else:
        tf_data2 = data2

    # filter bin to make sure the min_cov_filter is satisfied
    col_sum1 = tf_data1.sum(axis=0).A1
    col_sum2 = tf_data2.sum(axis=0).A1
    # the same bin_filter will also be used
    # in the chunk transfer below
    bin_filter = np.logical_and(col_sum1 > min_cov_filter, col_sum2 > min_cov_filter)
    tf1, idf1 = tf_idf(tf_data1[:, bin_filter], scale_factor=scale_factor)
    tf2, idf2 = tf_idf(tf_data2[:, bin_filter], scale_factor=scale_factor)

    # CCA part
    model = TruncatedSVD(n_components=n_components, algorithm=svd_algorithm, random_state=0)
    tf = tf1.dot(tf2.T)
    U = model.fit_transform(tf)

    # select non-zero singular values
    # transform the whole dataset 2 to get V
    sel_dim = model.singular_values_ != 0
    nnz_singular_values = model.singular_values_[sel_dim]
    nnz_components = model.components_[sel_dim]
    if max_cc_cell > data2.shape[0]:
        V = nnz_components.T
    else:
        # use the safe_sparse_dot to avoid memory error
        # safe_sparse_dot take both sparse and dense matrix,
        # for dense matrix, it just uses normal numpy dot product
        V = np.concatenate(
            [
                safe_sparse_dot(
                    safe_sparse_dot(U.T[sel_dim], tf1),
                    tf_idf(
                        data2[chunk_start : (chunk_start + chunk_size)][:, bin_filter],
                        scale_factor=scale_factor,
                        idf=idf2,
                    )[
                        0
                    ].T,  # [0] is the tf
                ).T
                for chunk_start in np.arange(0, data2.shape[0], chunk_size)
            ],
            axis=0,
        )
        V = V / np.square(nnz_singular_values)

    # transform the whole dataset 1 to get U
    if max_cc_cell > data1.shape[0]:
        U = U[:, sel_dim] / nnz_singular_values
    else:
        U = np.concatenate(
            [
                safe_sparse_dot(
                    tf_idf(
                        data1[chunk_start : (chunk_start + chunk_size)][:, bin_filter],
                        scale_factor=scale_factor,
                        idf=idf1,
                    )[
                        0
                    ],  # [0] is the tf
                    safe_sparse_dot(nnz_components, tf2).T,
                )
                for chunk_start in np.arange(0, data1.shape[0], chunk_size)
            ],
            axis=0,
        )
        U = U / nnz_singular_values
    return U, V


class LSI:
    def __init__(
        self,
        scale_factor=100000,
        n_components=100,
        algorithm="arpack",
        random_state=0,
        idf=None,
        model=None,
    ):
        self.scale_factor = scale_factor
        if idf is not None:
            self.idf = idf.copy()
        if idf is not None:
            self.model = model
        else:
            self.model = TruncatedSVD(n_components=n_components, algorithm=algorithm, random_state=random_state)

    def fit(self, data):
        tf, idf = tf_idf(data, self.scale_factor)
        self.idf = idf.copy()
        n_rows, n_cols = tf.shape
        self.model.n_components = min(n_rows, n_cols, self.model.n_components)
        self.model.fit(tf)
        return self

    def fit_transform(self, data):
        tf, idf = tf_idf(data, self.scale_factor)
        self.idf = idf.copy()
        n_rows, n_cols = tf.shape
        self.model.n_components = min(n_rows, n_cols, self.model.n_components)
        tf_reduce = self.model.fit_transform(tf)
        return tf_reduce / self.model.singular_values_

    def transform(self, data, chunk_size=50000, scaler=None):
        tf_reduce = []
        for chunk_start in np.arange(0, data.shape[0], chunk_size):
            tf, _ = tf_idf(data[chunk_start : (chunk_start + chunk_size)], self.scale_factor, self.idf)
            tf_reduce.append(self.model.transform(tf))
        return np.concatenate(tf_reduce, axis=0) / self.model.singular_values_


class SVD:
    def __init__(
        self,
        n_components=100,
        algorithm="randomized",
        random_state=0,
    ):
        self.model = TruncatedSVD(n_components=n_components, algorithm=algorithm, random_state=random_state)

    def fit(self, data):
        self.model.fit(data)
        return self

    def fit_transform(self, data):
        return self.model.fit_transform(data)

    def transform(self, data, chunk_size=50000, scaler=None):
        tf_reduce = []
        for chunk_start in np.arange(0, data.shape[0], chunk_size):
            if issparse(data):
                tmp = data[chunk_start : (chunk_start + chunk_size)].toarray()
            else:
                tmp = data[chunk_start : (chunk_start + chunk_size)]
            if scaler:
                tmp = scaler.transform(tmp)
            tf_reduce.append(self.model.transform(tmp))
        return np.concatenate(tf_reduce, axis=0)


def downsample(data1, data2, scale1, scale2, todense, max_cc_cell=20000, random_state=0):
    scaler1, scaler2 = [None, None]
    np.random.seed(random_state)
    if data1.shape[0] > max_cc_cell:
        sel1 = np.random.choice(np.arange(data1.shape[0]), min(max_cc_cell, data1.shape[0]), False)
        tf1 = data1[sel1]
    else:
        tf1 = data1.copy()
    if todense:
        if issparse(tf1):
            tf1 = tf1.toarray()

    if data2.shape[0] > max_cc_cell:
        sel2 = np.random.choice(np.arange(data2.shape[0]), min(max_cc_cell, data2.shape[0]), False)
        tf2 = data2[sel2]
    else:
        tf2 = data2.copy()
    if todense:
        if issparse(tf2):
            tf2 = tf2.toarray()

    if scale1:
        scaler1 = StandardScaler()
        tf1 = scaler1.fit_transform(tf1)
    if scale2:
        scaler2 = StandardScaler()
        tf2 = scaler2.fit_transform(tf2)
    return tf1, tf2, scaler1, scaler2
