from math import floor

import numpy as np
import pandas as pd
import pynndescent
from sklearn.metrics import adjusted_rand_score


def _purity(label1, label2):
    """Calculate purity score."""
    # contingency table
    ct = pd.crosstab(label1, label2)
    purity = ct.apply(max, axis=1).sum() / label1.size
    return purity


def calculate_purity(adata, label1, label2):
    """
    Calculate purity score of two kinds of labels for the same set of obs.

    Parameters
    ----------
    adata
        anndata object
    label1
        column name of the first label
    label2
        column name of the second label

    Returns
    -------
    purity score
    """
    p = _purity(label1=adata.obs[label1], label2=adata.obs[label2])
    try:
        adata.uns["purity"][f"{label1}-{label2}"] = p
    except KeyError:
        adata.uns["purity"] = {f"{label1}-{label2}": p}
    return p


def calculate_adjust_rand_index(adata, label1, label2):
    """
    Calculate adjusted rand index of two kinds of labels for the same set of obs.

    Parameters
    ----------
    adata
        anndata object
    label1
        column name of the first label
    label2
        column name of the second label

    Returns
    -------
    adjusted rand index
    """
    ari = adjusted_rand_score(adata.obs[label1].values, adata.obs[label2].values)
    try:
        adata.uns["ARI"][f"{label1}-{label2}"] = ari
    except KeyError:
        adata.uns["ARI"] = {f"{label1}-{label2}": ari}
    return ari


def _alignment_score(obsm_list, k=20, random_state=0, downsample_obs=None, n_jobs=-1):
    """Calculate alignment score of multiple datasets."""
    n_ds = len(obsm_list)
    if n_ds == 0:
        raise ValueError("Empty obsm_list provided.")
    if n_ds == 1:
        # only one dataset
        return 0, np.array([0])

    # downsample datasets to the same size
    min_obs = min(obsm.shape[0] for obsm in obsm_list)
    if downsample_obs is not None:
        min_obs = min(min_obs, downsample_obs)
    downsample_obsm_list = []
    for obsm in obsm_list:
        if obsm.shape[0] == min_obs:
            downsample_obsm_list.append(obsm)
        else:
            use_obs = np.sort(np.random.choice(range(obsm.shape[0]), min_obs, replace=False))
            downsample_obsm_list.append(obsm[use_obs, :].copy())
    obsm_list = downsample_obsm_list

    # determine K
    max_k = min_obs - 1
    if max_k == 0:
        # only one cell per dataset
        return 0, np.repeat(0, n_ds)

    if k is None:
        k = min(max(floor(0.01 * min_obs * n_ds), 10), max_k)
    else:
        k = min(k, max_k)

    # build KNN
    index = pynndescent.NNDescent(
        np.concatenate(obsm_list),
        metric="euclidean",
        n_neighbors=k + 1,
        random_state=random_state,
        parallel_batch_queries=True,
        n_jobs=n_jobs,
    )
    knn, _ = index.neighbor_graph

    # because each dataset has the same number of obs
    # this step turn the knn idx matrix into dataset idx matrix
    knn_dataset = knn // min_obs

    # number of knn from the same dataset
    num_same_dataset = (knn_dataset[:, [0]] == knn_dataset[:, 1:]).sum(axis=1)

    # cell alignment score,
    # smaller value means more neighbors from the same dataset
    # larger value means more neighbors from different dataset
    alignment_per_cell = 1 - (num_same_dataset - k / n_ds) / (k - k / n_ds)

    # aggregate alignment per dataset
    alignment_per_cell = pd.Series(alignment_per_cell)
    alignment_per_dataset = alignment_per_cell.groupby(alignment_per_cell.index // min_obs).mean()

    overall_alignment = alignment_per_cell.mean()
    return overall_alignment, alignment_per_dataset.values


def calculate_alignment_score(adata, dataset_col, obsm_key, downsample_obs=None, k=20, random_state=0, n_jobs=-1):
    """
    Calculate alignment score of multiple datasets.

    Parameters
    ----------
    adata
        anndata object
    dataset_col
        adata.obs column name of the dataset label
    obsm_key
        adata.obsm key of the obsm matrix
    downsample_obs
        downsample the obs of each dataset to this size
    k
        number of neighbors to use
    random_state
        random state for KNN
    n_jobs
        number of jobs to use for KNN

    Returns
    -------
    overall alignment score, alignment score per dataset
    """
    obsm_list = []
    datasets = adata.obs[dataset_col].unique().tolist()
    for dataset in datasets:
        ds_obsm = adata.obsm[obsm_key][adata.obs[dataset_col] == dataset, :]
        obsm_list.append(ds_obsm)

    overall_alignment, alignment_per_dataset = _alignment_score(
        obsm_list=obsm_list, k=k, random_state=random_state, downsample_obs=downsample_obs, n_jobs=n_jobs
    )
    alignment_per_dataset = {ds: score for ds, score in zip(datasets, alignment_per_dataset)}

    # save result to adata.uns
    adata.uns["alignment_score"] = {"overall": overall_alignment, "overall_per_dataset": alignment_per_dataset}
    return overall_alignment, pd.Series(alignment_per_dataset)


def calculate_cluster_alignment_score(
    adata, cluster_col, dataset_col, obsm_key, downsample_obs=None, k=20, random_state=0, n_jobs=-1
):
    """
    Calculate alignment score of multiple datasets for each cluster.

    Parameters
    ----------
    adata
        anndata object
    cluster_col
        adata.obs column name of the cluster label
    dataset_col
        adata.obs column name of the dataset label
    obsm_key
        adata.obsm key of the obsm matrix
    downsample_obs
        downsample the obs of each dataset to this size
    k
        number of neighbors to use
    random_state
        random state for KNN
    n_jobs
        number of jobs to use for KNN

    Returns
    -------
    overall alignment score for each cluster,
    alignment score per dataset for each cluster
    """
    cluster_overall = {}
    cluster_per_ds = {}
    for cluster, sub_df in adata.obs.groupby(cluster_col):
        obsm_list = []
        datasets = []
        for dataset, ds_df in sub_df.groupby(dataset_col):
            datasets.append(dataset)
            ds_obsm = adata[ds_df.index].obsm[obsm_key]
            obsm_list.append(ds_obsm)

        overall_alignment, alignment_per_dataset = _alignment_score(
            obsm_list=obsm_list, k=k, random_state=random_state, downsample_obs=downsample_obs, n_jobs=n_jobs
        )
        alignment_per_dataset = {ds: score for ds, score in zip(datasets, alignment_per_dataset)}
        cluster_overall[cluster] = overall_alignment
        cluster_per_ds[cluster] = alignment_per_dataset

    # save result to adata.uns
    adata.uns["alignment_score"] = {
        f"{cluster_col}_overall": cluster_overall,
        f"{cluster_col}_overall_per_dataset": cluster_per_ds,
    }

    # prepare return
    cluster_overall = pd.Series(cluster_overall)
    cluster_overall.index.name = dataset_col

    cluster_per_ds = pd.DataFrame(cluster_per_ds)
    cluster_per_ds.index.name = dataset_col
    cluster_per_ds.columns.name = cluster_col
    return cluster_overall, cluster_per_ds


def calculate_kbet_accept_rate(adata, dataset_col, obsm_key, k=20, test_size=1000, downsample_obs=5000):
    """
    Calculate the average observed accept rate of KBET test.

    Parameters
    ----------
    adata
        anndata object
    dataset_col
        adata.obs column name of the dataset label
    obsm_key
        adata.obsm key of the obsm matrix
    k
        number of neighbors to use, pass to k0 parameter of kBET
    test_size
        number of samples to use for test, pass to testSize parameter of kBET
    downsample_obs
        downsample the obs of each dataset to this size

    Returns
    -------
    average observed accept rate of kBET test,
    complete kBET test result in a dictionary
    """
    # prepare R package
    import rpy2.robjects.numpy2ri
    from rpy2.robjects.packages import importr

    rpy2.robjects.numpy2ri.activate()

    from ..clustering.rutilities import install_github_r_package

    install_github_r_package("theislab/kBET")
    kbet = importr("kBET")

    # prepare kBET input, downsample obs
    data = []
    batch = []
    datasets = []
    for i, (dataset, sub_df) in enumerate(adata.obs.groupby(dataset_col)):
        if sub_df.shape[0] > downsample_obs:
            use_obs = sub_df.sample(downsample_obs, replace=False).index
        else:
            use_obs = sub_df.index
        data.append(adata.obsm[obsm_key][adata.obs_names.isin(use_obs), :])
        batch.append(np.repeat(i, use_obs.size))
        datasets.append(dataset)
    data = np.concatenate(data)
    batch = np.concatenate(batch)

    # run kBET R function
    test_size = min(test_size, int(data.shape[0] * 0.1))
    test_size = max(test_size, 25)
    k = min(k, test_size)
    summary, results, ave_pval, stats, params, outsider = kbet.kBET(
        data,
        batch,
        plot=False,
        k0=k,
        # knn = NULL,
        testSize=test_size,
        do_pca=False,
        # dim.pca = 50,
        # heuristic = True,
        # n_repeat=100,
        # alpha=0.05,
        # addTest = FALSE,
        verbose=True,
        # adapt=TRUE
    )

    # reformat kBET results
    summary = pd.DataFrame(summary)
    results = pd.DataFrame(results)
    stats = {k: np.array(v) for k, v in stats.items()}
    params = {k: np.array(v) for k, v in params.items()}
    outsider = {k: np.array(v) for k, v in outsider.items()}
    kbet_result = {
        "summary": summary,
        "results": results,
        "ave_pval": ave_pval,
        "stats": stats,
        "params": params,
        "outsider": outsider,
    }

    # calculate average observed accept rate
    overall_accept_rate = 1 - stats["kBET.observed"].mean()
    adata.uns["kbet"] = {"overall_accept_rate": overall_accept_rate, "dataset_col": dataset_col, "obsm_key": obsm_key}
    return overall_accept_rate, kbet_result
