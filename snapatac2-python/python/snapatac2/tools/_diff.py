from __future__ import annotations

from typing import Literal
import numpy as np
from scipy.stats import chi2, norm, zscore
import logging

from snapatac2._snapatac2 import AnnData, AnnDataSet
from snapatac2.tools._misc import aggregate_X

def marker_regions(
    data: AnnData | AnnDataSet,
    groupby: str | list[str],
    pvalue: float = 0.01,
) -> dict[str, list[str]]:
    """
    A quick-and-dirty way to get marker regions.

    Parameters
    ----------
    data
        AnnData or AnnDataSet object.
    groupby
        Grouping variable.
    pvalue
        P-value threshold.
    """
    count = aggregate_X(data, groupby, normalize="RPKM")
    z = zscore(np.log2(1 + count.X), axis = 0)
    peaks = {}
    for i in range(z.shape[0]):
        select = norm.sf(z[i, :]) < pvalue
        if np.where(select)[0].size >= 1:
            peaks[count.obs_names[i]] = count.var_names[select]
    return peaks

def mad(data, axis=None):
    """ Compute Median Absolute Deviation """
    return np.median(np.absolute(data - np.median(data, axis)), axis)

def modified_zscore(matrix, axis=0):
    """ Compute Modified Z-score for a matrix along specified axis """
    median = np.median(matrix, axis=axis)
    median_absolute_deviation = mad(matrix, axis=axis)
    min_non_zero = np.min(median_absolute_deviation[median_absolute_deviation > 0])
    median_absolute_deviation[median_absolute_deviation == 0] = min_non_zero

    if axis == 0:
        modified_z_scores = 0.6745 * (matrix - median) / median_absolute_deviation
    elif axis == 1:
        modified_z_scores = 0.6745 * (matrix.T - median).T / median_absolute_deviation
    else:
        raise ValueError("Invalid axis, it should be 0 or 1")

    return modified_z_scores

def diff_test(
    data: AnnData | AnnDataSet,
    cell_group1: list[int] | list[str],
    cell_group2: list[int] | list[str],
    features : list[str] | list[int] | None = None,
    covariates: list[str] | None = None,
    direction: Literal["positive", "negative", "both"] = "both",
    min_log_fc: float = 0.25,
    min_pct: float = 0.05,
) -> 'polars.DataFrame':
    """
    Identify differentially accessible regions.

    Parameters
    ----------
    data
        AnnData or AnnDataSet object.
    cell_group1
        cells belonging to group 1. This can be a list of cell barcodes, indices or 
        boolean mask vector.
    cell_group2
        cells belonging to group 2. This can be a list of cell barcodes, indices or 
        boolean mask vector.
    features
        Features/peaks to test. If None, all features are tested.
    covariates
    direction
        "positive", "negative", or "both".
        "positive": return features that are enriched in group 1.
        "negative": return features that are enriched in group 2.
        "both": return features that are enriched in group 1 or group 2.
    min_log_fc
        Limit testing to features which show, on average, at least
        X-fold difference (log2-scale) between the two groups of cells.
    min_pct
        Only test features that are detected in a minimum fraction of min_pct
        cells in either of the two populations. 

    Returns
    -------
    pl.DataFrame
        A DataFrame with 4 columns: "feature name", "log2(fold_change)",
        "p-value", and "adjusted p-value".
    """
    import polars as pl

    def to_indices(xs, type):
        xs = [_convert_to_bool_if_np_bool(x) for x in xs]
        if all(isinstance(x, bool) for x in xs):
            return [i for i, value in enumerate(xs) if value]
        elif all([isinstance(item, str) for item in xs]):
            if type == "obs":
                if data.isbacked:
                    return data.obs_ix(xs)
                else:
                    return [data.obs_names.get_loc(x) for x in xs]
            else:
                if data.isbacked:
                    return data.var_ix(xs)
                else:
                    return [data.var_names.get_loc(x) for x in xs]
        else:
            return xs

    cell_group1 = to_indices(cell_group1, "obs")
    n_group1 = len(cell_group1)
    cell_group2 = to_indices(cell_group2, "obs")
    n_group2 = len(cell_group2)

    cell_by_peak = data.X[cell_group1 + cell_group2, :].tocsc()
    test_var = np.array([0] * n_group1 + [1] * n_group2)
    if covariates is not None:
        raise NameError("covariates is not implemented")

    features = range(data.n_vars) if features is None else to_indices(features, "var")
    logging.info("Input contains {} features, now perform filtering with 'min_log_fc = {}' and 'min_pct = {}' ...".format(len(features), min_log_fc, min_pct))
    filtered = _filter_features(
        cell_by_peak[:n_group1, :],
        cell_by_peak[n_group1:, :],
        features,
        direction,
        min_pct,
        min_log_fc,
    )

    if len(filtered) == 0:
        logging.warning("Zero feature left after filtering, perhaps 'min_log_fc' or 'min_pct' is too large")
        return pl.DataFrame()
    else:
        features, log_fc = zip(*filtered)
        logging.info("Testing {} features ...".format(len(features)))
        pvals = _diff_test_helper(cell_by_peak, test_var, features, covariates)
        var_names = data.var_names
        return pl.DataFrame({
            "feature name": [var_names[i] for i in features],
            "log2(fold_change)": np.array(log_fc),
            "p-value": np.array(pvals),
            "adjusted p-value": _p_adjust_bh(pvals),
        }).sort("adjusted p-value")

def _p_adjust_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]

def _filter_features(mat1, mat2, peak_indices, direction,
    min_pct, min_log_fc, pseudo_count = 1,
):
    def rpm(m):
        x = np.ravel(np.sum(m, axis = 0)) + pseudo_count
        s = x.sum()
        return x / (s / 1000000)

    def pass_min_pct(i):
        cond1 = mat1[:, i].count_nonzero() / mat1.shape[0] >= min_pct 
        cond2 = mat2[:, i].count_nonzero() / mat2.shape[0] >= min_pct 
        return cond1 or cond2

    def adjust_sign(fc):
        if direction == "both":
            return abs(fc)
        elif direction == "positive":
            return fc
        elif direction == "negative":
            return -fc
        else:
            raise NameError("direction must be 'positive', 'negative' or 'both'")

    log_fc = np.log2(rpm(mat1) / rpm(mat2))
    peak_indices = [i for i in peak_indices if pass_min_pct(i)]
    return [(i, log_fc[i])  for i in peak_indices if adjust_sign(log_fc[i]) >= min_log_fc]

def _diff_test_helper(mat, z, peaks=None, covariate=None) -> list[float]:
    """
    Parameters
    ----------
    mat
        cell by peak matrix.
    z
        variables to test
    peaks
        peak indices
    covariate 
        additional variables to regress out.
    """

    if len(z.shape) == 1:
        z = z.reshape((-1, 1))
    
    if covariate is None:
        X = np.log1p(np.sum(mat, axis=1))
    else:
        X = covariate

    mat = mat.tocsc()
    if peaks is not None:
        mat = mat[:, peaks]

    return _likelihood_ratio_test_many(np.asarray(X), np.asarray(z), mat)


def _likelihood_ratio_test_many(X, z, Y) -> list[float]:
    """
    Parameters
    ----------
    X
        (n_sample, n_feature).
    z
        (n_sample, 1), the additional variable.
    Y
        (n_sample, k), labels
    
    Returns
    -------
    P-values of whether adding z to the models improves the prediction.
    """
    from tqdm import tqdm
 
    X0 = X
    X1 = np.concatenate((X, z), axis=1)

    _, n = Y.shape
    Y.data = np.ones(Y.data.shape)

    result = []
    for i in tqdm(range(n)):
        result.append(
            _likelihood_ratio_test(X0, X1, np.asarray(np.ravel(Y[:, i].todense())))
        )
    return result

def _likelihood_ratio_test(
    X0: np.ndarray,
    X1: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Comparing null model with alternative model using the likehood ratio test.

    Parameters
    ----------
    X0
        (n_sample, n_feature), variables used in null model.
    X1
        (n_sample, n_feature2), variables used in alternative model.
        Note X1 contains X0.
    Y
        (n_sample, ), labels.

    Returns
    -------
    The P-value.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    model = LogisticRegression(penalty=None, random_state=0, n_jobs=1,
        solver="lbfgs", warm_start=False,
        max_iter = 1000,
        ).fit(X0, y)
    reduced = -log_loss(y, model.predict_proba(X0), normalize=False)

    model = LogisticRegression(penalty=None, random_state=0, n_jobs=1,
        solver="lbfgs", warm_start=False,
        max_iter = 1000,
        ).fit(X1, y)
    full = -log_loss(y, model.predict_proba(X1), normalize=False)
    chi = -2 * (reduced - full)
    return chi2.sf(chi, X1.shape[1] - X0.shape[1])

def _convert_to_bool_if_np_bool(value):
    if isinstance(value, np.bool_):
        return bool(value)
    return value