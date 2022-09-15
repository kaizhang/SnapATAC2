from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import sys

from snapatac2._snapatac2 import AnnData, AnnDataSet

def diff_test(
    data: AnnData | AnnDataSet,
    cell_group1: list[int] | list[str] | 'np.ndarray[bool]',
    cell_group2: list[int] | list[str] | 'np.ndarray[bool]',
    features : list[str] | list[int] | 'np.ndarray[bool]' | None = None,
    covariates: list[str] | None = None,
    min_log_fc: float = 0.25,
    min_pct: float = 0.05,
) -> pd.DataFrame:
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
    min_log_fc
        Limit testing to features which show, on average, at least
        X-fold difference (log2-scale) between the two groups of cells.
    min_pct
        Only test features that are detected in a minimum fraction of min_pct
        cells in either of the two populations. 
    """
    def to_indices(xs, n, type):
        if isinstance(xs, np.ndarray):
            if xs.dtype == 'bool':
                if xs.shape != (n, ):
                    raise NameError("the length of boolean mask must be the same as the number of cells")
                else:
                    return np.where(xs)[0].tolist()
            else:
                xs = xs.tolist()

        if isinstance(xs, list):
            if all([isinstance(item, int) for item in xs]):
                return xs
            elif all([isinstance(item, str) for item in xs]):
                if type == "obs":
                    return data.obs_ix(xs)
                else:
                    return data.var_ix(xs)
            else:
                raise NameError("invalid type")
        else:
            raise NameError("invalid type")
    cell_group1 = to_indices(cell_group1, data.n_obs, "obs")
    n_group1 = len(cell_group1)
    cell_group2 = to_indices(cell_group2, data.n_obs, "obs")
    n_group2 = len(cell_group2)

    cell_by_peak = data.X[cell_group1 + cell_group2, :].tocsc()
    test_var = np.array([0] * n_group1 + [1] * n_group2)
    if covariates is not None:
        raise NameError("covariates is not implemented")

    features = range(data.n_vars) if features is None else to_indices(features, data.n_vars, "var")
    print("Input has {} features ...".format(len(features)), file=sys.stderr)
    print("Filtering features ...", file=sys.stderr)
    features, log_fc = zip(*_filter_features(
        cell_by_peak[:n_group1, :],
        cell_by_peak[n_group1:, :],
        features,
        min_pct,
        min_log_fc,
    ))

    print("Testing {} features ...".format(len(features)), file=sys.stderr)
    pvals = _diff_test_helper(cell_by_peak, test_var, features, covariates)
    return pd.DataFrame({
        "feature_id": [data.var_names[i] for i in features],
        "log2(fold_change)": log_fc,
        "p-value": pvals,
    })

def _filter_features(mat1, mat2, peak_indices, min_pct, min_log_fc, pseudo_count = 1):
    def rpm(m):
        x = np.ravel(np.sum(m, axis = 0)) + pseudo_count
        s = x.sum()
        return x / (s / 1000000)

    def pass_min_pct(i):
        cond1 = mat1[:, i].count_nonzero() / mat1.shape[0] >= min_pct 
        cond2 = mat2[:, i].count_nonzero() / mat2.shape[0] >= min_pct 
        return cond1 or cond2

    log_fc = np.log2(rpm(mat1) / rpm(mat2))
    peak_indices = [i for i in peak_indices if pass_min_pct(i)]
    return [(i, log_fc[i])  for i in peak_indices if abs(log_fc[i]) >= min_log_fc]

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
        X = np.log(np.sum(mat, axis=1))
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

    model = LogisticRegression(penalty="none", random_state=0, n_jobs=1,
        solver="lbfgs", multi_class='ovr', warm_start=False,
        max_iter = 1000,
        ).fit(X0, y)
    reduced = -log_loss(y, model.predict_proba(X0), normalize=False)

    model = LogisticRegression(penalty="none", random_state=0, n_jobs=1,
        solver="lbfgs", multi_class='ovr', warm_start=False,
        max_iter = 1000,
        ).fit(X1, y)
    full = -log_loss(y, model.predict_proba(X1), normalize=False)
    chi = -2 * (reduced - full)
    return chi2.sf(chi, X1.shape[1] - X0.shape[1])