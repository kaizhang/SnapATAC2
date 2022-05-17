import numpy as np
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

def diff_analysis(mat, z, peaks=None, covariate=None):
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
    
    depth_per_cell = np.log(np.sum(mat, axis=1))
    if covariate is None:
        X = depth_per_cell
    else:
        X = np.concatenate((depth_per_cell, covariate), axis=1)

    if peaks is None:
        mat = mat.tocsc()
    else:
        mat = mat.tocsc()[:, peaks]

    return _diff_test_helper(np.asarray(X), np.asarray(z), mat)


def _diff_test_helper(X, z, Y):
    """
    Parameters
    ----------
    X
        (n_sample, n_feature),
    z
        (n_sample, 1), additional variables
    Y
        (n_sample, k), labels
    """
    from tqdm import tqdm
 
    X0 = X
    X1 = np.concatenate((X, z), axis=1)

    _, n = Y.shape
    Y.data = np.ones(Y.data.shape)

    result = []
    for i in tqdm(range(n)):
        result.append(
            likelihood_ratio_test(X0, X1, np.asarray(np.ravel(Y[:, i].todense())))
        )
    return result

def likelihood_ratio_test(X0, X1, y) -> float:
    """
    Comparing null model with alternative model using the likehood ratio test.

    Parameters
    ----------
    X0
        (n_sample, n_feature), variables used in null model
    X1
        (n_sample, n_feature2), variables used in alternative model
    Y
        (n_sample, ), labels
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