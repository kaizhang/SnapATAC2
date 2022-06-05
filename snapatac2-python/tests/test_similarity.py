import snapatac2.tools._spectral as sp

import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse, random
from hypothesis import reproduce_failure, given, example, settings, HealthCheck, strategies as st
from hypothesis.extra.numpy import *
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import jaccard, cosine

def cosine_similarity(x, y, w = None):
    def dist(v1, v2):
        if np.amax(v1) == 0 or np.amax(v2) == 0:
            return 0
        else:
            return 1 - cosine(v1, v2, w)
    return pairwise_distances(x, y, metric = dist)

@given(
    mat1 =arrays(bool, (17, 100)),
    mat2 =arrays(bool, (19, 100)),
    w = arrays(
        np.float64, (100,),
        elements = {"allow_nan": False, "allow_infinity": False, "min_value": 1, "max_value": 1000},
    )
)
@settings(deadline = None, suppress_health_check = [HealthCheck.function_scoped_fixture])
def test_jaccard(mat1, mat2, w):
    csr1 = csr_matrix(mat1)
    csr2 = csr_matrix(mat2)

    np.testing.assert_array_equal(
        sp.jaccard_similarity(csr1),
        sp.jaccard_similarity(csr1, csr1),
    )

    np.testing.assert_array_almost_equal(
        pairwise_distances(mat1, mat1, metric= lambda x, y: 1 - jaccard(x, y)),
        sp.jaccard_similarity(csr1),
        decimal=12,
    )
    np.testing.assert_array_almost_equal(
        pairwise_distances(mat1, mat1, metric= lambda x, y: 1 - jaccard(x, y, w)),
        sp.jaccard_similarity(csr1, None, w),
        decimal=12,
    )

    np.testing.assert_array_almost_equal(
        pairwise_distances(mat1, mat2, metric= lambda x, y: 1 - jaccard(x, y)),
        sp.jaccard_similarity(csr1, csr2),
        decimal=12,
    )
    np.testing.assert_array_almost_equal(
        pairwise_distances(mat1, mat2, metric= lambda x, y: 1 - jaccard(x, y, w)),
        sp.jaccard_similarity(csr1, csr2, w),
        decimal=12,
    )

@given(
    mat1 = arrays(
        np.float64, (15, 100),
        elements = {"allow_nan": False, "allow_infinity": False, "min_value": 1, "max_value": 10000000},
    ),
    mat2 = arrays(
        np.float64, (17, 100),
        elements = {"allow_nan": False, "allow_infinity": False, "min_value": 1, "max_value": 10000000},
    ),
    w = arrays(
        np.float64, (100,),
        elements = {"allow_nan": False, "allow_infinity": False, "min_value": 1, "max_value": 1000},
    )
)
@settings(deadline = None, suppress_health_check = [HealthCheck.function_scoped_fixture])
def test_cosine(mat1, mat2, w):
    csr1 = csr_matrix(mat1)
    csr2 = csr_matrix(mat2)

    np.testing.assert_array_almost_equal(
        sp.cosine_similarity(csr1),
        sp.cosine_similarity(csr1, csr1),
    )

    np.testing.assert_array_almost_equal(
        cosine_similarity(mat1, mat1),
        sp.cosine_similarity(csr1),
        decimal=12,
    )
    np.testing.assert_array_almost_equal(
        cosine_similarity(mat1, mat1, w),
        sp.cosine_similarity(csr1, None, w),
        decimal=12,
    )

    np.testing.assert_array_almost_equal(
        cosine_similarity(mat1, mat2),
        sp.cosine_similarity(csr1, csr2),
        decimal=12,
    )
    np.testing.assert_array_almost_equal(
        cosine_similarity(mat1, mat2, w),
        sp.cosine_similarity(csr1, csr2, w),
        decimal=12,
    )