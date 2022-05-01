import snapatac2.tools._spectral as sp

import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse, random
from hypothesis import given, example, settings, HealthCheck, strategies as st
from hypothesis.extra.numpy import *
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jaccard

@given(
    mat1 =arrays(bool, (57, 100)),
    mat2 =arrays(bool, (47, 100)),
)
@settings(suppress_health_check = [HealthCheck.function_scoped_fixture])
def test_jaccard(mat1, mat2):
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
        pairwise_distances(mat1, mat2, metric= lambda x, y: 1 - jaccard(x, y)),
        sp.jaccard_similarity(csr1, csr2),
        decimal=12,
    )

@given(
    mat1 = arrays(
        np.float64, (57, 100),
        elements = {"allow_nan": False, "allow_infinity": False, "min_value": -100000000, "max_value": 10000000},
    ),
    mat2 = arrays(
        np.float64, (45, 100),
        elements = {"allow_nan": False, "allow_infinity": False, "min_value": -100000000, "max_value": 10000000},
    ),
)
@settings(suppress_health_check = [HealthCheck.function_scoped_fixture])
def test_cosine(mat1, mat2):
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
        cosine_similarity(mat1, mat2),
        sp.cosine_similarity(csr1, csr2),
        decimal=12,
    )