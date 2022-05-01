from snapatac2.tools._spectral import jaccard_similarity

import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse, random
from hypothesis import given, example, settings, HealthCheck, strategies as st
from hypothesis.extra.numpy import *
from sklearn.metrics import jaccard_score, pairwise_distances
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
        jaccard_similarity(csr1),
        jaccard_similarity(csr1, csr1),
    )

    np.testing.assert_array_almost_equal(
        pairwise_distances(mat1, mat1, metric= lambda x, y: 1 - jaccard(x, y)),
        jaccard_similarity(csr1),
        decimal=12,
    )

    np.testing.assert_array_almost_equal(
        pairwise_distances(mat1, mat2, metric= lambda x, y: 1 - jaccard(x, y)),
        jaccard_similarity(csr1, csr2),
        decimal=12,
    )