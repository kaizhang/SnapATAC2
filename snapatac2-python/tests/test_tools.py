import snapatac2 as snap

import numpy as np
import pandas as pd
from pathlib import Path
from natsort import natsorted
from collections import defaultdict
import pytest
from hypothesis import given, example, settings, HealthCheck, strategies as st
from hypothesis.extra.numpy import *

def h5ad(dir=Path("./")):
    import uuid
    dir.mkdir(exist_ok=True)
    return str(dir / Path(str(uuid.uuid4()) + ".h5ad"))


@given(
    x = arrays(integer_dtypes(endianness='='), (500, 50)),
    groups = st.lists(st.integers(min_value=0, max_value=5), min_size=500, max_size=500),
    var = st.lists(st.integers(min_value=0, max_value=100000), min_size=50, max_size=50),
)
@settings(deadline=None, suppress_health_check = [HealthCheck.function_scoped_fixture])
def test_aggregation(x, groups, var, tmp_path):
    def assert_equal(a, b):
        assert a.keys() == b.keys()
        np.testing.assert_array_equal(
            np.array(list(a.values())),
            np.array(list(b.values())),
        )

    groups = [str(g) for g in groups]
    obs_names = [str(i) for i in range(len(groups))]
    var_names = [str(i) for i in range(len(var))]
    adata = snap.AnnData(
        X=x,
        obs = dict(ident=obs_names, groups=groups),
        var = dict(ident=var_names, txt=var),
        filename = h5ad(tmp_path),
    )

    expected = defaultdict(list)
    for g, v in zip(groups, list(x)):
        expected[g].append(v)
    for k in expected.keys():
        expected[k] = np.array(expected[k]).sum(axis = 0)
    expected = dict(natsorted(expected.items()))

    np.testing.assert_array_equal(
        x.sum(axis=0),
        snap.tl.aggregate_X(adata),
    )
    np.testing.assert_array_equal(
        np.array([x.sum(axis=0)]),
        snap.tl.aggregate_X(adata, file = h5ad(tmp_path)).X[:],
    )

    assert_equal(expected, snap.tl.aggregate_X(adata, groupby=groups))
    assert_equal(expected, snap.tl.aggregate_X(adata, groupby="groups"))
    np.testing.assert_array_equal(
        np.array(list(expected.values())),
        snap.tl.aggregate_X(adata, file = h5ad(tmp_path), groupby=groups).X[:],
    )