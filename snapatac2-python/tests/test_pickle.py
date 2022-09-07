import snapatac2 as snap

import pickle
import pytest
from hypothesis import given, settings, HealthCheck, strategies as st
from hypothesis.extra.numpy import *

@given(
    d = st.integers(min_value=0, max_value=1000000),
    sc = st.decimals(allow_nan=False, allow_infinity=False),
)
@settings(deadline=None, suppress_health_check = [HealthCheck.function_scoped_fixture])
def test_pickle_edge(d, sc):
    link = snap.LinkData()
    link.distance = d
    link.regr_score = sc
    assert link == pickle.loads(pickle.dumps(link))

@given(
    id = st.text(),
    type = st.text(),
)
@settings(deadline=None, suppress_health_check = [HealthCheck.function_scoped_fixture])
def test_pickle_node(id, type):
    node = snap.NodeData(id)
    node.type = type
    assert node == pickle.loads(pickle.dumps(node))