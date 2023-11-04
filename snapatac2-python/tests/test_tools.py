import snapatac2 as snap

import numpy as np
import anndata as ad
import pandas as pd
from pathlib import Path
from natsort import natsorted
from collections import defaultdict
import pytest
from hypothesis import given, settings, HealthCheck, strategies as st
from hypothesis.extra.numpy import *
from scipy.sparse import csr_matrix
import gzip

from distutils import dir_util
from pytest import fixture
import os

@fixture
def datadir(tmpdir, request):
    '''
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    '''
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir

def h5ad(dir=Path("./")):
    import uuid
    dir.mkdir(exist_ok=True)
    return str(dir / Path(str(uuid.uuid4()) + ".h5ad"))


@given(
    x = arrays(integer_dtypes(endianness='='), (500, 50)),
    groups = st.lists(st.integers(min_value=0, max_value=5), min_size=500, max_size=500),
    var = st.lists(st.integers(min_value=0, max_value=100000), min_size=50, max_size=50),
)
@settings(max_examples=10, deadline=None, suppress_health_check = [HealthCheck.function_scoped_fixture])
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
        expected[k] = np.array(expected[k], dtype="float64").sum(axis = 0)
    expected = dict(natsorted(expected.items()))

    np.testing.assert_array_equal(
        x.sum(axis=0),
        snap.tl.aggregate_X(adata),
    )
    np.testing.assert_array_equal(
        np.array(list(expected.values())),
        snap.tl.aggregate_X(adata, file = h5ad(tmp_path), groupby=groups).X[:],
    )


def test_make_fragment(datadir, tmp_path):
    bam = str(datadir.join('test.bam'))
    bed = str(datadir.join('test.bed.gz'))
    output = str(tmp_path) + "/out.bed.gz"
    snap.pp.make_fragment_file(bam, output, True, barcode_regex="(^[ATCG]+):")

    with gzip.open(bed, 'rt') as fl:
        expected = sorted(fl.readlines())

    with gzip.open(output, 'rt') as fl:
        actual = sorted(fl.readlines())
    
    assert expected == actual

@given(
    mat = arrays(
        np.float64, (50, 100),
        elements = {"allow_subnormal": False, "allow_nan": False, "allow_infinity": False, "min_value": 1, "max_value": 100},
    ),
)
@settings(deadline = None, suppress_health_check = [HealthCheck.function_scoped_fixture])
def test_reproducibility(mat):
    adata = ad.AnnData(X=csr_matrix(mat))
    embeddings = []
    for _ in range(3):
        embeddings.append(snap.tl.spectral(adata, features=None, random_state=0, inplace=False)[1])
    for x in embeddings:
        np.testing.assert_array_equal(x, embeddings[0])

    snap.tl.spectral(adata, features=None, random_state=0)
    knn = []
    for _ in range(3):
        knn.append(snap.pp.knn(adata, random_state=0, n_neighbors=25, inplace=False).todense())
    for x in knn:
        np.testing.assert_array_equal(x, knn[0])

    snap.pp.knn(adata, random_state=0, n_neighbors=25)
    leiden = []
    for _ in range(3):
        leiden.append(snap.tl.leiden(adata, random_state=0, resolution=1, n_iterations=10, inplace=False))
    for x in leiden:
        np.testing.assert_array_equal(x, leiden[0])

def read_bed(bed_file):
    with gzip.open(bed_file, 'rt') as f:
        return sorted([line.strip().split('\t')[:4] for line in f if line.startswith('chr')])

def test_import(datadir):
    test_files = [snap.datasets.pbmc500(downsample=True), str(datadir.join('test_clean.tsv.gz'))]

    for fl in test_files:
        data = snap.pp.import_data(
            fl,
            chrom_sizes=snap.genome.hg38,
            min_num_fragments=0,
            sorted_by_barcode=False,
        )

        data.obs['group'] = 'test_import'
        outputs = snap.ex.export_fragments(data, groupby="group", out_dir=str(datadir), suffix='.bed.gz')

        assert read_bed(list(outputs.values())[0]) == read_bed(fl)