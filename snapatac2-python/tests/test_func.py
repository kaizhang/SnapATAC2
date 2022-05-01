import unittest
import numpy as np
import snapatac2._snapatac2 as sa
from snapatac2.tools._counts import _group_by
import random
from scipy import sparse
import pytest

def test_group_arr():
    labels = [1, 2, 3, 4]
    input = {}
    mats = []
    for i in labels:
        mat = np.random.randint(1, 10000, 1000).reshape(10, 100)
        mats.append(np.hstack((np.full((10, 1), i), mat)))
        input[i] = mat.sum(axis = 0)
    merged = np.vstack(mats)
    np.random.shuffle(merged)
    labels = merged[:, 0]
    mats = merged[:, 1:]
    result = _group_by(mats, labels)
    output = {}
    for (k, v) in result.items():
        output[k] = np.ravel(v.sum(axis = 0))

    for k, v in input.items():
        np.testing.assert_array_equal(v, output[k])

def test_group_csr():
    labels = [1, 2, 3, 4]
    input = {}
    mats = []
    for i in labels:
        mat = np.random.randint(1, 10000, 1000).reshape(10, 100)
        mats.append(np.hstack((np.full((10, 1), i), mat)))
        input[i] = mat.sum(axis = 0)
    merged = np.vstack(mats)
    np.random.shuffle(merged)
    labels = merged[:, 0]
    mats = sparse.csr_matrix(merged[:, 1:])
    result = _group_by(mats, labels)
    output = {}
    for (k, v) in result.items():
        output[k] = np.ravel(v.sum(axis = 0))

    for k, v in input.items():
        np.testing.assert_array_equal(v, output[k])

def repeat(times):
    def repeatHelper(f):
        def callHelper(*args):
            for i in range(0, times):
                f(*args)

        return callHelper
    return repeatHelper

class TestRegression(unittest.TestCase):
    @repeat(50)
    def test_regression(self):
        slope = random.uniform(0, 1)
        intersect = random.uniform(0, 1)
        data = 10000 * np.random.random_sample(100000)
        (slope_, intersect_) = sa.simple_lin_reg(map(lambda x: (x, x*slope + intersect), data))
        self.assertAlmostEqual(slope_, slope, msg="slope different")
        self.assertAlmostEqual(intersect_, intersect_, msg="intersect different")
