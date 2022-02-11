import unittest
import numpy as np
import snapatac2._snapatac2 as sa
import snapatac2.tools._spectral as spectral
import random
from scipy import sparse

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

if __name__ == '__main__':
    unittest.main()
