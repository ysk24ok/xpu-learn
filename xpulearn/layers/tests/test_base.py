from unittest import TestCase

import numpy as np

from xpulearn.layers import Dropout
from xpulearn.testing import approx_fprime


class TestDropout(TestCase):

    def test_gradient_checking(self):
        layer = Dropout(0.36)
        X = np.random.randn(5, 4)
        dout = np.random.randn(5, 4)
        layer.forwardprop(X, training=True)
        expected = approx_fprime(X, layer.forwardprop, True, False)
        got = layer.backprop(dout)
        np.testing.assert_array_almost_equal(got / dout, expected, decimal=6)
