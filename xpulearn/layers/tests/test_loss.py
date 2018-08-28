from unittest import TestCase

import numpy as np

from xpulearn.layers import Activation, Loss
from xpulearn.testing import approx_fprime


class TestMeanSquaredError(TestCase):

    def setUp(self):
        self.X = np.array([
            [85.44274866],
            [39.03089165],
            [63.4915475],
            [9.50850924]
        ])
        self.y = np.array([100, 40, 60, 20])

    def test_forwardprop(self):
        layer = Loss('mse')
        got = layer.forwardprop(self.X, self.y)
        expected = 41.889377
        np.testing.assert_almost_equal(got, expected, decimal=6)

    def test_backprop(self):
        layer = Loss('mse')
        got = layer.backprop(self.X, self.y)
        expected = np.array([
            [-3.639313],
            [-0.242277],
            [0.872887],
            [-2.622873]
        ])
        np.testing.assert_array_almost_equal(got, expected)

    def test_gradient_checking(self):
        layer = Loss('mse')
        X = np.random.rand(5, 1)
        y = np.random.rand(5,)
        expected = approx_fprime(X, layer.forwardprop, y)
        got = layer.backprop(X, y)
        np.testing.assert_array_almost_equal(got, expected, decimal=6)


class TestBinaryCrossentropy(TestCase):

    def setUp(self):
        self.X = np.array([
            [0.51429865],
            [0.48346224],
            [0.09769054],
            [0.43424723]
        ])
        self.y = np.array([1, 1, 0, 1])

    def test_forwardprop(self):
        layer = Loss('binary_crossentropy')
        got = layer.forwardprop(self.X, self.y)
        expected = 0.582168
        np.testing.assert_almost_equal(got, expected, decimal=6)

    def test_backprop(self):
        layer = Loss('binary_crossentropy')
        got = layer.backprop(self.X, self.y)
        expected = np.array([
            [-0.486099],
            [-0.517103],
            [0.277067],
            [-0.575709]
        ])
        np.testing.assert_array_almost_equal(got, expected)

    def test_gradient_checking(self):
        layer = Loss('binary_crossentropy')
        X = np.random.rand(5, 1)
        y = np.random.randint(0, high=2, size=(5,))
        expected = approx_fprime(X, layer.forwardprop, y)
        got = layer.backprop(X, y)
        np.testing.assert_array_almost_equal(got, expected, decimal=5)


class TestBinaryCrossentropyWithSigmoid(TestCase):

    def setUp(self):
        self.pre_activation = Activation('sigmoid')
        self.X = np.array([
            [0.05721021],
            [-0.06617518],
            [-2.22315276],
            [-0.26454317],
        ])
        self.y = np.array([1, 1, 0, 1])

    def test_forwardprop(self):
        layer = Loss('binary_crossentropy', self.pre_activation)
        got = layer.forwardprop(self.X, self.y)
        expected = 0.582168
        np.testing.assert_almost_equal(got, expected, decimal=6)

    def test_backprop(self):
        layer = Loss('binary_crossentropy', self.pre_activation)
        layer.forwardprop(self.X, self.y)
        got = layer.backprop(self.X, self.y)
        expected = np.array([
            [-0.121425],
            [-0.129134],
            [0.024423],
            [-0.141438]
        ])
        np.testing.assert_array_almost_equal(got, expected)


class TestCategoricalCrossentropy(TestCase):

    def setUp(self):
        self.X = np.array([
            [0.32944613, 0.20429963, 0.46625424],
            [0.07902391, 0.09397702, 0.82699907],
            [0.03494478, 0.19695906, 0.76809616],
            [0.18104065, 0.30151745, 0.5174419]
        ])
        self.Y = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])

    def test_forwardprop(self):
        layer = Loss('categorical_crossentropy')
        got = layer.forwardprop(self.X, self.Y)
        expected = 2.370872
        np.testing.assert_almost_equal(got, expected, decimal=6)

    def test_backprop(self):
        layer = Loss('categorical_crossentropy')
        got = layer.backprop(self.X, self.Y)
        expected = np.array([
            [-0.75884941, 0.31418862, 0.4683878],
            [0.27145113, -2.66022481, 1.44507894],
            [0.25905253, 0.31131663, -0.32548008],
            [-1.38090534, 0.35791875, 0.51807233]
        ])
        np.testing.assert_array_almost_equal(got, expected)

    def test_gradient_checking(self):
        layer = Loss('categorical_crossentropy')
        X = np.random.rand(5, 4)
        Y = np.random.randint(0, high=2, size=(5, 1))
        expected = approx_fprime(X, layer.forwardprop, Y)
        got = layer.backprop(X, Y)
        np.testing.assert_array_almost_equal(got, expected, decimal=5)


class TestCategoricalCrossentropyWithSoftmax(TestCase):

    def setUp(self):
        self.pre_activation = Activation('softmax')
        self.X = np.array([
            [0.05721021, -0.42061497, 0.40452842],
            [-0.06617518, 0.10712463, 2.28187792],
            [-2.22315276, -0.49392597, 0.86699306],
            [-0.26454317, 0.24556312, 0.78563249],
        ])
        self.Y = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])

    def test_forwardprop(self):
        layer = Loss('categorical_crossentropy', self.pre_activation)
        got = layer.forwardprop(self.X, self.Y)
        expected = 2.370872
        np.testing.assert_almost_equal(got, expected, decimal=6)

    def test_backprop(self):
        layer = Loss('categorical_crossentropy', self.pre_activation)
        layer.forwardprop(self.X, self.Y)
        got = layer.backprop(self.X, self.Y)
        expected = np.array([
            [-0.16763847, 0.05107491, 0.11656356],
            [0.01975598, -0.22650574, 0.20674977],
            [0.00873619, 0.04923977, -0.05797596],
            [-0.20473984, 0.07537936, 0.12936047]
        ])
        np.testing.assert_array_almost_equal(got, expected)
