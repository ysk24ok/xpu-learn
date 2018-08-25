from unittest import TestCase

import numpy as np

from xpulearn.layers import Activation
from xpulearn.testing import approx_fprime


class TestLinearActivation(TestCase):

    def test_gradient_checking(self):
        layer = Activation('linear')
        X = np.random.randn(5, 4)
        expected = approx_fprime(X, layer.activation.forwardprop)
        got = layer.activation.backprop(X)
        np.testing.assert_array_almost_equal(got, expected, decimal=6)


class TestSigmoidActivation(TestCase):

    def setUp(self):
        self.X = np.array([
            [0.05721021],
            [-0.06617518],
            [-2.22315276],
            [-0.26454317],
        ])

    def test_forwardprop(self):
        layer = Activation('sigmoid')
        got = layer.forwardprop(self.X)
        expected = np.array([
            [0.51429865],
            [0.48346224],
            [0.09769054],
            [0.43424723]
        ])
        np.testing.assert_array_almost_equal(got, expected)

    def test_backprop(self):
        layer = Activation('sigmoid')
        layer.forwardprop(self.X)
        dout = np.array([
            [-0.486099],
            [-0.517103],
            [0.277067],
            [-0.575709]
        ])
        got = layer.backprop(dout)
        expected = np.array([
            [-0.121425],
            [-0.129134],
            [0.024423],
            [-0.141438]
        ])
        np.testing.assert_array_almost_equal(got, expected)

    def test_gradient_checking(self):
        layer = Activation('sigmoid')
        X = np.random.randn(5, 4)
        expected = approx_fprime(X, layer.activation.forwardprop)
        got = layer.activation.backprop(X)
        np.testing.assert_array_almost_equal(got, expected, decimal=5)


class TestSoftmaxActivation(TestCase):

    def setUp(self):
        self.X = np.array([
            [0.05721021, -0.42061497, 0.40452842],
            [-0.06617518, 0.10712463, 2.28187792],
            [-2.22315276, -0.49392597, 0.86699306],
            [-0.26454317, 0.24556312, 0.78563249],
        ])

    def test_forwardprop(self):
        layer = Activation('softmax')
        got = layer.forwardprop(self.X)
        expected = np.array([
            [0.32944613, 0.20429963, 0.46625424],
            [0.07902391, 0.09397702, 0.82699907],
            [0.03494478, 0.19695906, 0.76809616],
            [0.18104065, 0.30151745, 0.5174419]
        ])
        np.testing.assert_array_almost_equal(got, expected)

    def test_backprop(self):
        layer = Activation('softmax')
        layer.forwardprop(self.X)
        dout = np.array([
            [-0.75884941, 0.31418862, 0.4683878],
            [0.27145113, -2.66022481, 1.44507894],
            [0.25905253, 0.31131663, -0.32548008],
            [-1.38090534, 0.35791875, 0.51807233]
        ])
        got = layer.backprop(dout)
        expected = np.array([
            [-0.16763847, 0.05107491, 0.11656356],
            [0.01975598, -0.22650574, 0.20674977],
            [0.00873619, 0.04923977, -0.05797596],
            [-0.20473984, 0.07537936, 0.12936047]
        ])
        np.testing.assert_array_almost_equal(got, expected)

    def test_gradient_checking(self):
        layer = Activation('softmax')
        X = np.random.randn(5, 4)
        expected = approx_fprime(X, layer.activation.forwardprop)
        got = layer.activation.backprop(X)
        np.testing.assert_array_almost_equal(got, expected, decimal=5)


class TestTanhActivation(TestCase):

    def test_gradient_checking(self):
        layer = Activation('tanh')
        X = np.random.randn(5, 4)
        expected = approx_fprime(X, layer.activation.forwardprop)
        got = layer.activation.backprop(X)
        np.testing.assert_array_almost_equal(got, expected, decimal=5)


class TestReLUActivation(TestCase):

    def test_gradient_checking(self):
        layer = Activation('relu')
        X = np.random.randn(5, 4)
        expected = approx_fprime(X, layer.activation.forwardprop)
        got = layer.activation.backprop(X)
        np.testing.assert_array_almost_equal(got, expected, decimal=5)
