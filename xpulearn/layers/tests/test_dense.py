import numpy as np

from xpulearn.layers import Dense


class TestDense(object):

    def test_init_params(self):
        num_units = 5
        input_dim = 3
        layer = Dense(num_units)
        layer.init_params(1, input_dim, 'float32')
        assert layer.params['W'].data.shape == (5, 3)
        assert layer.grads['W'].data.shape == (5, 3)
        assert layer.params['b'].data.shape == (5, 1)
        assert layer.grads['b'].data.shape == (5, 1)

    def test_forwardprop(self):
        np.random.seed(1)
        batch_size = 10
        num_units = 5
        input_dim = 3
        layer = Dense(num_units)
        layer.init_params(1, input_dim, 'float32')
        X = np.random.randn(batch_size, input_dim)
        got = layer.forwardprop(X)
        expected = np.array([
            [-0.994051, 2.491420, -1.688450, 1.494747, -0.469031],
            [0.239515, 2.443118, -0.588799, 2.538522, -1.212735],
            [0.851173, -1.310085, 1.201349, -0.001997, -0.118903],
            [1.589297, -1.041406, 1.676329, -0.792949, -0.136512],
            [-1.335992, -0.366129, -1.028475, -1.021384, 0.821308],
            [-0.422810, 1.616934, -0.917776, 0.823074, -0.329634],
            [-0.780226, 0.289959, -0.790212, -0.607942, 0.421267],
            [-2.314738, -1.974611, -1.305083, -2.284607, 1.757137],
            [1.462764, 0.882393, 0.945148, 1.112977, -0.956884],
            [-1.858222, 1.754981, -2.103090, 2.087128, -0.286993],
        ])
        np.testing.assert_array_almost_equal(got, expected, decimal=6)

    def test_backprop(self):
        np.random.seed(1)
        batch_size = 10
        num_units = 5
        input_dim = 3
        layer = Dense(num_units)
        layer.init_params(1, input_dim, 'float32')
        X = np.random.randn(batch_size, input_dim)
        dX = np.random.randn(batch_size, num_units)
        layer.forwardprop(X)
        # retval
        got = layer.backprop(dX)
        expected = np.array([
            [1.793061, -0.902373, 0.832279],
            [-0.794842, -0.040276, 0.629131],
            [1.078324, -0.215516, -1.248045],
            [-1.435316, 0.434247, -0.941786],
            [0.996186, 2.939824, -6.667177],
            [-1.506833, 1.212406, 0.430842],
            [-1.481623, 0.314852, 1.981542],
            [0.011099, 0.280583, 0.015564],
            [0.961135, -0.976752, 2.148008],
            [0.911804, -1.130095, 0.503705]
        ])
        np.testing.assert_array_almost_equal(got, expected, decimal=6)
        # W
        expected_W = np.array([
            [0.292488, 0.523424, 0.118045],
            [0.100129, 0.015908, 0.170990],
            [-0.271093, -0.222951, 0.045963],
            [-0.282474, -0.185886, 0.155331],
            [0.151023, 0.051978, -0.154625]
        ])
        np.testing.assert_array_almost_equal(
            layer.grads['W'].data, expected_W, decimal=6)
        # b
        expected_b = np.array([
            [-0.295089],
            [0.156521],
            [0.509844],
            [0.294329],
            [0.321440]
        ])
        np.testing.assert_array_almost_equal(
            layer.grads['b'].data, expected_b, decimal=6)
