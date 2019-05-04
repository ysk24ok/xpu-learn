import numpy as np

from xpulearn.layers import Activation, Dense
from xpulearn.layers.recurrent import RNNCell


class TestRNNCell(object):

    def test_forwardprop(self):
        batch_size = 10
        num_units = 5
        input_dim = 3
        np.random.seed(1)
        layers = {
            'Wh': Dense(num_units, weight_initializer='randn'),
            'Wx': Dense(num_units, weight_initializer='randn'),
            'A': Activation('tanh')
        }
        layers['Wh'].init_params('Wh', num_units, 'float32')
        layers['Wx'].init_params('Wx', input_dim, 'float32')
        X = np.random.randn(batch_size, input_dim)
        H = np.random.randn(batch_size, num_units)
        cell = RNNCell(layers)
        X_out, H_out = cell.forwardprop(X, H)
        expected = np.array([
            [-0.182840, 0.256156, 0.021512, 0.266472, -0.019181],
            [-0.353075, 0.326332, -0.195956, 0.156601, 0.330342],
            [-0.110181, 0.107622, -0.151312, -0.004074, 0.395894],
            [0.111699, -0.171953, 0.246262, 0.029211, 0.089950],
            [0.388796, -0.216016, 0.299333, 0.020340, -0.234369],
            [0.020676, -0.152718, 0.024064, -0.112336, 0.206854],
            [-0.279467, 0.399926, -0.451018, -0.004664, 0.249223],
            [-0.000103, 0.166221, -0.141924, 0.001657, 0.062013],
            [-0.293892, 0.173311, -0.318570, -0.070900, 0.322794],
            [-0.207429, 0.548085, -0.217438, 0.351065, 0.364682],
        ])
        np.testing.assert_array_almost_equal(X_out, expected)
        np.testing.assert_array_almost_equal(X_out, H_out)

    def test_backprop(self):
        np.set_printoptions(formatter={'float': '{:.6f}'.format})
        pass
