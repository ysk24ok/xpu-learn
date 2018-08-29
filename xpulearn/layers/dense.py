import numpy as np

from .. import Parameter
from .base import Layer


class Dense(Layer):

    """Fully connected layer

    Parameters:
        W (numpy.ndarray):
            2D array of shape [#units of this layer, #units of input-side layer]
        b (numpy.ndarray):
            2D array of shape [#units of this layer, #units of input-side layer]
    """

    def __init__(self, num_units):
        """
        Arguments:
            num_units (int): #units of this layer
        """
        super(Dense, self).__init__()
        self.num_units = num_units
        self.params = {}
        self.grads = {}

    def init_params(self, layer_id, prev_layer_shape, dtype):
        # weight
        if 'W' not in self.params and 'W' not in self.grads:
            w_id = '{}_W'.format(layer_id)
            weight_shape = (self.num_units, *prev_layer_shape)
            self.params['W'] = Parameter(
                w_id, np.random.randn(*weight_shape).astype(dtype) / 10)
            self.grads['W'] = Parameter(
                w_id, np.zeros(weight_shape, dtype=dtype))
        # bias
        if 'b' not in self.params and 'b' not in self.grads:
            b_id = '{}_b'.format(layer_id)
            bias_shape = (self.num_units, 1)
            self.params['b'] = Parameter(
                b_id, np.zeros(bias_shape, dtype=dtype))
            self.grads['b'] = Parameter(
                b_id, np.zeros(bias_shape, dtype=dtype))

    def forwardprop(self, X_in):
        """
        Arguments:
            X_in (numpy.ndarray):
                2D array of shape [batch size, #units of input-side layer]
        Returns:
            (numpy.ndarray):
                2D array of shape [batch_size, #units of this layer]
        """
        self.X_in = X_in
        W = self.params['W'].data
        b = self.params['b'].data
        X_out = X_in @ W.T
        X_out += b.T
        return X_out

    def backprop(self, dout):
        """
        Arguments:
            dout (numpy.ndarray):
                2D array of shape [batch size, #units of this layer]
        Returns:
            (numpy.ndarray):
                2D array of shape [batch size, #units of input-side layer]
        """
        batch_size = self.X_in.shape[0]
        self.grads['W'].data[...] = dout.T @ self.X_in
        self.grads['W'].data /= batch_size
        self.grads['b'].data[...] = np.sum(dout.T, axis=1, keepdims=True)
        self.grads['b'].data /= batch_size
        return dout @ self.params['W'].data
